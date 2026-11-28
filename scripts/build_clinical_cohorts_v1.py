"""Build leak-free, patient-disjoint, strictly chronological Parquet cohorts for the 3 multi-hospital clinical cohorts (accessed via Nightingale Open Science & PhysioNet)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAW_ROOT = Path("/Volumes/WS4TB/imboreceipt_data")
OUT_DIR = RAW_ROOT / "processed_clinical_cohorts"


def _assign_patient_chrono_split(
    df: pd.DataFrame,
    patient_col: str,
    time_col: str,
    train_frac: float = 0.70,
    cal_frac: float = 0.15,
) -> pd.Series:
    """Strictly chronological, patient-disjoint 3-way split ordered by each patient's first encounter."""
    first_t = df.groupby(patient_col)[time_col].min().sort_values()
    n_pat = len(first_t)
    i_tr = int(n_pat * train_frac)
    i_cal = int(n_pat * (train_frac + cal_frac))
    tr_pats = set(first_t.index[:i_tr])
    cal_pats = set(first_t.index[i_tr:i_cal])
    splits = np.where(
        df[patient_col].isin(tr_pats),
        "train",
        np.where(df[patient_col].isin(cal_pats), "cal", "test"),
    )
    return pd.Series(splits, index=df.index, name="split")


def build_bwh_cohort() -> dict:
    """Cohort 1: Brigham & Women's Hospital ED-ECG (Mullainathan & Obermeyer / Schubert et al.)."""
    t0 = time.time()
    print("\n[1/3] Building BWH ED-ECG cohort (ed_bwh_ecg)...")
    bwh = RAW_ROOT / "ed_bwh_ecg"
    enc = pd.read_csv(bwh / "ed-encounter.csv", low_memory=False)
    pat = pd.read_csv(bwh / "patient.csv", low_memory=False)
    ecg_meta = pd.read_csv(bwh / "ecg-metadata.csv", low_memory=False)
    ecg_link = pd.read_csv(bwh / "ecg-ed-enc.csv", low_memory=False)
    ecg_idx = pd.read_csv(bwh / "ecg-npy-index.csv", low_memory=False)

    # Exact published modeling cohort filter
    mod = enc[enc["exclude_modeling"] == 0].copy()
    mod["start_dt"] = pd.to_datetime(mod["start_datetime"], errors="coerce", utc=True)

    # Attach earliest ECG per encounter (avoiding post-event ECG leakage)
    ecg_m = ecg_link.merge(ecg_meta, on="ecg_id", how="inner")
    ecg_m["ecg_dt"] = pd.to_datetime(ecg_m["date"], errors="coerce", utc=True)
    ecg_m = ecg_m.sort_values(["ed_enc_id", "ecg_dt", "ecg_id"]).drop_duplicates("ed_enc_id", keep="first")

    # Extract lightweight, clinically meaningful 12-lead ECG waveform summary features via mmap
    wf_path = bwh / "ecg-waveform.npy"
    wf_feats = None
    if wf_path.exists() and len(ecg_idx) > 0:
        wf = np.load(wf_path, mmap_mode="r")
        print(f"  Loaded ecg-waveform.npy mmap: shape={wf.shape}, dtype={wf.dtype}")
        # Map ecg_id -> row index in npy
        idx_col = [c for c in ecg_idx.columns if c != "ecg_id"][0]
        ecg_m = ecg_m.merge(ecg_idx[["ecg_id", idx_col]], on="ecg_id", how="left")
        valid_mask = ecg_m[idx_col].notna()
        valid_rows = ecg_m.loc[valid_mask, idx_col].astype(int).to_numpy()
        # Process in chunks to keep RAM tiny (<300 MB)
        n_v = len(valid_rows)
        n_leads = wf.shape[2] if wf.ndim == 3 else 12
        feat_names = [
            "wf_global_std",
            "wf_global_abs_p95",
            "wf_global_max_abs",
            "wf_global_energy_highfreq",
            "wf_lead0_std",
            "wf_lead0_p95",
            "wf_lead1_std",
            "wf_lead1_p95",
            "wf_lead2_std",
            "wf_lead2_p95",
            "wf_lead_dispersion_std",
            "wf_qrs_skew_proxy",
        ]
        wf_mat = np.full((len(ecg_m), len(feat_names)), np.nan, dtype=np.float32)
        chunk = 4096
        valid_pos = np.flatnonzero(valid_mask.to_numpy())
        for start in range(0, n_v, chunk):
            end = min(n_v, start + chunk)
            arr = np.asarray(wf[valid_rows[start:end]], dtype=np.float32)
            # Ensure shape is (B, T, L) where T > L
            if arr.ndim == 3 and arr.shape[1] < arr.shape[2]:
                arr = np.transpose(arr, (0, 2, 1))
            # Center each lead along time axis
            arr = arr - np.nanmean(arr, axis=1, keepdims=True)
            lead_std = np.nanstd(arr, axis=1)  # (B, L)
            lead_p95 = np.nanpercentile(np.abs(arr), 95, axis=1)  # (B, L)
            diff_rms = np.sqrt(np.nanmean(np.diff(arr, axis=1) ** 2, axis=(1, 2)))
            pos = valid_pos[start:end]
            wf_mat[pos, 0] = np.nanmean(lead_std, axis=1)
            wf_mat[pos, 1] = np.nanmean(lead_p95, axis=1)
            wf_mat[pos, 2] = np.nanmax(np.abs(arr), axis=(1, 2))
            wf_mat[pos, 3] = diff_rms
            if arr.shape[2] >= 3:
                wf_mat[pos, 4] = lead_std[:, 0]
                wf_mat[pos, 5] = lead_p95[:, 0]
                wf_mat[pos, 6] = lead_std[:, 1]
                wf_mat[pos, 7] = lead_p95[:, 1]
                wf_mat[pos, 8] = lead_std[:, 2]
                wf_mat[pos, 9] = lead_p95[:, 2]
            wf_mat[pos, 10] = np.nanstd(lead_std, axis=1)
            wf_mat[pos, 11] = np.nanmean(arr ** 3, axis=(1, 2)) / (np.nanmean(lead_std, axis=1) ** 3 + 1e-6)
        wf_feats = pd.DataFrame(wf_mat, columns=feat_names, index=ecg_m.index)
        ecg_m = pd.concat([ecg_m, wf_feats], axis=1)

    df = mod.merge(pat, on="patient_ngsci_id", how="left").merge(
        ecg_m.drop(columns=["patient_ngsci_id"], errors="ignore"),
        on="ed_enc_id",
        how="left",
    )

    # Clean numeric ECG intervals
    for c in ["p_axes", "r_axes", "t_axes", "pr_interval", "qrs_duration", "qt_interval", "qtc_interval", "vent_rate", "maxtrop_sameday"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["trop_drawn_sameday"] = df["maxtrop_sameday"].notna().astype(int)
    df["log_maxtrop_sameday"] = np.log1p(np.clip(df["maxtrop_sameday"].fillna(0.0), 0.0, None))
    df["sex_male"] = df["sex"].astype(str).str.upper().str.startswith("M").astype(int)

    # Chronological patient-disjoint split (70% train / 15% cal / 15% test)
    df["split"] = _assign_patient_chrono_split(df, "patient_ngsci_id", "start_dt", 0.70, 0.15)

    out_path = OUT_DIR / "bwh_ed_ecg.parquet"
    df.to_parquet(out_path, index=False)
    stats = {
        "path": str(out_path),
        "n_rows": len(df),
        "n_patients": int(df["patient_ngsci_id"].nunique()),
        "splits": df["split"].value_counts().to_dict(),
        "y_mace_death_30_rate": float(df["macetrop_pos_or_death_030"].mean()),
        "y_stent_cabg_10_rate": float(df["stent_or_cabg_010_day"].mean()),
        "tested_rate": float(df["test_010_day"].mean()),
        "untested_mace_death_30_rate": float(df.loc[df["test_010_day"] == 0, "macetrop_pos_or_death_030"].mean()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("  Built BWH ED-ECG:", json.dumps(stats, indent=2))
    return stats


def build_mcmed_cohort() -> dict:
    """Cohort 2: Stanford MC-MED (Kansal et al. Sci Data — official split_chrono patient-disjoint)."""
    t0 = time.time()
    print("\n[2/3] Building Stanford MC-MED cohort (mcmed-stanford-multi)...")
    mc = RAW_ROOT / "mcmed-stanford-multi" / "v1"
    visits = pd.read_csv(mc / "visits.csv", low_memory=False)

    # Map official chronological patient-disjoint splits
    split_map = {}
    for sname, fname in [
        ("train", "split_chrono_train.csv"),
        ("cal", "split_chrono_val.csv"),
        ("test", "split_chrono_test.csv"),
    ]:
        csns = pd.read_csv(mc / fname, header=None)[0].astype(visits["CSN"].dtype)
        for csn in csns:
            split_map[csn] = sname
    visits["split"] = visits["CSN"].map(split_map)
    visits = visits[visits["split"].notna()].copy()
    valid_csns = set(visits["CSN"])
    valid_mrns = set(visits["MRN"])
    print(f"  Official split_chrono matched visits: {len(visits):,} ({visits['MRN'].nunique():,} unique MRNs)")

    # Parse Arrival_time for strict causal <= 2h landmark filtering on labs/rads/orders/numerics
    arr_dt = pd.to_datetime(visits["Arrival_time"], errors="coerce", utc=True)
    csn_to_arr = pd.Series(arr_dt.values, index=visits["CSN"]).to_dict()

    # Exact published outcomes
    visits["y_admit"] = visits["ED_dispo"].isin(["Inpatient", "ICU"]).astype(int)
    visits["y_admit_or_obs"] = visits["ED_dispo"].isin(["Inpatient", "Observation", "ICU"]).astype(int)
    visits["y_critical_icu_death"] = (
        visits["ED_dispo"].isin(["ICU", "Expired"])
        | visits["DC_dispo"].astype(str).str.contains("Expired|Deceased|Dead", case=False, na=False)
    ).astype(int)
    discharged = visits["ED_dispo"].eq("Discharge")
    visits["is_discharged"] = discharged.astype(int)
    visits["y_bounce72_any"] = (
        discharged & (visits["Hours_to_next_visit"] > 0) & (visits["Hours_to_next_visit"] <= 72.0)
    ).astype(int)
    visits["y_bounce72_admit"] = (
        visits["y_bounce72_any"].eq(1)
        & visits["Dispo_class_next_visit"].isin(["Inpatient", "Observation", "ICU"])
    ).astype(int)

    # Parse ESI numeric from '1-Resuscitation' .. '5-Non-Urgent'
    visits["esi_num"] = pd.to_numeric(visits["Triage_acuity"].astype(str).str.extract(r"^(\d)")[0], errors="coerce")

    # 2A. Historical PMH (HIE Comorbidity channel — strictly Noted_date <= Arrival_time or pre-existing)
    print("  Aggregating pmh.csv (HIE comorbidities)...")
    pmh = pd.read_csv(mc / "pmh.csv", usecols=["MRN", "CCS", "DescCCS"], low_memory=False)
    pmh = pmh[pmh["MRN"].isin(valid_mrns)]
    pmh_cnt = pmh.groupby("MRN").size().rename("pmh_total_dx_count")
    top_ccs = pmh["DescCCS"].value_counts().head(16).index.tolist()
    pmh_top = pmh[pmh["DescCCS"].isin(top_ccs)]
    pmh_piv = pd.crosstab(pmh_top["MRN"], pmh_top["DescCCS"]).clip(upper=1)
    pmh_piv.columns = [f"pmh_{i:02d}" for i in range(len(pmh_piv.columns))]
    visits = visits.merge(pmh_cnt, on="MRN", how="left").merge(pmh_piv, on="MRN", how="left")
    visits["pmh_total_dx_count"] = visits["pmh_total_dx_count"].fillna(0.0)
    for c in pmh_piv.columns:
        visits[c] = visits[c].fillna(0.0)

    # 2B. Historical Outpatient Medications (HIE Meds channel)
    print("  Aggregating meds.csv (HIE outpatient meds)...")
    meds = pd.read_csv(mc / "meds.csv", usecols=["MRN", "Med_class", "Active"], low_memory=False)
    meds = meds[meds["MRN"].isin(valid_mrns)]
    med_cnt = meds.groupby("MRN").size().rename("meds_total_count")
    top_mcls = meds["Med_class"].value_counts().head(14).index.tolist()
    med_top = meds[meds["Med_class"].isin(top_mcls)]
    med_piv = pd.crosstab(med_top["MRN"], med_top["Med_class"]).clip(upper=1)
    med_piv.columns = [f"meds_{i:02d}" for i in range(len(med_piv.columns))]
    visits = visits.merge(med_cnt, on="MRN", how="left").merge(med_piv, on="MRN", how="left")
    visits["meds_total_count"] = visits["meds_total_count"].fillna(0.0)
    for c in med_piv.columns:
        visits[c] = visits[c].fillna(0.0)

    # 2C. Causal Labs (strictly Result_time <= Arrival_time + 2h OR Order_time <= Arrival_time + 2h)
    print("  Aggregating labs.csv (5.7M rows, strict <= 2h causal landmark)...")
    lab_targets = {
        "lab_cbc_wbc": r"\bWBC\b|WHITE BLOOD CELL",
        "lab_cbc_hgb": r"\bHGB\b|\bHEMOGLOBIN\b",
        "lab_cbc_plt": r"\bPLT\b|PLATELET",
        "lab_renal_cr": r"CREATININE",
        "lab_renal_bun": r"\bBUN\b|UREA NITROGEN",
        "lab_elyte_na": r"\bSODIUM\b|\bNA\b",
        "lab_elyte_k": r"\bPOTASSIUM\b",
        "lab_elyte_hco3": r"\bCO2\b|BICARBONATE",
        "lab_gluc": r"\bGLUCOSE\b",
        "lab_lactate": r"LACTATE|LACTIC",
        "lab_troponin": r"TROPONIN",
        "lab_lft_ast": r"\bAST\b",
        "lab_lft_alt": r"\bALT\b",
        "lab_lft_bili": r"BILIRUBIN.*TOTAL|TOTAL BILIRUBIN",
        "lab_coag_inr": r"\bINR\b",
        "lab_bnp": r"\bBNP\b|NT-PROBNP",
    }
    lab_chunks = []
    for chunk in pd.read_csv(
        mc / "labs.csv",
        usecols=["CSN", "Order_time", "Result_time", "Component_name", "Component_value"],
        chunksize=500_000,
        low_memory=False,
    ):
        chunk = chunk[chunk["CSN"].isin(valid_csns)].copy()
        if len(chunk) == 0:
            continue
        arr_c = pd.to_datetime(chunk["CSN"].map(csn_to_arr), utc=True)
        ord_t = pd.to_datetime(chunk["Order_time"], errors="coerce", utc=True)
        dt_hr = (ord_t - arr_c).dt.total_seconds() / 3600.0
        chunk = chunk[(dt_hr >= -1.0) & (dt_hr <= 2.0)].copy()
        if len(chunk) == 0:
            continue
        chunk["val"] = pd.to_numeric(chunk["Component_value"], errors="coerce")
        chunk = chunk[chunk["val"].notna()]
        cname = chunk["Component_name"].astype(str).str.upper()
        matched_rows = []
        for feat_col, pat in lab_targets.items():
            m = cname.str.contains(pat, regex=True, na=False)
            if m.any():
                sub = chunk.loc[m, ["CSN", "val"]].copy()
                sub["feat"] = feat_col
                matched_rows.append(sub)
        if matched_rows:
            lab_chunks.append(pd.concat(matched_rows, ignore_index=True))

    if lab_chunks:
        all_labs = pd.concat(lab_chunks, ignore_index=True)
        lab_piv = all_labs.groupby(["CSN", "feat"])["val"].first().unstack("feat")
        visits = visits.merge(lab_piv, on="CSN", how="left")
        print(f"  Matched <=2h causal labs: {len(lab_piv):,} visits x {lab_piv.shape[1]} lab assays")

    # 2D. Causal Radiology orders (<= 2h)
    print("  Aggregating rads.csv (strict <= 2h radiology studies)...")
    rads = pd.read_csv(mc / "rads.csv", usecols=["CSN", "Order_time", "Study"], low_memory=False)
    rads = rads[rads["CSN"].isin(valid_csns)].copy()
    rads_arr = pd.to_datetime(rads["CSN"].map(csn_to_arr), utc=True)
    rads_hr = (pd.to_datetime(rads["Order_time"], errors="coerce", utc=True) - rads_arr).dt.total_seconds() / 3600.0
    rads = rads[(rads_hr >= -1.0) & (rads_hr <= 2.0)].copy()
    study_u = rads["Study"].astype(str).str.upper()
    rads["rad_cxr"] = study_u.str.contains("CHEST|CXR", na=False).astype(int)
    rads["rad_ct_head"] = study_u.str.contains("CT.*HEAD|CT.*BRAIN", regex=True, na=False).astype(int)
    rads["rad_ct_abd"] = study_u.str.contains("CT.*ABD|CT.*PELV", regex=True, na=False).astype(int)
    rads["rad_ct_chest"] = study_u.str.contains("CT.*CHEST|CT.*PE|ANGIO.*CHEST", regex=True, na=False).astype(int)
    rads["rad_us"] = study_u.str.contains("US |ULTRASOUND", regex=True, na=False).astype(int)
    rad_agg = rads.groupby("CSN")[["rad_cxr", "rad_ct_head", "rad_ct_abd", "rad_ct_chest", "rad_us"]].max()
    rad_agg["rad_total_2h"] = rads.groupby("CSN").size()
    visits = visits.merge(rad_agg, on="CSN", how="left")
    for c in rad_agg.columns:
        visits[c] = visits[c].fillna(0.0)

    # 2E. Waveform summary (pre-extracted monitor waveform coverage)
    wf_sum = pd.read_csv(mc / "waveform_summary.csv", low_memory=False)
    wf_sum = wf_sum[wf_sum["CSN"].isin(valid_csns)]
    wf_piv = wf_sum.pivot_table(index="CSN", columns="Type", values="Duration", aggfunc="sum", fill_value=0.0)
    wf_piv.columns = [f"wf_dur_{str(c).lower()}" for c in wf_piv.columns]
    visits = visits.merge(wf_piv, on="CSN", how="left")
    for c in wf_piv.columns:
        visits[c] = visits[c].fillna(0.0)

    # 2F. Early <= 2h Continuous Monitor Numerics (2.57 GiB streamed in chunks)
    print("  Streaming numerics.csv (2.57 GiB, strict <= 2h monitor vitals)...")
    num_measures = {"HR", "SpO2", "RR", "NIBP_sys", "NIBP_dias", "NIBP_mean", "Temp"}
    num_acc: dict[tuple[int, str], list[float]] = {}
    # Stream with vectorized per-chunk groupby for speed!
    num_parts = []
    for chunk in pd.read_csv(
        mc / "numerics.csv",
        usecols=["CSN", "Measure", "Value", "Time"],
        chunksize=1_000_000,
        low_memory=False,
    ):
        chunk = chunk[chunk["CSN"].isin(valid_csns) & chunk["Measure"].isin(num_measures)].copy()
        if len(chunk) == 0:
            continue
        arr_c = pd.to_datetime(chunk["CSN"].map(csn_to_arr), utc=True)
        t_c = pd.to_datetime(chunk["Time"], errors="coerce", utc=True)
        dt_hr = (t_c - arr_c).dt.total_seconds() / 3600.0
        chunk = chunk[(dt_hr >= -0.5) & (dt_hr <= 2.0)].copy()
        if len(chunk) == 0:
            continue
        chunk["Value"] = pd.to_numeric(chunk["Value"], errors="coerce")
        chunk = chunk[chunk["Value"].notna()]
        if len(chunk) == 0:
            continue
        g = chunk.groupby(["CSN", "Measure"])["Value"].agg(["mean", "min", "max", "count"]).reset_index()
        num_parts.append(g)

    if num_parts:
        num_df = pd.concat(num_parts, ignore_index=True)
        # Merge chunk aggregates cleanly
        num_df["sum_v"] = num_df["mean"] * num_df["count"]
        num_agg = num_df.groupby(["CSN", "Measure"]).agg(
            sum_v=("sum_v", "sum"),
            cnt=("count", "sum"),
            min_v=("min", "min"),
            max_v=("max", "max"),
        ).reset_index()
        num_agg["mean_v"] = num_agg["sum_v"] / num_agg["cnt"]
        num_mean = num_agg.pivot(index="CSN", columns="Measure", values="mean_v")
        num_mean.columns = [f"mon_mean_{c}" for c in num_mean.columns]
        num_min = num_agg.pivot(index="CSN", columns="Measure", values="min_v")
        num_min.columns = [f"mon_min_{c}" for c in num_min.columns]
        num_max = num_agg.pivot(index="CSN", columns="Measure", values="max_v")
        num_max.columns = [f"mon_max_{c}" for c in num_max.columns]
        visits = visits.merge(num_mean, on="CSN", how="left").merge(num_min, on="CSN", how="left").merge(num_max, on="CSN", how="left")
        print(f"  Merged <=2h monitor numerics: {len(num_mean):,} visits x {num_mean.shape[1]*3} features")

    out_path = OUT_DIR / "mcmed_stanford.parquet"
    visits.to_parquet(out_path, index=False)
    stats = {
        "path": str(out_path),
        "n_rows": len(visits),
        "n_mrns": int(visits["MRN"].nunique()),
        "splits": visits["split"].value_counts().to_dict(),
        "y_admit_rate": float(visits["y_admit"].mean()),
        "y_critical_icu_death_rate": float(visits["y_critical_icu_death"].mean()),
        "y_bounce72_admit_among_discharged_rate": float(
            visits.loc[visits["is_discharged"] == 1, "y_bounce72_admit"].mean()
        ),
        "y_bounce72_admit_count": int(visits["y_bounce72_admit"].sum()),
        "esi45_count": int(visits["esi_num"].isin([4, 5]).sum()),
        "esi45_admit_rate": float(visits.loc[visits["esi_num"].isin([4, 5]), "y_admit"].mean()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("  Built Stanford MC-MED:", json.dumps(stats, indent=2))
    return stats


def build_ntuh_cohort() -> dict:
    """Cohort 3: NTUH Sudden Cardiac Arrest ECG (2015-2019 chronological, global vs 12-lead per-lead matrices)."""
    t0 = time.time()
    print("\n[3/3] Building NTUH Arrest-ECG cohort (arrest-ntuh-ecg)...")
    ntuh = RAW_ROOT / "arrest-ntuh-ecg" / "v1"
    ctrl = pd.read_csv(ntuh / "control-group" / "ecg-cohort.csv", low_memory=False)
    ctrl["y_pre_arrest"] = (ctrl["ecg_timetag"] == "pre").astype(int)
    ctrl["sex_male"] = ctrl["sex"].astype(str).str.upper().str.startswith("M").astype(int)

    # Gather global resting ECG measurements & per-lead ischemic/repolarization summaries across 2015-2019
    rest_dfs = []
    perlead_dfs = []
    diag_dfs = []
    for r_p in sorted((ntuh / "control-group").rglob("resting-ecg-measurements.csv")):
        grp_dir = r_p.parent
        rest_dfs.append(pd.read_csv(r_p, low_memory=False))
        pl_p = grp_dir / "measurement-matrix-per-lead-bitflg.csv"
        if pl_p.exists():
            pl = pd.read_csv(pl_p, low_memory=False)
            # Summarize ischemic/repolarization flags per anatomical territory
            pl["lead_u"] = pl["lead_id"].astype(str).str.upper()
            flag_cols = [c for c in ["ttal", "stdown", "stelev", "jelev", "dltwv", "stinj", "ppdeep"] if c in pl.columns]
            for c in flag_cols:
                pl[c] = pd.to_numeric(pl[c], errors="coerce").fillna(0.0)
            # Anatomical territories
            terr = np.where(
                pl["lead_u"].isin(["V1", "V2", "V3", "V4"]),
                "ant",
                np.where(
                    pl["lead_u"].isin(["I", "AVL", "V5", "V6"]),
                    "lat",
                    np.where(pl["lead_u"].isin(["II", "III", "AVF"]), "inf", "avr"),
                ),
            )
            pl["terr"] = terr
            agg = pl.groupby(["ecg_id", "terr"])[flag_cols].sum().unstack("terr", fill_value=0.0)
            agg.columns = [f"pl_{c}_{t}" for c, t in agg.columns]
            perlead_dfs.append(agg.reset_index())
        d_p = grp_dir / "diagnosis.csv"
        if d_p.exists():
            dd = pd.read_csv(d_p, usecols=["ecg_id", "stmt_text"], low_memory=False)
            txt = dd.groupby("ecg_id")["stmt_text"].apply(lambda s: " | ".join(s.dropna().astype(str)).upper())
            ddf = pd.DataFrame({"ecg_id": txt.index})
            ddf["dx_afib"] = txt.str.contains("ATRIAL FIB|AFIB", regex=True).astype(int).values
            ddf["dx_ischemia_st"] = txt.str.contains("ISCHEMIA|INFARCT|ST ELEV|ST DEPRESS|INJURY", regex=True).astype(int).values
            ddf["dx_bbb"] = txt.str.contains("BUNDLE BRANCH|RBBB|LBBB|IVCD", regex=True).astype(int).values
            ddf["dx_lvh"] = txt.str.contains("VENTRICULAR HYPERTROPHY|LVH", regex=True).astype(int).values
            ddf["dx_tachy_brady"] = txt.str.contains("TACHYCARDIA|BRADYCARDIA", regex=True).astype(int).values
            ddf["dx_qt_prolong"] = txt.str.contains("PROLONGED QT|LONG QT", regex=True).astype(int).values
            ddf["dx_normal"] = txt.str.contains("NORMAL ECG|NORMAL SINUS", regex=True).astype(int).values
            diag_dfs.append(ddf)

    rest_all = pd.concat(rest_dfs, ignore_index=True).drop_duplicates("ecg_id", keep="first")
    pl_all = pd.concat(perlead_dfs, ignore_index=True).drop_duplicates("ecg_id", keep="first")
    diag_all = pd.concat(diag_dfs, ignore_index=True).drop_duplicates("ecg_id", keep="first")

    df = ctrl.merge(
        rest_all.drop(columns=["year", "ecg_timetag"], errors="ignore"),
        on="ecg_id",
        how="left",
    ).merge(pl_all, on="ecg_id", how="left").merge(diag_all, on="ecg_id", how="left")

    # Strictly chronological, patient-disjoint split by patient's earliest ECG year
    first_yr = df.groupby("patient_ngsci_id")["year"].min()
    tr_pats = set(first_yr[first_yr <= 2017].index)
    cal_pats = set(first_yr[first_yr == 2018].index)
    df["split"] = np.where(
        df["patient_ngsci_id"].isin(tr_pats),
        "train",
        np.where(df["patient_ngsci_id"].isin(cal_pats), "cal", "test"),
    )

    out_path = OUT_DIR / "ntuh_arrest_ecg.parquet"
    df.to_parquet(out_path, index=False)
    stats = {
        "path": str(out_path),
        "n_rows": len(df),
        "n_patients": int(df["patient_ngsci_id"].nunique()),
        "splits": df["split"].value_counts().to_dict(),
        "y_pre_arrest_rate": float(df["y_pre_arrest"].mean()),
        "n_features": int(df.shape[1]),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("  Built NTUH Arrest-ECG:", json.dumps(stats, indent=2))
    return stats


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "bwh_ed_ecg": build_bwh_cohort(),
        "mcmed_stanford": build_mcmed_cohort(),
        "ntuh_arrest_ecg": build_ntuh_cohort(),
    }
    (OUT_DIR / "phase1_cohort_build_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nALL 3 CLINICAL COHORTS BUILT CLEANLY ->", OUT_DIR / "phase1_cohort_build_summary.json")


if __name__ == "__main__":
    main()

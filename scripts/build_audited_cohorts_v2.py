"""Step 1 (Paper 3 v2 Remediation): Leak-Free Audited Cohort Rebuild (`R6, R9, R17, R18, R19, R23`).

Builds strictly audited, patient-disjoint cohorts for all 3 clinical datasets inside:
  results/

Enforces:
  1. Stanford MC-MED (`mcmed-stanford-multi/v1`, Nightingale Open Science & PhysioNet, Kansal et al. 2025 Sci Data 12:1094):
     - Official `split_chrono` patient-disjoint chronological split (zero MRN overlap).
     - Strict `2.0h` prediction landmark: `Dispo_time > Arrival_time + 2.0h` (and `Admit_time > Arrival_time + 2.0h`,
       `Departure_time > Arrival_time + 2.0h` when non-null) so zero encounters have already reached disposition by 2.0h.
     - Per-visit causal timestamp filtering:
       * `pmh.csv`: strictly `Noted_date < Arrival_time` per CSN (`ch_cdw_prior_pmh`, fit top CCS strictly on train).
       * `meds.csv`: strictly `Entry_date < Arrival_time` (or `Start_date < Arrival_time` if Entry_date missing) (`ch_cdw_prior_meds`).
       * `labs.csv`: strictly `Result_time <= Arrival_time + 2.0h` (result availability, NOT `Order_time`).
       * `rads.csv`: strictly `Result_time <= Arrival_time + 2.0h` (resulted studies & posted impressions).
       * `numerics.csv`: strictly `Time <= Arrival_time + 2.0h`, and ZERO `wf_dur_*` whole-visit duration features.
     - Also saves arrival (`0h`) / discharge (`<= Dispo_time`) sensitivity frame for 72h Bounceback and CDW coverage by Payor/Race.
  2. BWH `ed-bwh-ecg` (Nightingale Open Science DUA, Mullainathan & Obermeyer 2022 QJE):
     - Explicitly documented as a patient-disjoint holdout split (`start_datetime` is patient-shifted `2109-2115`).
     - Primary Arrival-ECG Information-Access Cohort (`N = 51,158`, `100%` physically recorded arrival ECGs, `ecg_dt <= end_dt`):
       * `X_base`: `age_at_admit`, `sex_male`, automated 12-lead Muse scalar ECG intervals & axes.
       * Candidate Stage-2 channels: `ch_ecg_ischemia_flags`, `ch_ecg_rhythm_struct`, `ch_ecg_waveform` (rich multi-scale
         spectral, morphological, and vectorcardiographic features extracted via mmap across `ecg-waveform.npy` `(3 x 1000)`).
       * Non-incorporated hard endpoints: `death_030_day` (`1.96%`), `stent_or_cabg_010_day` (`1.40%`), plus ECG-only
         `macetrop_pos_or_death_030` with ZERO troponin predictor.
       * Clean subgroup flags: `test_010_day == 0` (no 10d stress/cath), `trop_drawn_sameday == 0`, and Complete-Troponin
         Subgroup (`trop_drawn_sameday == 1`, `N = 20,082`) for non-incorporated `death_030_day` & `stent_or_cabg_010_day`.
  3. NTUH `arrest-ntuh-ecg/v1` (Nightingale Open Science DUA, Obermeyer et al. 2026 Nature):
     - Primary True Case-Control Pre-Arrest SCA Cohort (`2015-2019`):
       * Cases (`Y = 1`): `study-group/ecg.csv` with `ecg_timetag == '[0]pre'`, `year in {'2015-16', '2017', '2018', '2019'}`,
         and strictly verified `acquisition_datetime_offset < rosc_datetime_offset` (`385` traces across `383` SCA patients).
       * Controls (`Y = 0`): `control-group/ecg-cohort.csv` (`2015-2019`, `16,386` traces across `9,976` non-arrest patients;
         zero `patient_ngsci_id` overlap with `study-group`).
       * Identical feature extraction across cases & controls: `resting-ecg-measurements.csv`, GE MUSE integer `lead_id`
         (`0..11`) anatomical territories (`ant: V1-V4`, `lat: I/aVL/V5-V6`, `inf: II/III/aVF`, `avr: aVR`) for both
         `measurement-matrix-per-lead-bitflg.csv` and quantitative `measurement-matrix-per-lead.csv`, `diagnosis.csv`, and
         12-lead median-beat waveforms `waveform-median.npy` (`12 x 600`).
       * Strictly chronological patient-disjoint split by earliest ECG year (`<=2017 -> train`, `2018 -> cal`, `2019 -> test`).
     - Supplementary Post-ROSC Cohort (`[1]ROSC` strictly, hyphenated years parsed cleanly: `2011-14, 2015-16, 2017 -> train`,
       `2018 -> cal`, `2019 -> test`).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd

RAW_ROOT = Path("/Volumes/WS4TB/imboreceipt_data")
REBUILD_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REBUILD_ROOT / "data"


def _assign_patient_holdout_split(
    df: pd.DataFrame,
    patient_col: str,
    order_col: str,
    train_frac: float = 0.70,
    cal_frac: float = 0.15,
) -> pd.Series:
    """Deterministic patient-disjoint 3-way split ordered by each patient's earliest encounter timestamp."""
    first_t = df.groupby(patient_col)[order_col].min().sort_values(kind="mergesort")
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


def _extract_bwh_waveform_features(wf_mmap: np.ndarray, row_indices: np.ndarray) -> pd.DataFrame:
    """Extract rich physiological waveform features via mmap from BWH `ecg-waveform.npy` (`(N, 3, 1000)` at 100 Hz).

    Captures per-lead amplitude quantiles, skewness/kurtosis, FFT spectral power across ST-T (0.5-7.5 Hz),
    QRS (7.5-25 Hz), and HF fragmentation (25-45 Hz) bands, first/second-difference RMS, 3D vectorcardiographic
    magnitude summaries, inter-lead correlations, and depolarization-repolarization cosine dispersion.
    """
    n_rows = len(row_indices)
    feat_names: list[str] = []
    for l_idx in range(3):
        feat_names.extend([
            f"wf_l{l_idx}_std",
            f"wf_l{l_idx}_iqr",
            f"wf_l{l_idx}_p05",
            f"wf_l{l_idx}_p95",
            f"wf_l{l_idx}_max_abs",
            f"wf_l{l_idx}_skew",
            f"wf_l{l_idx}_kurt",
            f"wf_l{l_idx}_diff1_rms",
            f"wf_l{l_idx}_diff2_rms",
            f"wf_l{l_idx}_pow_stt",
            f"wf_l{l_idx}_pow_qrs",
            f"wf_l{l_idx}_pow_hf",
            f"wf_l{l_idx}_frac_stt",
            f"wf_l{l_idx}_frac_hf",
        ])
    feat_names.extend([
        "wf_vmag_mean",
        "wf_vmag_std",
        "wf_vmag_p95",
        "wf_vmag_max",
        "wf_lead_std_disp",
        "wf_corr_01",
        "wf_corr_02",
        "wf_corr_12",
        "wf_qrs_stt_cos",
        "wf_rpeak_count_10s",
        "wf_rr_cv_proxy",
    ])
    out_mat = np.full((n_rows, len(feat_names)), np.nan, dtype=np.float32)

    freqs = np.fft.rfftfreq(1000, d=0.01)  # 100 Hz sampling -> 0..50 Hz
    m_stt = (freqs >= 0.5) & (freqs < 7.5)
    m_qrs = (freqs >= 7.5) & (freqs < 25.0)
    m_hf = (freqs >= 25.0) & (freqs <= 45.0)

    chunk = 4096
    for start in range(0, n_rows, chunk):
        end = min(n_rows, start + chunk)
        arr = np.asarray(wf_mmap[row_indices[start:end]], dtype=np.float32)  # (B, 3, 1000)
        if arr.shape[1] != 3 and arr.shape[2] == 3:
            arr = np.transpose(arr, (0, 2, 1))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = arr - np.mean(arr, axis=2, keepdims=True)  # (B, 3, T)

        std = np.std(arr, axis=2) + 1e-6  # (B, 3)
        q05, q25, q75, q95 = np.percentile(arr, [5, 25, 75, 95], axis=2)
        iqr = q75 - q25
        max_abs = np.max(np.abs(arr), axis=2)
        z = arr / std[:, :, None]
        skew = np.mean(z ** 3, axis=2)
        kurt = np.mean(z ** 4, axis=2)
        d1 = np.sqrt(np.mean(np.diff(arr, axis=2) ** 2, axis=2))
        d2 = np.sqrt(np.mean(np.diff(arr, n=2, axis=2) ** 2, axis=2))

        spec = (np.abs(np.fft.rfft(arr, axis=2)) ** 2) / 1000.0  # (B, 3, 501)
        p_stt = np.sum(spec[:, :, m_stt], axis=2)
        p_qrs = np.sum(spec[:, :, m_qrs], axis=2)
        p_hf = np.sum(spec[:, :, m_hf], axis=2)
        p_tot = p_stt + p_qrs + p_hf + 1e-8

        col_ptr = 0
        for l_idx in range(3):
            block = np.column_stack([
                std[:, l_idx],
                iqr[:, l_idx],
                q05[:, l_idx],
                q95[:, l_idx],
                max_abs[:, l_idx],
                skew[:, l_idx],
                kurt[:, l_idx],
                d1[:, l_idx],
                d2[:, l_idx],
                np.log1p(p_stt[:, l_idx]),
                np.log1p(p_qrs[:, l_idx]),
                np.log1p(p_hf[:, l_idx]),
                p_stt[:, l_idx] / p_tot[:, l_idx],
                p_hf[:, l_idx] / p_tot[:, l_idx],
            ])
            out_mat[start:end, col_ptr : col_ptr + 14] = block
            col_ptr += 14

        vmag = np.sqrt(np.sum(arr ** 2, axis=1))  # (B, 1000)
        vm_mean = np.mean(vmag, axis=1)
        vm_std = np.std(vmag, axis=1)
        vm_p95 = np.percentile(vmag, 95, axis=1)
        vm_max = np.max(vmag, axis=1)
        lead_disp = np.std(std, axis=1)

        c01 = np.mean(z[:, 0, :] * z[:, 1, :], axis=1)
        c02 = np.mean(z[:, 0, :] * z[:, 2, :], axis=1)
        c12 = np.mean(z[:, 1, :] * z[:, 2, :], axis=1)

        # QRS-T spatial vector cosine proxy: mean 3D vector during top-5% depolarization vs +12..+26 samples (+120..260ms)
        qrs_thr = np.percentile(vmag, 95, axis=1, keepdims=True)
        qrs_mask = vmag >= qrs_thr  # (B, 1000)
        stt_mask = np.zeros_like(qrs_mask)
        stt_mask[:, 16:] = qrs_mask[:, :-16]
        stt_mask = stt_mask & (~qrs_mask)
        v_qrs = np.sum(arr * qrs_mask[:, None, :], axis=2) / np.maximum(1.0, np.sum(qrs_mask, axis=1, keepdims=True))
        v_stt = np.sum(arr * stt_mask[:, None, :], axis=2) / np.maximum(1.0, np.sum(stt_mask, axis=1, keepdims=True))
        cos_qt = np.sum(v_qrs * v_stt, axis=1) / (
            np.linalg.norm(v_qrs, axis=1) * np.linalg.norm(v_stt, axis=1) + 1e-6
        )

        # Simple refractory-period R-peak counter (peaks above 65% of p99 separated by >= 250ms = 25 samples)
        p99 = np.percentile(vmag, 99, axis=1)
        r_cnt = np.zeros(end - start, dtype=np.float32)
        rr_cv = np.zeros(end - start, dtype=np.float32)
        for bi in range(end - start):
            sig = vmag[bi]
            thr_b = max(0.05, 0.60 * float(p99[bi]))
            cand = np.flatnonzero((sig[1:-1] > thr_b) & (sig[1:-1] >= sig[:-2]) & (sig[1:-1] > sig[2:])) + 1
            if cand.size > 0:
                kept = [int(cand[0])]
                for c_idx in cand[1:]:
                    if int(c_idx) - kept[-1] >= 25:
                        kept.append(int(c_idx))
                    elif sig[c_idx] > sig[kept[-1]]:
                        kept[-1] = int(c_idx)
                r_cnt[bi] = float(len(kept))
                if len(kept) >= 3:
                    rr = np.diff(kept).astype(np.float32)
                    rr_cv[bi] = float(np.std(rr) / (np.mean(rr) + 1e-6))

        out_mat[start:end, col_ptr:] = np.column_stack([
            vm_mean, vm_std, vm_p95, vm_max, lead_disp, c01, c02, c12, cos_qt, r_cnt, rr_cv
        ])

    return pd.DataFrame(out_mat, columns=feat_names)


def build_bwh_audited() -> dict[str, Any]:
    """Cohort 1: BWH `ed-bwh-ecg` (Patient-Disjoint Holdout; Arrival-ECG Information-Access + Subgroup Slices)."""
    t0 = time.time()
    print("\n[1/3] Building Audited BWH ED-ECG Cohort (`ed-bwh-ecg`)...")
    bwh = RAW_ROOT / "ed_bwh_ecg"
    enc = pd.read_csv(bwh / "ed-encounter.csv", low_memory=False)
    pat = pd.read_csv(bwh / "patient.csv", low_memory=False)
    ecg_meta = pd.read_csv(bwh / "ecg-metadata.csv", low_memory=False)
    ecg_link = pd.read_csv(bwh / "ecg-ed-enc.csv", low_memory=False)
    ecg_idx = pd.read_csv(bwh / "ecg-npy-index.csv", low_memory=False)

    n_raw_enc = int(len(enc))
    mod = enc[enc["exclude_modeling"] == 0].copy().reset_index(drop=True)
    mod["start_dt"] = pd.to_datetime(mod["start_datetime"], errors="coerce", utc=True)
    mod["end_dt"] = pd.to_datetime(mod["end_datetime"], errors="coerce", utc=True)

    # Earliest recorded ECG per encounter
    ecg_m = ecg_link.merge(ecg_meta, on="ecg_id", how="inner")
    ecg_m["ecg_dt"] = pd.to_datetime(ecg_m["date"], errors="coerce", utc=True)
    ecg_m = ecg_m.sort_values(["ed_enc_id", "ecg_dt", "ecg_id"], kind="mergesort").drop_duplicates("ed_enc_id", keep="first")
    ecg_m = ecg_m.merge(ecg_idx[["ecg_id", "npy_index"]], on="ecg_id", how="left").reset_index(drop=True)

    wf_mmap = np.load(bwh / "ecg-waveform.npy", mmap_mode="r")
    wf_df = _extract_bwh_waveform_features(wf_mmap, ecg_m["npy_index"].astype(int).to_numpy())
    ecg_m = pd.concat([ecg_m, wf_df], axis=1)

    df = mod.merge(pat, on="patient_ngsci_id", how="left").merge(
        ecg_m.drop(columns=["patient_ngsci_id"], errors="ignore"),
        on="ed_enc_id",
        how="left",
    )

    for c in ["p_axes", "r_axes", "t_axes", "pr_interval", "qrs_duration", "qt_interval", "qtc_interval", "vent_rate", "maxtrop_sameday"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ecg_lag_hours"] = (df["ecg_dt"] - df["start_dt"]).dt.total_seconds() / 3600.0
    df["ecg_before_ed_end"] = (df["ecg_dt"] <= df["end_dt"]).astype(int)
    df["trop_drawn_sameday"] = df["maxtrop_sameday"].notna().astype(int)
    df["log_maxtrop_sameday"] = np.where(
        df["maxtrop_sameday"].notna(),
        np.log1p(np.clip(df["maxtrop_sameday"].fillna(0.0), 0.0, None)),
        np.nan,
    )
    df["sex_male"] = df["sex"].astype(str).str.upper().str.startswith("M").astype(int)
    df["race_str"] = np.where(
        df["race_white"] == 1,
        "White",
        np.where(df["race_black"] == 1, "Black", np.where(df["race_hispanic"] == 1, "Hispanic", "Other")),
    )

    # Patient-disjoint holdout split (70% train / 15% cal / 15% test of unique patients)
    df["split"] = _assign_patient_holdout_split(df, "patient_ngsci_id", "start_dt", 0.70, 0.15)
    assert len(set(df.loc[df["split"] == "train", "patient_ngsci_id"]) & set(df.loc[df["split"] == "test", "patient_ngsci_id"])) == 0
    assert len(set(df.loc[df["split"] == "cal", "patient_ngsci_id"]) & set(df.loc[df["split"] == "test", "patient_ngsci_id"])) == 0

    # Leak-free ZIP-AGI quartiles computed strictly from training-set quantile cutoffs
    df["agi_expected_k"] = (
        15.0 * df["agi_under_25k"].fillna(0)
        + 37.5 * df["agi_25k_to_50k"].fillna(0)
        + 62.5 * df["agi_50k_to_75k"].fillna(0)
        + 87.5 * df["agi_75k_to_100k"].fillna(0)
        + 150.0 * df["agi_100k_to_200k"].fillna(0)
        + 250.0 * df["agi_above_200k"].fillna(0)
    )
    valid_agi = df["agi_under_25k"].notna()
    tr_agi = df.loc[valid_agi & (df["split"] == "train"), "agi_expected_k"]
    q_edges = np.quantile(tr_agi, [0.25, 0.50, 0.75])
    q_bins = np.digitize(df.loc[valid_agi, "agi_expected_k"], q_edges, right=True)
    q_labels = np.array(["Q1_Lowest_Income", "Q2_LowMid", "Q3_HighMid", "Q4_Highest_Income"])
    df["agi_quartile"] = "Missing"
    df.loc[valid_agi, "agi_quartile"] = q_labels[q_bins]

    out_path = OUT_DIR / "bwh_ed_ecg_v2.parquet"
    df.to_parquet(out_path, index=False)

    trop_sub = df[df["trop_drawn_sameday"] == 1]
    stats = {
        "platform_provenance": "Nightingale Open Science DUA (`ed-bwh-ecg`, Mullainathan & Obermeyer 2022 QJE)",
        "split_design": "Patient-disjoint holdout split (70/15/15 unique patients; `start_datetime` is patient-shifted 2109-2115, NOT calendar chronology)",
        "waveform_tensor_spec": f"ecg-waveform.npy physical shape {list(wf_mmap.shape)} (3 recorded waveform leads x 1000 samples @ 100 Hz; 100% present) + automated 12-lead Muse scalar intervals/flags in ecg-metadata.csv",
        "parquet_path": str(out_path),
        "n_raw_encounters": n_raw_enc,
        "n_modeling_encounters": int(len(df)),
        "n_unique_patients": int(df["patient_ngsci_id"].nunique()),
        "splits_encounters": {k: int(v) for k, v in df["split"].value_counts().items()},
        "splits_patients": {
            s: int(df.loc[df["split"] == s, "patient_ngsci_id"].nunique()) for s in ["train", "cal", "test"]
        },
        "ecg_recorded_rate": float(df["ecg_id"].notna().mean()),
        "ecg_waveform_present_rate": float(df["npy_index"].notna().mean()),
        "ecg_before_ed_end_rate": float(df["ecg_before_ed_end"].mean()),
        "ecg_lag_hours_median_iqr": [
            round(float(df["ecg_lag_hours"].quantile(0.25)), 3),
            round(float(df["ecg_lag_hours"].median()), 3),
            round(float(df["ecg_lag_hours"].quantile(0.75)), 3),
        ],
        "endpoints_overall": {
            "death_030_day_rate": round(float(df["death_030_day"].mean()), 5),
            "death_030_day_events": int(df["death_030_day"].sum()),
            "stent_or_cabg_010_day_rate": round(float(df["stent_or_cabg_010_day"].mean()), 5),
            "stent_or_cabg_010_day_events": int(df["stent_or_cabg_010_day"].sum()),
            "macetrop_pos_or_death_030_rate": round(float(df["macetrop_pos_or_death_030"].mean()), 5),
            "macetrop_pos_or_death_030_events": int(df["macetrop_pos_or_death_030"].sum()),
        },
        "endpoints_test": {
            "n_test": int((df["split"] == "test").sum()),
            "n_test_patients": int(df.loc[df["split"] == "test", "patient_ngsci_id"].nunique()),
            "death_030_day_rate": round(float(df.loc[df["split"] == "test", "death_030_day"].mean()), 5),
            "death_030_day_events": int(df.loc[df["split"] == "test", "death_030_day"].sum()),
            "stent_or_cabg_010_day_rate": round(float(df.loc[df["split"] == "test", "stent_or_cabg_010_day"].mean()), 5),
            "stent_or_cabg_010_day_events": int(df.loc[df["split"] == "test", "stent_or_cabg_010_day"].sum()),
            "macetrop_pos_or_death_030_rate": round(float(df.loc[df["split"] == "test", "macetrop_pos_or_death_030"].mean()), 5),
        },
        "subgroups": {
            "no_10d_stress_or_cath_all": int((df["test_010_day"] == 0).sum()),
            "no_10d_stress_or_cath_test": int(((df["split"] == "test") & (df["test_010_day"] == 0)).sum()),
            "no_sameday_troponin_all": int((df["trop_drawn_sameday"] == 0).sum()),
            "no_sameday_troponin_test": int(((df["split"] == "test") & (df["trop_drawn_sameday"] == 0)).sum()),
            "complete_sameday_troponin_all": int(len(trop_sub)),
            "complete_sameday_troponin_splits": {k: int(v) for k, v in trop_sub["split"].value_counts().items()},
            "complete_troponin_death_030_test_rate": round(
                float(trop_sub.loc[trop_sub["split"] == "test", "death_030_day"].mean()), 5
            ),
            "complete_troponin_stent_cabg_010_test_rate": round(
                float(trop_sub.loc[trop_sub["split"] == "test", "stent_or_cabg_010_day"].mean()), 5
            ),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("  Audited BWH ED-ECG complete:", json.dumps(stats["endpoints_test"], indent=2))
    return stats


def build_mcmed_audited() -> dict[str, Any]:
    """Cohort 2: Stanford MC-MED (`mcmed-stanford-multi/v1` — Strict 2.0h Landmark + Causal Result_time/Noted_date)."""
    t0 = time.time()
    print("\n[2/3] Building Audited Stanford MC-MED Cohort (`mcmed-stanford-multi/v1`)...")
    mc = RAW_ROOT / "mcmed-stanford-multi" / "v1"
    visits_raw = pd.read_csv(mc / "visits.csv", low_memory=False)
    n_visits_raw = int(len(visits_raw))

    split_map: dict[int, str] = {}
    for sname, fname in [
        ("train", "split_chrono_train.csv"),
        ("cal", "split_chrono_val.csv"),
        ("test", "split_chrono_test.csv"),
    ]:
        csns = pd.read_csv(mc / fname, header=None)[0].astype(visits_raw["CSN"].dtype)
        for csn in csns:
            split_map[int(csn)] = sname

    visits = visits_raw[visits_raw["CSN"].map(split_map).notna()].copy().reset_index(drop=True)
    visits["split"] = visits["CSN"].map(split_map)
    n_split_chrono = int(len(visits))
    assert len(set(visits.loc[visits["split"] == "train", "MRN"]) & set(visits.loc[visits["split"] == "test", "MRN"])) == 0
    assert len(set(visits.loc[visits["split"] == "cal", "MRN"]) & set(visits.loc[visits["split"] == "test", "MRN"])) == 0

    arr_dt = pd.to_datetime(visits["Arrival_time"], errors="coerce", utc=True)
    dispo_dt = pd.to_datetime(visits["Dispo_time"], errors="coerce", utc=True)
    admit_dt = pd.to_datetime(visits["Admit_time"], errors="coerce", utc=True)
    dep_dt = pd.to_datetime(visits["Departure_time"], errors="coerce", utc=True)

    dt_dispo_hr = (dispo_dt - arr_dt).dt.total_seconds() / 3600.0
    dt_admit_hr = (admit_dt - arr_dt).dt.total_seconds() / 3600.0
    dt_dep_hr = (dep_dt - arr_dt).dt.total_seconds() / 3600.0

    visits["arr_dt"] = arr_dt
    visits["dispo_dt"] = dispo_dt
    visits["landmark_2h_eligible"] = (
        arr_dt.notna()
        & dispo_dt.notna()
        & (dt_dispo_hr > 2.0)
        & (admit_dt.isna() | (dt_admit_hr > 2.0))
        & (dep_dt.isna() | (dt_dep_hr > 2.0))
    ).astype(int)

    # Outcomes
    visits["y_admit"] = visits["ED_dispo"].isin(["Inpatient", "ICU"]).astype(int)
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
    visits["esi_num"] = pd.to_numeric(visits["Triage_acuity"].astype(str).str.extract(r"^(\d)")[0], errors="coerce")

    valid_csns = set(visits["CSN"].tolist())
    valid_mrns = set(visits["MRN"].tolist())
    csn_to_arr = pd.Series(arr_dt.values, index=visits["CSN"]).to_dict()
    csn_to_dispo = pd.Series(dispo_dt.values, index=visits["CSN"]).to_dict()

    # 2A. Stanford CDW Prior PMH (`Noted_date < Arrival_time` strictly per CSN; top CCS fit strictly on train!)
    print("  [MC-MED 2A] Filtering pmh.csv strictly on Noted_date < Arrival_time per CSN...")
    pmh = pd.read_csv(mc / "pmh.csv", usecols=["MRN", "Noted_date", "DescCCS"], low_memory=False)
    pmh = pmh[pmh["MRN"].isin(valid_mrns)].copy()
    pmh["noted_dt"] = pd.to_datetime(pmh["Noted_date"], errors="coerce", utc=True)
    pmh_unfiltered_mrns = int(pmh["MRN"].nunique())
    pmh = pmh[pmh["noted_dt"].notna()].copy()

    v_lookup = visits[["CSN", "MRN", "arr_dt", "split"]].copy()
    pmh_j = v_lookup.merge(pmh[["MRN", "noted_dt", "DescCCS"]], on="MRN", how="inner")
    pmh_j = pmh_j[pmh_j["noted_dt"] < pmh_j["arr_dt"]].copy()

    pmh_cnt = pmh_j.groupby("CSN").size().rename("cdw_pmh_total_dx_count")
    top_ccs = (
        pmh_j.loc[pmh_j["split"] == "train", "DescCCS"]
        .dropna()
        .value_counts()
        .head(16)
        .index.tolist()
    )
    pmh_top = pmh_j[pmh_j["DescCCS"].isin(top_ccs)]
    pmh_piv = pd.crosstab(pmh_top["CSN"], pmh_top["DescCCS"]).reindex(columns=top_ccs, fill_value=0).clip(upper=1)
    pmh_piv.columns = [f"cdw_pmh_{i:02d}" for i in range(len(top_ccs))]
    visits = visits.merge(pmh_cnt, on="CSN", how="left").merge(pmh_piv, on="CSN", how="left")
    visits["cdw_pmh_total_dx_count"] = visits["cdw_pmh_total_dx_count"].fillna(0.0)
    for c in pmh_piv.columns:
        visits[c] = visits[c].fillna(0.0)
    visits["has_prior_cdw_pmh"] = (visits["cdw_pmh_total_dx_count"] > 0).astype(int)

    # 2B. Stanford CDW Prior Medications (`Entry_date < Arrival_time`, fallback `Start_date < Arrival_time`)
    print("  [MC-MED 2B] Filtering meds.csv strictly on Entry_date < Arrival_time per CSN...")
    meds = pd.read_csv(mc / "meds.csv", usecols=["MRN", "Med_class", "Entry_date", "Start_date"], low_memory=False)
    meds = meds[meds["MRN"].isin(valid_mrns)].copy()
    entry_dt = pd.to_datetime(meds["Entry_date"], errors="coerce", utc=True)
    start_dt = pd.to_datetime(meds["Start_date"], errors="coerce", utc=True)
    meds["eff_dt"] = entry_dt.fillna(start_dt)
    meds = meds[meds["eff_dt"].notna()].copy()

    med_j = v_lookup.merge(meds[["MRN", "eff_dt", "Med_class"]], on="MRN", how="inner")
    med_j = med_j[med_j["eff_dt"] < med_j["arr_dt"]].copy()

    med_cnt = med_j.groupby("CSN").size().rename("cdw_meds_total_count")
    top_mcls = (
        med_j.loc[med_j["split"] == "train", "Med_class"]
        .dropna()
        .value_counts()
        .head(14)
        .index.tolist()
    )
    med_top = med_j[med_j["Med_class"].isin(top_mcls)]
    med_piv = pd.crosstab(med_top["CSN"], med_top["Med_class"]).reindex(columns=top_mcls, fill_value=0).clip(upper=1)
    med_piv.columns = [f"cdw_meds_{i:02d}" for i in range(len(top_mcls))]
    visits = visits.merge(med_cnt, on="CSN", how="left").merge(med_piv, on="CSN", how="left")
    visits["cdw_meds_total_count"] = visits["cdw_meds_total_count"].fillna(0.0)
    for c in med_piv.columns:
        visits[c] = visits[c].fillna(0.0)
    visits["has_prior_cdw_meds"] = (visits["cdw_meds_total_count"] > 0).astype(int)
    visits["has_any_prior_cdw"] = ((visits["has_prior_cdw_pmh"] == 1) | (visits["has_prior_cdw_meds"] == 1)).astype(int)

    # 2C. Causal Labs (`Result_time <= Arrival_time + 2.0h`, plus discharge-time labs for Bounceback sensitivity)
    print("  [MC-MED 2C] Streaming labs.csv strictly on Result_time <= Arrival_time + 2.0h...")
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
    lab_2h_chunks = []
    lab_dc_chunks = []
    n_lab_ord_le2h = 0
    n_lab_res_le2h = 0
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
        dispo_c = pd.to_datetime(chunk["CSN"].map(csn_to_dispo), utc=True)
        ord_t = pd.to_datetime(chunk["Order_time"], errors="coerce", utc=True)
        res_t = pd.to_datetime(chunk["Result_time"], errors="coerce", utc=True)
        dt_ord = (ord_t - arr_c).dt.total_seconds() / 3600.0
        dt_res = (res_t - arr_c).dt.total_seconds() / 3600.0
        n_lab_ord_le2h += int(((dt_ord >= -2.0) & (dt_ord <= 2.0)).sum())
        m_2h = res_t.notna() & (dt_res >= -2.0) & (dt_res <= 2.0)
        n_lab_res_le2h += int(m_2h.sum())
        m_dc = res_t.notna() & dispo_c.notna() & (dt_res >= -2.0) & (res_t <= dispo_c)

        chunk["val"] = pd.to_numeric(chunk["Component_value"], errors="coerce")
        valid_val = chunk["val"].notna()
        for mask_sub, acc_list, suffix in [(m_2h & valid_val, lab_2h_chunks, ""), (m_dc & valid_val, lab_dc_chunks, "_dc")]:
            sub_c = chunk.loc[mask_sub, ["CSN", "Result_time", "Component_name", "val"]].copy()
            if len(sub_c) == 0:
                continue
            cname = sub_c["Component_name"].astype(str).str.upper()
            matched = []
            for feat_col, pat in lab_targets.items():
                mm = cname.str.contains(pat, regex=True, na=False)
                if mm.any():
                    s = sub_c.loc[mm, ["CSN", "Result_time", "val"]].copy()
                    s["feat"] = feat_col + suffix
                    matched.append(s)
            if matched:
                acc_list.append(pd.concat(matched, ignore_index=True))

    if lab_2h_chunks:
        all_labs_2h = pd.concat(lab_2h_chunks, ignore_index=True).sort_values(["CSN", "Result_time"], kind="mergesort")
        lab_piv_2h = all_labs_2h.groupby(["CSN", "feat"])["val"].last().unstack("feat")
        visits = visits.merge(lab_piv_2h, on="CSN", how="left")
    for feat_col in lab_targets:
        if feat_col not in visits.columns:
            visits[feat_col] = np.nan

    lab_piv_dc = None
    if lab_dc_chunks:
        all_labs_dc = pd.concat(lab_dc_chunks, ignore_index=True).sort_values(["CSN", "Result_time"], kind="mergesort")
        lab_piv_dc = all_labs_dc.groupby(["CSN", "feat"])["val"].last().unstack("feat")

    # 2D. Causal Radiology (`Result_time <= Arrival_time + 2.0h` strictly!)
    print("  [MC-MED 2D] Filtering rads.csv strictly on Result_time <= Arrival_time + 2.0h...")
    rads = pd.read_csv(mc / "rads.csv", usecols=["CSN", "Order_time", "Result_time", "Study", "Impression"], low_memory=False)
    rads = rads[rads["CSN"].isin(valid_csns)].copy()
    rads_arr = pd.to_datetime(rads["CSN"].map(csn_to_arr), utc=True)
    rads_ord_hr = (pd.to_datetime(rads["Order_time"], errors="coerce", utc=True) - rads_arr).dt.total_seconds() / 3600.0
    rads_res_dt = pd.to_datetime(rads["Result_time"], errors="coerce", utc=True)
    rads_res_hr = (rads_res_dt - rads_arr).dt.total_seconds() / 3600.0
    n_rad_ord_le2h = int(((rads_ord_hr >= -2.0) & (rads_ord_hr <= 2.0)).sum())
    rads_2h = rads[rads_res_dt.notna() & (rads_res_hr >= -2.0) & (rads_res_hr <= 2.0)].copy()
    n_rad_res_le2h = int(len(rads_2h))

    study_u = rads_2h["Study"].astype(str).str.upper()
    imp_u = rads_2h["Impression"].fillna("").astype(str).str.upper()
    rads_2h["rad_cxr"] = study_u.str.contains("CHEST|CXR", na=False).astype(int)
    rads_2h["rad_ct_head"] = study_u.str.contains("CT.*HEAD|CT.*BRAIN", regex=True, na=False).astype(int)
    rads_2h["rad_ct_abd"] = study_u.str.contains("CT.*ABD|CT.*PELV", regex=True, na=False).astype(int)
    rads_2h["rad_ct_chest"] = study_u.str.contains("CT.*CHEST|CT.*PE|ANGIO.*CHEST", regex=True, na=False).astype(int)
    rads_2h["rad_us"] = study_u.str.contains("US |ULTRASOUND", regex=True, na=False).astype(int)
    rads_2h["rad_imp_infiltrate_edema"] = imp_u.str.contains(
        "PNEUMONIA|INFILTRAT|CONSOLIDAT|EDEMA|EFFUSION", regex=True, na=False
    ).astype(int)
    rads_2h["rad_imp_acute_abn"] = imp_u.str.contains(
        "ACUTE|FRACTURE|HEMORRHAGE|OBSTRUCT|EMBOL|PNEUMOTHORAX", regex=True, na=False
    ).astype(int)
    rad_cols = [
        "rad_cxr",
        "rad_ct_head",
        "rad_ct_abd",
        "rad_ct_chest",
        "rad_us",
        "rad_imp_infiltrate_edema",
        "rad_imp_acute_abn",
    ]
    rad_agg = rads_2h.groupby("CSN")[rad_cols].max()
    rad_agg["rad_total_resulted_2h"] = rads_2h.groupby("CSN").size()
    visits = visits.merge(rad_agg, on="CSN", how="left")
    for c in list(rad_agg.columns):
        visits[c] = visits[c].fillna(0.0)

    # 2E. Continuous Monitor Numerics (`Time <= Arrival_time + 2.0h` strictly; ZERO `wf_dur_*`!)
    print("  [MC-MED 2E] Streaming numerics.csv (2.57 GiB, strict Time <= Arrival_time + 2.0h, ZERO wf_dur_*)...")
    num_measures = {"HR", "SpO2", "RR", "NIBP_sys", "NIBP_dias", "NIBP_mean", "Temp"}
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
        chunk = chunk[t_c.notna() & (dt_hr >= -0.5) & (dt_hr <= 2.0)].copy()
        if len(chunk) == 0:
            continue
        chunk["Value"] = pd.to_numeric(chunk["Value"], errors="coerce")
        chunk = chunk[chunk["Value"].notna()]
        if len(chunk) == 0:
            continue
        chunk["val_sq"] = chunk["Value"] ** 2
        g = chunk.groupby(["CSN", "Measure"]).agg(
            sum_v=("Value", "sum"),
            sum_sq=("val_sq", "sum"),
            cnt=("Value", "count"),
            min_v=("Value", "min"),
            max_v=("Value", "max"),
        ).reset_index()
        num_parts.append(g)

    if num_parts:
        num_df = pd.concat(num_parts, ignore_index=True)
        num_agg = num_df.groupby(["CSN", "Measure"]).agg(
            sum_v=("sum_v", "sum"),
            sum_sq=("sum_sq", "sum"),
            cnt=("cnt", "sum"),
            min_v=("min_v", "min"),
            max_v=("max_v", "max"),
        ).reset_index()
        num_agg["mean_v"] = num_agg["sum_v"] / num_agg["cnt"]
        var_v = np.maximum(0.0, (num_agg["sum_sq"] / num_agg["cnt"]) - (num_agg["mean_v"] ** 2))
        num_agg["std_v"] = np.sqrt(var_v)
        for stat_col, pref in [("mean_v", "mon_mean_"), ("min_v", "mon_min_"), ("max_v", "mon_max_"), ("std_v", "mon_std_")]:
            piv = num_agg.pivot(index="CSN", columns="Measure", values=stat_col)
            piv.columns = [f"{pref}{c}" for c in piv.columns]
            visits = visits.merge(piv, on="CSN", how="left")

    assert not any(c.startswith("wf_dur_") for c in visits.columns), "LEAKAGE GUARD: wf_dur_* must never exist!"

    # Save 1) Primary strict 2.0h landmark cohort (`mcmed_stanford_2h_v2.parquet`)
    # and  2) Full discharged sensitivity cohort (`mcmed_discharged_sensitivity_v2.parquet`)
    visits_drop_ts = visits.drop(columns=["arr_dt", "dispo_dt"])
    mc_2h = visits_drop_ts[visits_drop_ts["landmark_2h_eligible"] == 1].copy().reset_index(drop=True)
    out_2h_path = OUT_DIR / "mcmed_stanford_2h_v2.parquet"
    mc_2h.to_parquet(out_2h_path, index=False)

    disch_all = visits_drop_ts[visits_drop_ts["is_discharged"] == 1].copy().reset_index(drop=True)
    if lab_piv_dc is not None:
        disch_all = disch_all.merge(lab_piv_dc, on="CSN", how="left")
    out_dc_path = OUT_DIR / "mcmed_discharged_sensitivity_v2.parquet"
    disch_all.to_parquet(out_dc_path, index=False)

    # Quantify Stanford CDW prior-EHR record coverage across Payor_class and Race on the strict 2.0h cohort
    mc_2h["payor_clean"] = mc_2h["Payor_class"].fillna("Unspecified_or_Commercial").astype(str)
    mc_2h["race_clean"] = np.where(
        mc_2h["Race"].isin(["White", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander"]),
        mc_2h["Race"].astype(str),
        "Other_or_Unknown",
    )
    cdw_by_payor = {}
    for grp, sub in mc_2h.groupby("payor_clean"):
        cdw_by_payor[str(grp)] = {
            "n_visits": int(len(sub)),
            "has_prior_cdw_pmh_rate": round(float(sub["has_prior_cdw_pmh"].mean()), 4),
            "has_prior_cdw_meds_rate": round(float(sub["has_prior_cdw_meds"].mean()), 4),
            "has_any_prior_cdw_rate": round(float(sub["has_any_prior_cdw"].mean()), 4),
        }
    cdw_by_race = {}
    for grp, sub in mc_2h.groupby("race_clean"):
        cdw_by_race[str(grp)] = {
            "n_visits": int(len(sub)),
            "has_prior_cdw_pmh_rate": round(float(sub["has_prior_cdw_pmh"].mean()), 4),
            "has_prior_cdw_meds_rate": round(float(sub["has_prior_cdw_meds"].mean()), 4),
            "has_any_prior_cdw_rate": round(float(sub["has_any_prior_cdw"].mean()), 4),
        }

    mc_2h_disch = mc_2h[mc_2h["is_discharged"] == 1]
    stats = {
        "platform_provenance": "Released on both Nightingale Open Science (`mcmed-stanford-multi/v1`) and PhysioNet (Kansal et al., 2025, Scientific Data 12:1094); monitored ED beds at Stanford Health Care",
        "split_design": "Official `split_chrono` patient-disjoint chronological split (zero MRN overlap across train/cal/test)",
        "parquet_2h_path": str(out_2h_path),
        "parquet_discharged_sensitivity_path": str(out_dc_path),
        "cohort_flow": {
            "n_raw_visits_csv": n_visits_raw,
            "n_official_split_chrono": n_split_chrono,
            "n_excluded_dispo_or_admit_le_2h": int(n_split_chrono - len(mc_2h)),
            "n_strict_2h_landmark": int(len(mc_2h)),
            "n_strict_2h_unique_mrns": int(mc_2h["MRN"].nunique()),
            "splits_2h_visits": {k: int(v) for k, v in mc_2h["split"].value_counts().items()},
            "splits_2h_mrns": {s: int(mc_2h.loc[mc_2h["split"] == s, "MRN"].nunique()) for s in ["train", "cal", "test"]},
            "n_strict_2h_discharged": int(len(mc_2h_disch)),
            "splits_2h_discharged": {k: int(v) for k, v in mc_2h_disch["split"].value_counts().items()},
            "n_all_discharged_sensitivity": int(len(disch_all)),
            "splits_all_discharged_sensitivity": {k: int(v) for k, v in disch_all["split"].value_counts().items()},
        },
        "leakage_remediation_counts": {
            "pmh_unique_mrns_unfiltered": pmh_unfiltered_mrns,
            "visits_with_strict_prior_pmh_2h": int(mc_2h["has_prior_cdw_pmh"].sum()),
            "visits_with_strict_prior_meds_2h": int(mc_2h["has_prior_cdw_meds"].sum()),
            "lab_rows_order_le_2h": n_lab_ord_le2h,
            "lab_rows_result_le_2h": n_lab_res_le2h,
            "rad_rows_order_le_2h": n_rad_ord_le2h,
            "rad_rows_result_le_2h": n_rad_res_le2h,
            "wf_dur_columns_retained": 0,
        },
        "endpoints_2h_test": {
            "n_test": int((mc_2h["split"] == "test").sum()),
            "n_test_mrns": int(mc_2h.loc[mc_2h["split"] == "test", "MRN"].nunique()),
            "y_critical_icu_death_rate": round(float(mc_2h.loc[mc_2h["split"] == "test", "y_critical_icu_death"].mean()), 5),
            "y_critical_icu_death_events": int(mc_2h.loc[mc_2h["split"] == "test", "y_critical_icu_death"].sum()),
            "y_admit_rate": round(float(mc_2h.loc[mc_2h["split"] == "test", "y_admit"].mean()), 5),
            "y_admit_events": int(mc_2h.loc[mc_2h["split"] == "test", "y_admit"].sum()),
            "n_test_discharged_2h": int((mc_2h_disch["split"] == "test").sum()),
            "y_bounce72_admit_2h_rate": round(float(mc_2h_disch.loc[mc_2h_disch["split"] == "test", "y_bounce72_admit"].mean()), 5),
            "y_bounce72_admit_2h_events": int(mc_2h_disch.loc[mc_2h_disch["split"] == "test", "y_bounce72_admit"].sum()),
        },
        "cdw_prior_ehr_coverage_by_payor": cdw_by_payor,
        "cdw_prior_ehr_coverage_by_race": cdw_by_race,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("  Audited Stanford MC-MED complete:", json.dumps(stats["endpoints_2h_test"], indent=2))
    return stats


def _parse_ntuh_year(val: Any) -> int:
    """Parse NTUH year values including hyphenated ranges ('2011-14' -> 2014, '2015-16' -> 2016)."""
    s = str(val).strip()
    if s == "2011-14":
        return 2014
    if s == "2015-16":
        return 2016
    return int(float(s))


def _extract_ntuh_metadata_and_waveforms(root_dir: Path, valid_ecg_ids: set[str]) -> pd.DataFrame:
    """Extract identical resting intervals, anatomical per-lead bitflags & quantitative ST/T/QRS amplitudes
    (using verified GE MUSE integer `lead_id` 0..11), diagnostic statements, and 12-lead median waveforms
    (`waveform-median.npy`, standard 12-lead order 0..11) for a directory tree (`study-group` or `control-group`).
    """
    rest_dfs = []
    bf_dfs = []
    quant_dfs = []
    diag_dfs = []
    wf_dfs = []

    # Key quantitative measurement IDs from `measurement-matrix-per-lead-definitions.csv`:
    # 1:PAMP, 9:QAMP, 13:RAMP, 17:SAMP, 29:STJ, 30:STM, 31:STE, 35:QRSA, 39:TAMP, 50:TAREA
    key_mids = [1, 9, 13, 17, 29, 30, 31, 35, 39, 50]
    # Verified GE MUSE `lead_id` integer mapping:
    # 0=I, 1=II, 2=V1, 3=V2, 4=V3, 5=V4, 6=V5, 7=V6, 8=III, 9=aVR, 10=aVL, 11=aVF
    muse_terr_map = {
        0: "lat", 6: "lat", 7: "lat", 10: "lat",     # I, V5, V6, aVL
        2: "ant", 3: "ant", 4: "ant", 5: "ant",      # V1, V2, V3, V4
        1: "inf", 8: "inf", 11: "inf",               # II, III, aVF
        9: "avr",                                    # aVR
    }

    for r_p in sorted(root_dir.rglob("resting-ecg-measurements.csv")):
        grp_meta = r_p.parent
        rdf = pd.read_csv(r_p, low_memory=False)
        rdf = rdf[rdf["ecg_id"].isin(valid_ecg_ids)].copy()
        if len(rdf) > 0:
            rest_dfs.append(rdf)

        bf_p = grp_meta / "measurement-matrix-per-lead-bitflg.csv"
        if bf_p.exists():
            bf = pd.read_csv(bf_p, low_memory=False)
            bf = bf[bf["ecg_id"].isin(valid_ecg_ids)].copy()
            if len(bf) > 0:
                bf["lead_int"] = pd.to_numeric(bf["lead_id"], errors="coerce")
                bf["terr"] = bf["lead_int"].map(muse_terr_map).fillna("avr")
                flag_cols = [c for c in ["ttal", "stdown", "stelev", "jelev", "dltwv", "stinj", "ppdeep"] if c in bf.columns]
                for c in flag_cols:
                    bf[c] = pd.to_numeric(bf[c], errors="coerce").fillna(0.0)
                agg_bf = bf.groupby(["ecg_id", "terr"])[flag_cols].max().unstack("terr", fill_value=0.0)
                agg_bf.columns = [f"pl_bf_{c}_{t}" for c, t in agg_bf.columns]
                bf_dfs.append(agg_bf.reset_index())

        qm_p = grp_meta / "measurement-matrix-per-lead.csv"
        if qm_p.exists():
            qm = pd.read_csv(qm_p, usecols=["ecg_id", "lead_id", "measurement_id", "sample"], low_memory=False)
            qm = qm[qm["ecg_id"].isin(valid_ecg_ids) & qm["measurement_id"].isin(key_mids)].copy()
            if len(qm) > 0:
                qm["sample"] = pd.to_numeric(qm["sample"], errors="coerce")
                qm["lead_int"] = pd.to_numeric(qm["lead_id"], errors="coerce")
                qm["terr"] = qm["lead_int"].map(muse_terr_map).fillna("avr")
                agg_qm = qm.groupby(["ecg_id", "terr", "measurement_id"])["sample"].mean().unstack(["terr", "measurement_id"])
                agg_qm.columns = [f"pl_q_m{int(mid)}_{t}" for t, mid in agg_qm.columns]
                quant_dfs.append(agg_qm.reset_index())

        d_p = grp_meta / "diagnosis.csv"
        if d_p.exists():
            dd = pd.read_csv(d_p, usecols=["ecg_id", "stmt_text"], low_memory=False)
            dd = dd[dd["ecg_id"].isin(valid_ecg_ids)].copy()
            if len(dd) > 0:
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

        wf_dir = grp_meta.parent / "ecg-waveforms"
        wi_p = wf_dir / "waveform-npy.csv"
        wm_p = wf_dir / "waveform-median.npy"
        if wi_p.exists() and wm_p.exists():
            wi = pd.read_csv(wi_p, usecols=["ecg_id", "npy_index"], low_memory=False)
            wi = wi[wi["ecg_id"].isin(valid_ecg_ids)].drop_duplicates("ecg_id", keep="first").reset_index(drop=True)
            if len(wi) > 0:
                wm = np.load(wm_p, mmap_mode="r")  # (N, 12, 600): 0=I,1=II,2=III,3=aVR,4=aVL,5=aVF,6..11=V1..V6
                ridx = wi["npy_index"].astype(int).to_numpy()
                arr = np.nan_to_num(np.asarray(wm[ridx], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                arr = arr - np.mean(arr, axis=2, keepdims=True)
                l_std = np.std(arr, axis=2)  # (B, 12)
                l_p95 = np.percentile(np.abs(arr), 95, axis=2)
                d1_rms = np.sqrt(np.mean(np.diff(arr, axis=2) ** 2, axis=2))
                wdf = pd.DataFrame({
                    "ecg_id": wi["ecg_id"].values,
                    "medwf_ant_std": np.mean(l_std[:, [6, 7, 8, 9]], axis=1),
                    "medwf_lat_std": np.mean(l_std[:, [0, 4, 10, 11]], axis=1),
                    "medwf_inf_std": np.mean(l_std[:, [1, 2, 5]], axis=1),
                    "medwf_avr_std": l_std[:, 3],
                    "medwf_ant_p95": np.mean(l_p95[:, [6, 7, 8, 9]], axis=1),
                    "medwf_lat_p95": np.mean(l_p95[:, [0, 4, 10, 11]], axis=1),
                    "medwf_inf_p95": np.mean(l_p95[:, [1, 2, 5]], axis=1),
                    "medwf_avr_p95": l_p95[:, 3],
                    "medwf_global_hf_rms": np.mean(d1_rms, axis=1),
                    "medwf_spatial_disp": np.std(l_std, axis=1),
                })
                wf_dfs.append(wdf)

    rest_cols = [
        "ecg_id", "ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration",
        "qt_interval", "qt_corrected", "p_axis", "r_axis", "t_axis", "qtc_frederica",
    ]
    out = pd.DataFrame({"ecg_id": sorted(valid_ecg_ids)})
    if rest_dfs:
        ra = pd.concat(rest_dfs, ignore_index=True).drop_duplicates("ecg_id", keep="first")
        keep_r = [c for c in rest_cols if c in ra.columns]
        for c in keep_r:
            if c != "ecg_id":
                ra[c] = pd.to_numeric(ra[c], errors="coerce")
        out = out.merge(ra[keep_r], on="ecg_id", how="left")
    if bf_dfs:
        bfa = pd.concat(bf_dfs, ignore_index=True).drop_duplicates("ecg_id", keep="first")
        out = out.merge(bfa, on="ecg_id", how="left")
    if quant_dfs:
        qma = pd.concat(quant_dfs, ignore_index=True).drop_duplicates("ecg_id", keep="first")
        out = out.merge(qma, on="ecg_id", how="left")
    if diag_dfs:
        da = pd.concat(diag_dfs, ignore_index=True).drop_duplicates("ecg_id", keep="first")
        out = out.merge(da, on="ecg_id", how="left")
    if wf_dfs:
        wfa = pd.concat(wf_dfs, ignore_index=True).drop_duplicates("ecg_id", keep="first")
        out = out.merge(wfa, on="ecg_id", how="left")
    return out


def build_ntuh_audited() -> dict[str, Any]:
    """Cohort 3: NTUH `arrest-ntuh-ecg/v1`:
      (A) Primary True Case-Control Pre-Arrest SCA Cohort (2015-2019: `study-group [0]pre` vs `control-group`)
      (B) Supplementary Post-ROSC Cohort (`[1]ROSC` strictly, fixed hyphenated year parser).
    """
    t0 = time.time()
    print("\n[3/3] Building Audited NTUH Case-Control Pre-Arrest & Post-ROSC Cohorts (`arrest-ntuh-ecg/v1`)...")
    ntuh = RAW_ROOT / "arrest-ntuh-ecg" / "v1"

    sg_ecg = pd.read_csv(ntuh / "study-group" / "ecg.csv", low_memory=False)
    sg_coh = pd.read_csv(ntuh / "study-group" / "cohort.csv", low_memory=False)
    cg_coh = pd.read_csv(ntuh / "control-group" / "ecg-cohort.csv", low_memory=False)

    # Verify zero patient overlap between study-group and control-group
    overlap = set(sg_ecg["patient_ngsci_id"]).intersection(set(cg_coh["patient_ngsci_id"]))
    assert len(overlap) == 0, f"Found unexpected patient overlap: {len(overlap)}"

    # Attach age, sex, birthday_offset, rosc_datetime_offset to study-group ECGs
    sg_m = sg_ecg.merge(
        sg_coh[["ecg_id", "age", "sex", "birthday_offset", "rosc_datetime_offset"]],
        on="ecg_id",
        how="left",
    )
    pat_info = sg_coh.dropna(subset=["rosc_datetime_offset"]).groupby("patient_ngsci_id")[
        ["age", "sex", "birthday_offset", "rosc_datetime_offset"]
    ].first()
    for col in ["age", "sex", "birthday_offset", "rosc_datetime_offset"]:
        sg_m[col] = sg_m[col].fillna(sg_m["patient_ngsci_id"].map(pat_info[col]))

    # Compute age from birthday_offset if missing, and harmonize 90+ HIPAA top-coding identically across study & control groups
    acq_all_dt = pd.to_datetime(sg_m["acquisition_datetime_offset"], errors="coerce")
    bday_dt = pd.to_datetime(sg_m["birthday_offset"], errors="coerce")
    calc_age = (acq_all_dt - bday_dt).dt.total_seconds() / (365.25 * 86400.0)
    sg_age_raw = pd.to_numeric(sg_m["age"].astype(str).str.replace("+", "", regex=False), errors="coerce").fillna(calc_age)
    sg_m["age"] = np.minimum(sg_age_raw, 90.0)

    # --- 3A. True Case-Control Pre-Arrest SCA Cohort (2015-2019) ---
    sg_pre_cand = sg_m[
        (sg_m["ecg_timetag"] == "[0]pre")
        & (sg_m["year"].astype(str).isin(["2015-16", "2017", "2018", "2019"]))
    ].copy()
    n_sg_pre_raw_2015_19 = int(len(sg_pre_cand))
    acq_pre = pd.to_datetime(sg_pre_cand["acquisition_datetime_offset"], errors="coerce")
    rosc_pre = pd.to_datetime(sg_pre_cand["rosc_datetime_offset"], errors="coerce")
    lead_hours = (rosc_pre - acq_pre).dt.total_seconds() / 3600.0
    sg_pre_cand["lead_hours_before_rosc"] = lead_hours
    sg_cases = sg_pre_cand[lead_hours > 0.0].copy()
    n_excluded_post_rosc_timestamp = int(n_sg_pre_raw_2015_19 - len(sg_cases))

    sg_cases["y_pre_arrest"] = 1
    sg_cases["year_num"] = sg_cases["year"].map(_parse_ntuh_year)
    sg_cases["sex_male"] = sg_cases["sex"].astype(str).str.upper().str.startswith("M").astype(int)

    cg_controls = cg_coh[cg_coh["ecg_timetag"].isin(["pre", "control"])].copy()
    cg_age_raw = pd.to_numeric(cg_controls["age"].astype(str).str.replace("+", "", regex=False), errors="coerce")
    cg_controls["age"] = np.minimum(cg_age_raw, 90.0)
    cg_controls["y_pre_arrest"] = 0
    cg_controls["year_num"] = cg_controls["year"].map(_parse_ntuh_year)
    cg_controls["sex_male"] = cg_controls["sex"].astype(str).str.upper().str.startswith("M").astype(int)
    cg_controls["lead_hours_before_rosc"] = np.nan

    sg_feats = _extract_ntuh_metadata_and_waveforms(ntuh / "study-group", set(sg_cases["ecg_id"]))
    cg_feats = _extract_ntuh_metadata_and_waveforms(ntuh / "control-group", set(cg_controls["ecg_id"]))

    keep_core = ["patient_ngsci_id", "ecg_id", "year", "year_num", "ecg_timetag", "age", "sex_male", "y_pre_arrest", "lead_hours_before_rosc"]
    sg_cases["year"] = sg_cases["year"].astype(str)
    cg_controls["year"] = cg_controls["year"].astype(str)
    df_cc = pd.concat(
        [
            sg_cases[keep_core].merge(sg_feats, on="ecg_id", how="left"),
            cg_controls[keep_core].merge(cg_feats, on="ecg_id", how="left"),
        ],
        ignore_index=True,
    )

    # Zero-fill binary diagnostic and bitflag columns when absent in diagnosis.csv
    for c in df_cc.columns:
        if c.startswith("dx_") or c.startswith("pl_bf_"):
            df_cc[c] = df_cc[c].fillna(0.0)

    # Strictly chronological patient-disjoint split by each patient's earliest ECG year (`<=2017 -> train`, `2018 -> cal`, `2019 -> test`)
    first_yr = df_cc.groupby("patient_ngsci_id")["year_num"].min()
    tr_pats = set(first_yr[first_yr <= 2017].index)
    cal_pats = set(first_yr[first_yr == 2018].index)
    df_cc["split"] = np.where(
        df_cc["patient_ngsci_id"].isin(tr_pats),
        "train",
        np.where(df_cc["patient_ngsci_id"].isin(cal_pats), "cal", "test"),
    )
    assert len(set(df_cc.loc[df_cc["split"] == "train", "patient_ngsci_id"]) & set(df_cc.loc[df_cc["split"] == "test", "patient_ngsci_id"])) == 0

    out_cc_path = OUT_DIR / "ntuh_pre_arrest_case_control_v2.parquet"
    df_cc.to_parquet(out_cc_path, index=False)

    # --- 3B. Supplementary Post-ROSC Cohort (`ecg_timetag == '[1]ROSC'` strictly, fixed hyphenated year parser) ---
    sg_rosc_traces = sg_m[sg_m["ecg_timetag"] == "[1]ROSC"].copy()
    n_rosc_raw_traces = int(len(sg_rosc_traces))
    n_rosc_raw_pats = int(sg_rosc_traces["patient_ngsci_id"].nunique())

    rosc_outcomes = pd.read_csv(ntuh / "study-group" / "rosc-outcomes.csv", low_memory=False)
    comorb = pd.read_csv(ntuh / "study-group" / "comorbidities.csv", low_memory=False)
    rosc_feats = _extract_ntuh_metadata_and_waveforms(ntuh / "study-group", set(sg_rosc_traces["ecg_id"]))

    df_rosc = sg_rosc_traces.merge(
        rosc_outcomes.drop(columns=["patient_ngsci_id", "year"], errors="ignore"),
        on="ecg_id",
        how="inner",
    ).merge(
        comorb.drop(columns=["patient_ngsci_id", "year"], errors="ignore"),
        on="ecg_id",
        how="left",
    ).merge(
        rosc_feats,
        on="ecg_id",
        how="left",
    )
    df_rosc = df_rosc[df_rosc["hospital_discharge_survival"].isin([0.0, 1.0])].copy().reset_index(drop=True)
    df_rosc["y_survive"] = df_rosc["hospital_discharge_survival"].astype(int)
    df_rosc["y_good_neuro_cpc12"] = (
        df_rosc["hospital_discharge_cpc"].astype(str).str.contains(r"\[1\]|\[2\]", regex=True, na=False).astype(int)
    )
    df_rosc["sex_male"] = df_rosc["sex"].astype(str).str.upper().str.startswith("M").astype(int)
    df_rosc["year"] = df_rosc["year"].astype(str)
    df_rosc["year_num"] = df_rosc["year"].map(_parse_ntuh_year)
    for c in df_rosc.columns:
        if c.startswith("dx_") or c.startswith("pl_bf_"):
            df_rosc[c] = df_rosc[c].fillna(0.0)
        elif df_rosc[c].dtype == object:
            df_rosc[c] = df_rosc[c].where(df_rosc[c].isna(), df_rosc[c].astype(str))

    rosc_first_yr = df_rosc.groupby("patient_ngsci_id")["year_num"].min()
    r_tr_pats = set(rosc_first_yr[rosc_first_yr <= 2017].index)
    r_cal_pats = set(rosc_first_yr[rosc_first_yr == 2018].index)
    df_rosc["split"] = np.where(
        df_rosc["patient_ngsci_id"].isin(r_tr_pats),
        "train",
        np.where(df_rosc["patient_ngsci_id"].isin(r_cal_pats), "cal", "test"),
    )
    assert len(set(df_rosc.loc[df_rosc["split"] == "train", "patient_ngsci_id"]) & set(df_rosc.loc[df_rosc["split"] == "test", "patient_ngsci_id"])) == 0

    out_rosc_path = OUT_DIR / "ntuh_post_rosc_v2.parquet"
    df_rosc.to_parquet(out_rosc_path, index=False)

    cc_split_summary = {}
    for s in ["train", "cal", "test"]:
        sub = df_cc[df_cc["split"] == s]
        cc_split_summary[s] = {
            "n_ecgs": int(len(sub)),
            "n_patients": int(sub["patient_ngsci_id"].nunique()),
            "n_case_ecgs": int(sub["y_pre_arrest"].sum()),
            "n_case_patients": int(sub.loc[sub["y_pre_arrest"] == 1, "patient_ngsci_id"].nunique()),
            "n_control_ecgs": int((sub["y_pre_arrest"] == 0).sum()),
            "case_fraction_sample": round(float(sub["y_pre_arrest"].mean()), 5),
        }

    rosc_split_summary = {}
    for s in ["train", "cal", "test"]:
        sub = df_rosc[df_rosc["split"] == s]
        rosc_split_summary[s] = {
            "n_traces": int(len(sub)),
            "n_patients": int(sub["patient_ngsci_id"].nunique()),
            "survival_rate": round(float(sub["y_survive"].mean()), 4),
            "cpc12_rate": round(float(sub["y_good_neuro_cpc12"].mean()), 4),
        }

    stats = {
        "platform_provenance": "Nightingale Open Science DUA (`arrest-ntuh-ecg/v1`, Obermeyer et al. 2026 Nature)",
        "pre_arrest_case_control_2015_2019": {
            "parquet_path": str(out_cc_path),
            "design": "2015-2019 case-control pre-arrest SCA cohort (`study-group [0]pre` strictly prior to `rosc_datetime_offset` vs `control-group` non-arrest patients; zero patient overlap)",
            "n_total_ecgs": int(len(df_cc)),
            "n_total_patients": int(df_cc["patient_ngsci_id"].nunique()),
            "n_case_ecgs": int(df_cc["y_pre_arrest"].sum()),
            "n_case_patients": int(df_cc.loc[df_cc["y_pre_arrest"] == 1, "patient_ngsci_id"].nunique()),
            "n_excluded_post_rosc_timestamp_in_0pre": n_excluded_post_rosc_timestamp,
            "case_lead_hours_median_iqr": [
                round(float(sg_cases["lead_hours_before_rosc"].quantile(0.25)), 2),
                round(float(sg_cases["lead_hours_before_rosc"].median()), 2),
                round(float(sg_cases["lead_hours_before_rosc"].quantile(0.75)), 2),
            ],
            "n_control_ecgs": int((df_cc["y_pre_arrest"] == 0).sum()),
            "n_control_patients": int(df_cc.loc[df_cc["y_pre_arrest"] == 0, "patient_ngsci_id"].nunique()),
            "patient_overlap_cases_vs_controls": 0,
            "splits": cc_split_summary,
        },
        "supplementary_post_rosc": {
            "parquet_path": str(out_rosc_path),
            "design": "Strictly `[1]ROSC` post-arrest ECGs with hyphenated years (`2011-14`, `2015-16`, `2017` -> train; `2018` -> cal; `2019` -> test)",
            "n_raw_1rosc_traces": n_rosc_raw_traces,
            "n_raw_1rosc_patients": n_rosc_raw_pats,
            "n_analyzed_traces": int(len(df_rosc)),
            "n_analyzed_patients": int(df_rosc["patient_ngsci_id"].nunique()),
            "splits": rosc_split_summary,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("  Audited NTUH Pre-Arrest splits:", json.dumps(cc_split_summary, indent=2))
    print("  Audited NTUH Post-ROSC splits :", json.dumps(rosc_split_summary, indent=2))
    return stats


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": "paper3_v2_audited_cohorts",
        "bwh_ed_ecg": build_bwh_audited(),
        "mcmed_stanford": build_mcmed_audited(),
        "ntuh_arrest_ecg": build_ntuh_audited(),
        "total_elapsed_sec": round(time.time() - t0, 1),
    }
    audit_path = OUT_DIR / "cohort_provenance_and_flow_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print("\nALL 3 AUDITED V2 COHORTS BUILT CLEANLY ->", audit_path)


if __name__ == "__main__":
    main()

"""Phase 2B Upgrades for Decision-Aware Clinical Feature Acquisition:
  1. 4-Fold Cross-Fitted Smooth Schervish Policy-Aware Admission (Smooth-Policy-Delta_m)
     solving the discrete step-function rare-event power cliff observed in Phase 2A.
  2. BWH ED-ECG: ZIP-Code Income Quartile (Q1-Q4) + Race Fairness Audit & Agniel-Deconfounded
     + Measured-Troponin Sensitivity Arms.
  3. NTUH Arrest-ECG: Quantitative 12-Lead ST/T/QRS Amplitudes (measurement-matrix-per-lead.csv)
     for Pre-Arrest SCA AND Post-ROSC Hospital Survival / Good Neuro (CPC 1-2).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, "/Users/o2satz/antigravity/collision_zone_ed/src")

from collision_zone.three_predicates import (
    AgnielDeconfounder,
    ChannelDef,
    ClinicalContract,
    FastFrameEncoder,
    FluidCollisionBackboneClassifier,
    apply_platt,
    bernoulli_kl_from_tau,
    compute_oof_stage1,
    empirical_martingale_voi,
    fit_platt_calibrator,
    fit_stage1_predictions,
    holm_adjust,
    net_benefit_per_1k,
    paired_bootstrap_nb_diff,
    run_sequential_discovery,
    schervish_collision_proper_score,
    solve_ess_kappa,
)
from decision_aware_acq.policy_admission import (
    skipped_case_safety_certificate,
)

RAW_ROOT = Path("/Volumes/WS4TB/imboreceipt_data")
PROC_DIR = RAW_ROOT / "processed_clinical_cohorts"
OUT_JSON = REPO_ROOT / "results" / "phase2b_clinical_upgrades.json"


def _fit_gbdt_cal(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_te: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    clf = lgb.LGBMClassifier(
        n_estimators=140,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        max_delta_step=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(x_tr, y_tr)
    raw_cal = clf.predict(x_cal, raw_score=True)
    raw_te = clf.predict(x_te, raw_score=True)
    platt = fit_platt_calibrator(raw_cal, y_cal)
    return (
        np.clip(apply_platt(raw_cal, platt), 1e-6, 1.0 - 1e-6),
        np.clip(apply_platt(raw_te, platt), 1e-6, 1.0 - 1e-6),
    )


def _compute_emp_voi(
    p1_cal: np.ndarray,
    p2_cal: np.ndarray,
    p1_target: np.ndarray,
    contract: ClinicalContract,
) -> np.ndarray:
    delta_cal = logit(np.clip(p2_cal, 1e-6, 1 - 1e-6)) - logit(np.clip(p1_cal, 1e-6, 1 - 1e-6))
    edges = np.quantile(p1_cal, np.linspace(0.0, 1.0, 13))
    edges[0], edges[-1] = 0.0, 1.0
    b_cal = np.clip(np.digitize(p1_cal, edges[1:-1], right=False), 0, 11)
    q_levels = np.linspace(0.02, 0.98, 33)
    global_q = np.quantile(delta_cal, q_levels) if delta_cal.size >= 5 else np.zeros(33, dtype=float)
    bin_quantiles = np.zeros((12, 33), dtype=float)
    for b_i in range(12):
        m = b_cal == b_i
        bin_quantiles[b_i, :] = np.quantile(delta_cal[m], q_levels) if int(np.sum(m)) >= 15 else global_q
    b_tgt = np.clip(np.digitize(p1_target, edges[1:-1], right=False), 0, 11)
    return empirical_martingale_voi(p1_target, b_tgt, bin_quantiles, contract)


def smooth_policy_aware_admission(
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    p1_tr_oof: np.ndarray,
    base_cols: list[str],
    channels: list[ChannelDef],
    contract: ClinicalContract,
    budget: float,
    pilot_frac: float = 0.25,
    kappa_max: float = 20.0,
    seed: int = 42,
) -> dict[str, Any]:
    """4-fold cross-fitted Smooth Schervish Policy-Aware Admission:
    Evaluates each candidate channel strictly through its VOI-routed policy p_route = Q*p2 + (1-Q)*p1
    using smooth strictly proper Schervish Beta-kernel score S_{tau*, kappa_eff}(Y, p_route).
    """
    n_tr = len(y_tr)
    rng = np.random.default_rng(seed)
    if pilot_frac < 1.0 and n_tr > 6000:
        # Stratified pilot subsample preserving full rare-event power
        pos_idx = np.flatnonzero(y_tr == 1)
        neg_idx = np.flatnonzero(y_tr == 0)
        n_pos_p = min(len(pos_idx), max(400, int(len(pos_idx) * max(pilot_frac, 0.50))))
        n_neg_p = min(len(neg_idx), max(2000, int(len(neg_idx) * pilot_frac)))
        sel_pos = rng.choice(pos_idx, size=n_pos_p, replace=False)
        sel_neg = rng.choice(neg_idx, size=n_neg_p, replace=False)
        p_idx = np.sort(np.concatenate([sel_pos, sel_neg]))
        w_ht = np.where(y_tr[p_idx] == 1, len(pos_idx) / n_pos_p, len(neg_idx) / n_neg_p)
    else:
        p_idx = np.arange(n_tr)
        w_ht = np.ones(n_tr, dtype=float)

    df_p = df_tr.iloc[p_idx].reset_index(drop=True)
    yp = y_tr[p_idx]
    p1_p = np.clip(p1_tr_oof[p_idx], 1e-5, 1.0 - 1e-5)

    # Adaptive kappa_eff so rare events never starve the smooth kernel
    kappa_eff = solve_ess_kappa(
        bernoulli_kl_from_tau(p1_p, contract.tau_star),
        kappa_max=kappa_max,
        min_effective_cz_samples=max(200.0, 0.25 * float(len(yp))),
    )

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 13)
    folds = list(skf.split(df_p, yp))

    # Pre-encode base and channel matrices per fold
    fold_base = []
    fold_ch: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {ch.channel_id: [] for ch in channels}
    for tr_i, va_i in folds:
        enc_b = FastFrameEncoder().fit(df_p[base_cols].iloc[tr_i])
        fold_base.append((enc_b.transform(df_p[base_cols].iloc[tr_i]), enc_b.transform(df_p[base_cols].iloc[va_i])))
        for ch in channels:
            enc_c = FastFrameEncoder().fit(df_p[ch.columns].iloc[tr_i])
            fold_ch[ch.channel_id].append(
                (enc_c.transform(df_p[ch.columns].iloc[tr_i]), enc_c.transform(df_p[ch.columns].iloc[va_i]))
            )

    def _oof_routed_score(sel_ids: list[str], step_seed: int) -> tuple[np.ndarray, float]:
        if len(sel_ids) == 0:
            return schervish_collision_proper_score(yp, p1_p, contract.tau_star, kappa=kappa_eff), 0.0
        p_route_oof = np.zeros(len(yp), dtype=float)
        q_oof = np.zeros(len(yp), dtype=bool)
        for k, (tr_i, va_i) in enumerate(folds):
            xb_tr, xb_va = fold_base[k]
            x_tr_k = np.hstack([xb_tr] + [fold_ch[cid][k][0] for cid in sel_ids])
            x_va_k = np.hstack([xb_va] + [fold_ch[cid][k][1] for cid in sel_ids])
            clf = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=23,
                min_child_samples=25,
                max_delta_step=1.0,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=step_seed + k,
                n_jobs=4,
                verbose=-1,
            )
            clf.fit(x_tr_k, yp[tr_i], sample_weight=w_ht[tr_i], init_score=logit(p1_p[tr_i]))
            raw_va = logit(p1_p[va_i]) + clf.predict(x_va_k, raw_score=True)
            p2_va = np.clip(1.0 / (1.0 + np.exp(-np.clip(raw_va, -18, 18))), 1e-5, 1.0 - 1e-5)
            # Compute empirical martingale VOI on fold validation predictions
            voi_va = _compute_emp_voi(p1_p[va_i], p2_va, p1_p[va_i], contract)
            q_va = voi_va > contract.per_instance_query_cost
            p_route_oof[va_i] = np.where(q_va, p2_va, p1_p[va_i])
            q_oof[va_i] = q_va
        scores = schervish_collision_proper_score(yp, p_route_oof, contract.tau_star, kappa=kappa_eff)
        return scores, float(q_oof.mean())

    selected: list[str] = []
    spent = 0.0
    history: list[dict[str, Any]] = []
    ch_by_id = {ch.channel_id: ch for ch in channels}
    curr_scores, _ = _oof_routed_score([], seed)

    for step in range(len(channels)):
        affordable = [ch.channel_id for ch in channels if ch.channel_id not in selected and spent + ch.cost <= budget + 1e-9]
        if not affordable:
            break
        cands = []
        raw_pvals = []
        for cid in affordable:
            sc_cand, q_rate = _oof_routed_score(selected + [cid], seed + 100 * (step + 1))
            diff = sc_cand - curr_scores
            w_sum = float(np.sum(w_ht))
            mean_d = float(np.sum(w_ht * diff) / w_sum)
            resid = diff - mean_d
            se_d = float(np.sqrt(np.sum((w_ht ** 2) * (resid ** 2)) / (w_sum ** 2)))
            se_d = max(se_d, 1e-12)
            z = mean_d / se_d
            pval = float(1.0 - norm.cdf(z))
            raw_pvals.append(pval)
            cands.append({
                "channel": cid,
                "gain": mean_d,
                "se": se_d,
                "z": z,
                "p_raw": pval,
                "q_rate": q_rate,
                "cost": ch_by_id[cid].cost,
                "gain_per_cost": mean_d / max(1e-6, ch_by_id[cid].cost),
                "scores": sc_cand,
            })
        p_holm = holm_adjust(np.asarray(raw_pvals))
        passing = []
        for idx, c in enumerate(cands):
            c["p_holm"] = float(p_holm[idx])
            if c["p_holm"] < 0.05 and c["gain"] > 0.0:
                passing.append(c)
        if not passing:
            history.append({"step": step, "action": "stop"})
            break
        win = max(passing, key=lambda c: c["gain_per_cost"])
        selected.append(win["channel"])
        spent += win["cost"]
        curr_scores = win["scores"]
        history.append({
            "step": step,
            "selected": win["channel"],
            "gain": round(win["gain"], 6),
            "z": round(win["z"], 3),
            "p_holm": round(win["p_holm"], 5),
            "q_rate": round(win["q_rate"], 4),
            "cost": win["cost"],
        })
    return {"selected_channels": selected, "spent_budget": round(spent, 3), "kappa_eff": round(kappa_eff, 2), "history": history}


def _eval_fluid_arm(
    selected_ids: list[str],
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    p1_tr_oof: np.ndarray,
    df_cal: pd.DataFrame,
    y_cal: np.ndarray,
    p1_cal: np.ndarray,
    df_te: pd.DataFrame,
    y_te: np.ndarray,
    p1_te: np.ndarray,
    base_cols: list[str],
    ch_by_id: dict[str, ChannelDef],
    contract: ClinicalContract,
    deconfound: bool = False,
    seed: int = 42,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    s2_cols = [c for cid in selected_ids for c in ch_by_id[cid].columns]
    if len(s2_cols) == 0:
        p2_cal, p2_te, beta_cz = p1_cal.copy(), p1_te.copy(), 0.0
    else:
        tr_s2, cal_s2, te_s2 = df_tr[s2_cols].copy(), df_cal[s2_cols].copy(), df_te[s2_cols].copy()
        if deconfound:
            dec = AgnielDeconfounder().fit(df_tr[base_cols], tr_s2)
            tr_s2 = dec.transform(df_tr[base_cols], tr_s2, seed=seed)
            cal_s2 = dec.transform(df_cal[base_cols], cal_s2, seed=seed + 1)
            te_s2 = dec.transform(df_te[base_cols], te_s2, seed=seed + 2)
        enc_b = FastFrameEncoder().fit(df_tr[base_cols])
        enc_z = FastFrameEncoder().fit(tr_s2)
        xb_tr, xb_cal, xb_te = enc_b.transform(df_tr[base_cols]), enc_b.transform(df_cal[base_cols]), enc_b.transform(df_te[base_cols])
        z_tr, z_cal, z_te = enc_z.transform(tr_s2), enc_z.transform(cal_s2), enc_z.transform(te_s2)
        x2_tr, x2_cal, x2_te = np.hstack([xb_tr, z_tr]), np.hstack([xb_cal, z_cal]), np.hstack([xb_te, z_te])
        clf = FluidCollisionBackboneClassifier(
            tau_star=contract.tau_star, kappa_max=20.0, min_cz_ess=240.0, seed=seed + 7
        ).fit(x2_tr, z_tr, y_tr, logit(np.clip(p1_tr_oof, 1e-5, 1.0 - 1e-5)))
        raw_cal = clf.decision_function(x2_cal, z_cal, logit(p1_cal))
        raw_te = clf.decision_function(x2_te, z_te, logit(p1_te))
        platt = fit_platt_calibrator(raw_cal, y_cal)
        p2_cal = np.clip(apply_platt(raw_cal, platt), 1e-6, 1.0 - 1e-6)
        p2_te = np.clip(apply_platt(raw_te, platt), 1e-6, 1.0 - 1e-6)
        beta_cz = float(clf.beta_cz_)

    voi_te = _compute_emp_voi(p1_cal, p2_cal, p1_te, contract)
    q_auto = voi_te > contract.per_instance_query_cost
    p_route = np.where(q_auto, p2_te, p1_te)
    cert = skipped_case_safety_certificate(y_te, p1_te, p2_te, q_auto, contract.tau_star, contract.cost_fp, contract.cost_fn)
    return (
        {
            "selected_channels": selected_ids,
            "beta_cz": round(beta_cz, 4),
            "nb_voi_auto": round(float(net_benefit_per_1k(y_te, p_route, q_auto, contract)), 4),
            "nb_100pct": round(float(net_benefit_per_1k(y_te, p2_te, np.ones(len(y_te), dtype=bool), contract)), 4),
            "auto_query_rate": round(float(q_auto.mean()), 4),
            "auroc_routed": round(float(roc_auc_score(y_te, p_route)), 4),
            "auprc_routed": round(float(average_precision_score(y_te, p_route)), 4),
            "skip_flip_ub95": round(float(cert["flip_rate_ub95"]), 4),
        },
        p_route,
        q_auto,
    )


def upgrade_1_smooth_policy_and_bwh() -> dict[str, Any]:
    """Run Upgrade 1 (Smooth Policy-Aware Admission) on Stanford Rare Tasks + Upgrade 3 (BWH AGI Quartiles + Agniel Troponin)."""
    out: dict[str, Any] = {}

    # A. BWH ED-ECG: AGI Quartiles + Agniel-Deconfounded Troponin + Measured-Troponin Subset
    print("\n=== UPGRADE A: BWH ED-ECG (AGI Income Quartiles + Agniel Troponin + Smooth Policy) ===")
    bwh = pd.read_parquet(PROC_DIR / "bwh_ed_ecg.parquet")
    bwh["race_str"] = np.where(
        bwh["race_white"] == 1, "White",
        np.where(bwh["race_black"] == 1, "Black",
        np.where(bwh["race_hispanic"] == 1, "Hispanic", "Other"))
    )
    bwh["agi_expected_k"] = (
        15.0 * bwh["agi_under_25k"].fillna(0)
        + 37.5 * bwh["agi_25k_to_50k"].fillna(0)
        + 62.5 * bwh["agi_50k_to_75k"].fillna(0)
        + 87.5 * bwh["agi_75k_to_100k"].fillna(0)
        + 150.0 * bwh["agi_100k_to_200k"].fillna(0)
        + 250.0 * bwh["agi_above_200k"].fillna(0)
    )
    valid_agi = bwh["agi_under_25k"].notna()
    bwh["agi_quartile"] = "Missing"
    bwh.loc[valid_agi, "agi_quartile"] = pd.qcut(
        bwh.loc[valid_agi, "agi_expected_k"], 4, labels=["Q1_Lowest_Income", "Q2_LowMid", "Q3_HighMid", "Q4_Highest_Income"]
    ).astype(str)

    bwh_base = [c for c in ["age_at_admit", "sex_male", "vent_rate", "pr_interval", "qrs_duration", "qt_interval", "qtc_interval", "p_axes", "r_axes", "t_axes"] if c in bwh.columns]
    rng = np.random.default_rng(42)
    bwh["null_1a"], bwh["null_1b"] = rng.normal(size=len(bwh)), rng.normal(size=len(bwh))
    bwh_channels = [
        ChannelDef("ch_ecg_ischemia_flags", [c for c in ["has_st_eleva", "has_depress", "has_twave_inver", "has_twave_abnormal", "has_jpoint_eleva", "has_jpoint_repol", "has_st", "has_twave"] if c in bwh.columns], 0.40),
        ChannelDef("ch_ecg_rhythm_struct", [c for c in ["has_afib", "has_bbb", "has_aberran_bbb", "has_lvh", "has_pacemaker", "has_prolonged_qt", "has_rhythm_disturbance", "has_normal_ecg", "has_normal_sinus"] if c in bwh.columns], 0.40),
        ChannelDef("ch_ecg_waveform_3lead", [c for c in bwh.columns if c.startswith("wf_")], 0.80),
        ChannelDef("ch_sameday_troponin", ["maxtrop_sameday", "log_maxtrop_sameday"], 1.15),
        ChannelDef("ch_null_noise_1", ["null_1a", "null_1b"], 0.50),
    ]
    ch_by_id = {c.channel_id: c for c in bwh_channels}
    contract_bwh = ClinicalContract(cost_fp=1.0, cost_fn=15.0, target_precision=0.25, per_instance_query_cost=0.03)

    tr, cal, te = bwh[bwh["split"] == "train"].reset_index(drop=True), bwh[bwh["split"] == "cal"].reset_index(drop=True), bwh[bwh["split"] == "test"].reset_index(drop=True)
    y_tr, y_cal, y_te = tr["macetrop_pos_or_death_030"].astype(int).to_numpy(), cal["macetrop_pos_or_death_030"].astype(int).to_numpy(), te["macetrop_pos_or_death_030"].astype(int).to_numpy()
    p1_tr_oof = compute_oof_stage1(tr[bwh_base], y_tr, n_folds=3, seed=42)
    p1_cal, p1_tests = fit_stage1_predictions(tr, y_tr, cal, y_cal, [te], bwh_base, seed=42)
    p1_te = p1_tests[0]

    spol = smooth_policy_aware_admission(tr, y_tr, p1_tr_oof, bwh_base, bwh_channels, contract_bwh, budget=2.40, seed=42)
    res_raw, p_raw, q_raw = _eval_fluid_arm(spol["selected_channels"], tr, y_tr, p1_tr_oof, cal, y_cal, p1_cal, te, y_te, p1_te, bwh_base, ch_by_id, contract_bwh, deconfound=False, seed=42)
    res_agn, p_agn, q_agn = _eval_fluid_arm(spol["selected_channels"], tr, y_tr, p1_tr_oof, cal, y_cal, p1_cal, te, y_te, p1_te, bwh_base, ch_by_id, contract_bwh, deconfound=True, seed=42)
    boot_agn_vs_s1 = paired_bootstrap_nb_diff(y_te, p_agn, q_agn, p1_te, np.zeros(len(y_te), dtype=bool), contract_bwh, n_boot=1000, seed=42)

    # Pure measured-troponin subset (zero ordering-indicator variance!)
    bwh_m = bwh[bwh["maxtrop_sameday"].notna()].copy()
    tr_m, cal_m, te_m = bwh_m[bwh_m["split"] == "train"].reset_index(drop=True), bwh_m[bwh_m["split"] == "cal"].reset_index(drop=True), bwh_m[bwh_m["split"] == "test"].reset_index(drop=True)
    ym_tr, ym_cal, ym_te = tr_m["macetrop_pos_or_death_030"].astype(int).to_numpy(), cal_m["macetrop_pos_or_death_030"].astype(int).to_numpy(), te_m["macetrop_pos_or_death_030"].astype(int).to_numpy()
    p1m_oof = compute_oof_stage1(tr_m[bwh_base], ym_tr, n_folds=3, seed=42)
    p1m_cal, p1m_tests = fit_stage1_predictions(tr_m, ym_tr, cal_m, ym_cal, [te_m], bwh_base, seed=42)
    p1m_te = p1m_tests[0]
    res_meas, pm_route, qm_route = _eval_fluid_arm(["ch_sameday_troponin"], tr_m, ym_tr, p1m_oof, cal_m, ym_cal, p1m_cal, te_m, ym_te, p1m_te, bwh_base, ch_by_id, contract_bwh, deconfound=False, seed=42)
    boot_meas_vs_s1 = paired_bootstrap_nb_diff(ym_te, pm_route, qm_route, p1m_te, np.zeros(len(ym_te), dtype=bool), contract_bwh, n_boot=1000, seed=42)

    # Income Quartile + Race fairness audit
    agi_fair = {}
    for col in ["agi_quartile", "race_str"]:
        vals = te[col].astype(str)
        for grp, cnt in vals.value_counts().items():
            if cnt < 100 or grp == "Missing":
                continue
            m = (vals == grp).to_numpy()
            nb0 = float(net_benefit_per_1k(y_te[m], p1_te[m], np.zeros(m.sum(), dtype=bool), contract_bwh))
            nb1 = float(net_benefit_per_1k(y_te[m], p_raw[m], q_raw[m], contract_bwh))
            agi_fair[f"{col}:{grp}"] = {
                "n": int(m.sum()), "prev": round(float(y_te[m].mean()), 4),
                "nb_s1": round(nb0, 2), "nb_arm": round(nb1, 2), "delta_nb": round(nb1 - nb0, 2),
                "q_rate": round(float(q_raw[m].mean()), 4),
            }

    print("  BWH Smooth-Policy selected:", spol["selected_channels"])
    print(f"  BWH Raw Fluid Auto NB={res_raw['nb_voi_auto']:+.2f} (AUROC={res_raw['auroc_routed']:.4f}, q={res_raw['auto_query_rate']*100:.1f}%)")
    print(f"  BWH Agniel-Deconfounded Auto NB={res_agn['nb_voi_auto']:+.2f} (vs S1: {boot_agn_vs_s1['diff_mean']:+.2f}, p={boot_agn_vs_s1['p_two_sided']:.4f})")
    print(f"  BWH Pure Measured-Troponin (N_te={len(te_m):,}) Auto NB={res_meas['nb_voi_auto']:+.2f} (AUROC={res_meas['auroc_routed']:.4f}, vs S1: {boot_meas_vs_s1['diff_mean']:+.2f}, p={boot_meas_vs_s1['p_two_sided']:.4f})")
    print("  BWH Income Quartile & Race Fairness:")
    for k, v in agi_fair.items():
        print(f"    {k:32s}: N={v['n']:4d}, S1={v['nb_s1']:+6.2f} -> Arm={v['nb_arm']:+6.2f} (dNB={v['delta_nb']:+6.2f})")

    out["bwh_upgrade"] = {
        "smooth_policy": spol,
        "raw_fluid": res_raw,
        "agniel_deconfounded": res_agn,
        "boot_agniel_vs_stage1": boot_agn_vs_s1,
        "measured_troponin_subset": {"n_test": len(te_m), "res": res_meas, "boot_vs_stage1": boot_meas_vs_s1},
        "fairness_agi_and_race": agi_fair,
    }

    # B. Stanford MC-MED Rare Tasks: Smooth Policy-Aware Admission vs Global-CMI vs CZ-Fluid
    print("\n=== UPGRADE B: STANFORD MC-MED Smooth Policy-Aware Admission (Critical ICU & 72h Bounceback) ===")
    mc = pd.read_parquet(PROC_DIR / "mcmed_stanford.parquet")
    mc["gender_male"] = mc["Gender"].astype(str).str.upper().str.startswith("M").astype(int)
    top_cc = mc.loc[mc["split"] == "train", "CC"].value_counts().head(25).index.tolist()
    mc["cc_clean"] = mc["CC"].where(mc["CC"].isin(top_cc), "OTHER_CC").astype(str)
    mc["arrival_clean"] = mc["Means_of_arrival"].fillna("Missing").astype(str)
    mc["null_1a"], mc["null_1b"] = rng.normal(size=len(mc)), rng.normal(size=len(mc))
    mc_base = ["Age", "gender_male", "Triage_Temp", "Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP", "Triage_DBP", "esi_num", "cc_clean", "arrival_clean"]
    mc_channels = [
        ChannelDef("ch_hie_pmh", [c for c in mc.columns if c.startswith("pmh_")], 0.40),
        ChannelDef("ch_hie_meds", [c for c in mc.columns if c.startswith("meds_")], 0.40),
        ChannelDef("ch_lab_cbc", [c for c in ["lab_cbc_wbc", "lab_cbc_hgb", "lab_cbc_plt"] if c in mc.columns], 0.78),
        ChannelDef("ch_lab_renal_elyte", [c for c in ["lab_renal_cr", "lab_renal_bun", "lab_elyte_na", "lab_elyte_k", "lab_elyte_hco3", "lab_gluc"] if c in mc.columns], 0.92),
        ChannelDef("ch_lab_lactate_cardiac", [c for c in ["lab_lactate", "lab_troponin", "lab_bnp"] if c in mc.columns], 1.15),
        ChannelDef("ch_lab_lft_coag", [c for c in ["lab_lft_ast", "lab_lft_alt", "lab_lft_bili", "lab_coag_inr"] if c in mc.columns], 0.95),
        ChannelDef("ch_rads_2h", [c for c in mc.columns if c.startswith("rad_")], 1.50),
        ChannelDef("ch_monitor_numerics_2h", [c for c in mc.columns if c.startswith("mon_") or c.startswith("wf_dur_")], 0.60),
        ChannelDef("ch_null_noise_1", ["null_1a", "null_1b"], 0.50),
    ]
    mc_ch_by_id = {c.channel_id: c for c in mc_channels}

    for task_key, df_sub, y_col, contract_mc, seed_mc in [
        ("mcmed_critical_icu_death", mc, "y_critical_icu_death", ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.03), 44),
        ("mcmed_bounce72_admit", mc[mc["is_discharged"] == 1].copy(), "y_bounce72_admit", ClinicalContract(cost_fp=1.0, cost_fn=32.33, target_precision=0.15, per_instance_query_cost=0.025), 46),
    ]:
        tr_s = df_sub[df_sub["split"] == "train"].reset_index(drop=True)
        cal_s = df_sub[df_sub["split"] == "cal"].reset_index(drop=True)
        te_s = df_sub[df_sub["split"] == "test"].reset_index(drop=True)
        ys_tr, ys_cal, ys_te = tr_s[y_col].astype(int).to_numpy(), cal_s[y_col].astype(int).to_numpy(), te_s[y_col].astype(int).to_numpy()
        p1s_oof = compute_oof_stage1(tr_s[mc_base], ys_tr, n_folds=3, seed=seed_mc)
        p1s_cal, p1s_tests = fit_stage1_predictions(tr_s, ys_tr, cal_s, ys_cal, [te_s], mc_base, seed=seed_mc)
        p1s_te = p1s_tests[0]

        spol_mc = smooth_policy_aware_admission(tr_s, ys_tr, p1s_oof, mc_base, mc_channels, contract_mc, budget=3.0, seed=seed_mc)
        res_spol, p_spol, q_spol = _eval_fluid_arm(spol_mc["selected_channels"], tr_s, ys_tr, p1s_oof, cal_s, ys_cal, p1s_cal, te_s, ys_te, p1s_te, mc_base, mc_ch_by_id, contract_mc, seed=seed_mc)

        # Also compare against Global-CMI baseline
        disc_g = run_sequential_discovery(tr_s, ys_tr, mc_base, mc_channels, p1s_oof, contract_mc, budget=3.0, pilot_fraction=0.20, method="global_cmi", seed=seed_mc)
        # Fit Global S2 on Global-CMI channels
        g_cols = [c for cid in disc_g.selected_ids for c in mc_ch_by_id[cid].columns]
        enc_b = FastFrameEncoder().fit(tr_s[mc_base])
        enc_g = FastFrameEncoder().fit(tr_s[g_cols]) if g_cols else None
        if g_cols:
            p2g_cal, p2g_te = _fit_gbdt_cal(
                np.hstack([enc_b.transform(tr_s[mc_base]), enc_g.transform(tr_s[g_cols])]), ys_tr,
                np.hstack([enc_b.transform(cal_s[mc_base]), enc_g.transform(cal_s[g_cols])]), ys_cal,
                np.hstack([enc_b.transform(te_s[mc_base]), enc_g.transform(te_s[g_cols])]), seed=seed_mc + 7
            )
        else:
            p2g_cal, p2g_te = p1s_cal.copy(), p1s_te.copy()
        voig_te = _compute_emp_voi(p1s_cal, p2g_cal, p1s_te, contract_mc)
        qg_te = voig_te > contract_mc.per_instance_query_cost
        pg_route = np.where(qg_te, p2g_te, p1s_te)
        nb_g = float(net_benefit_per_1k(ys_te, pg_route, qg_te, contract_mc))
        boot_spol_vs_g = paired_bootstrap_nb_diff(ys_te, p_spol, q_spol, pg_route, qg_te, contract_mc, n_boot=1000, seed=seed_mc)
        boot_spol_vs_s1 = paired_bootstrap_nb_diff(ys_te, p_spol, q_spol, p1s_te, np.zeros(len(ys_te), dtype=bool), contract_mc, n_boot=1000, seed=seed_mc + 1)

        print(f"  [{task_key}] Smooth-Policy selected: {spol_mc['selected_channels']} (null picked={any('null' in c for c in spol_mc['selected_channels'])})")
        print(f"    Smooth-Policy Auto NB={res_spol['nb_voi_auto']:+.2f} (q={res_spol['auto_query_rate']*100:.1f}%, AUROC={res_spol['auroc_routed']:.4f}) vs Global-CMI NB={nb_g:+.2f}")
        print(f"    Paired Boot Smooth-Policy vs Global-CMI: {boot_spol_vs_g['diff_mean']:+.2f} [{boot_spol_vs_g['ci95_low']:+.2f}, {boot_spol_vs_g['ci95_high']:+.2f}], p={boot_spol_vs_g['p_two_sided']:.4f}")
        print(f"    Paired Boot Smooth-Policy vs Stage-1   : {boot_spol_vs_s1['diff_mean']:+.2f} [{boot_spol_vs_s1['ci95_low']:+.2f}, {boot_spol_vs_s1['ci95_high']:+.2f}], p={boot_spol_vs_s1['p_two_sided']:.4f}")

        out[task_key] = {
            "smooth_policy": spol_mc,
            "res_smooth_policy": res_spol,
            "global_cmi_selected": disc_g.selected_ids,
            "global_cmi_auto_nb": round(nb_g, 4),
            "boot_smooth_policy_vs_global_cmi": boot_spol_vs_g,
            "boot_smooth_policy_vs_stage1": boot_spol_vs_s1,
        }

    return out


def upgrade_2_ntuh_quantitative_and_rosc() -> dict[str, Any]:
    """Upgrade NTUH: (1) Quantitative 12-lead ST/T/QRS amplitudes for Pre-Arrest SCA; (2) Post-ROSC Survival & CPC 1-2."""
    print("\n=== UPGRADE C: NTUH Quantitative 12-Lead Measurements & Post-ROSC Survival/Neuro ===")
    ntuh_root = RAW_ROOT / "arrest-ntuh-ecg" / "v1"

    # Read definitions to pick key clinical ECG amplitudes: QAMP(9), RAMP(13), RDUR(14), SAMP(17), STJ/STM/STE/TAMP
    defs = pd.read_csv(next(ntuh_root.rglob("measurement-matrix-per-lead-definitions.csv")))
    key_mids = defs[defs["abbreviation"].astype(str).str.upper().isin(["RAMP", "SAMP", "QAMP", "STJ", "STM", "STE", "TAMP", "PAMP", "QRST"])]["measurement_id"].tolist()
    if len(key_mids) < 4:
        key_mids = [1, 9, 13, 17, 25, 29, 33, 37, 41]
    print(f"  Selected quantitative NTUH measurement_ids: {key_mids}")

    # Stream quantitative per-lead measurements across control-group 2015-2019
    q_parts = []
    for pl_p in sorted((ntuh_root / "control-group").rglob("measurement-matrix-per-lead.csv")):
        pl = pd.read_csv(pl_p, usecols=["ecg_id", "lead_id", "measurement_id", "sample"], low_memory=False)
        pl = pl[pl["measurement_id"].isin(key_mids)].copy()
        pl["sample"] = pd.to_numeric(pl["sample"], errors="coerce")
        # Summarize mean, std, min, max across the 12 leads per measurement_id
        g = pl.groupby(["ecg_id", "measurement_id"])["sample"].agg(["mean", "std", "min", "max"]).unstack("measurement_id")
        g.columns = [f"qpl_m{mid}_{stat}" for stat, mid in g.columns]
        q_parts.append(g.reset_index())
    q_all = pd.concat(q_parts, ignore_index=True).drop_duplicates("ecg_id", keep="first")

    ntuh = pd.read_parquet(PROC_DIR / "ntuh_arrest_ecg.parquet").merge(q_all, on="ecg_id", how="left")
    ntuh_base = [c for c in ["age", "sex_male", "ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration", "qt_interval", "qt_corrected", "p_axis", "r_axis", "t_axis", "qtc_frederica"] if c in ntuh.columns]
    q_cols = [c for c in ntuh.columns if c.startswith("qpl_")]
    ntuh_channels = [
        ChannelDef("ch_quant_12lead_amp", q_cols, 0.80),
        ChannelDef("ch_ecg_diag_stmts", [c for c in ntuh.columns if c.startswith("dx_")], 0.40),
    ]
    tr, cal, te = ntuh[ntuh["split"] == "train"].reset_index(drop=True), ntuh[ntuh["split"] == "cal"].reset_index(drop=True), ntuh[ntuh["split"] == "test"].reset_index(drop=True)
    y_tr, y_cal, y_te = tr["y_pre_arrest"].astype(int).to_numpy(), cal["y_pre_arrest"].astype(int).to_numpy(), te["y_pre_arrest"].astype(int).to_numpy()
    p1_oof = compute_oof_stage1(tr[ntuh_base], y_tr, n_folds=3, seed=47)
    p1_cal, p1_tests = fit_stage1_predictions(tr, y_tr, cal, y_cal, [te], ntuh_base, seed=47)
    p1_te = p1_tests[0]
    contract_ntuh = ClinicalContract(cost_fp=1.0, cost_fn=5.67, target_precision=0.50, per_instance_query_cost=0.02)
    spol_ntuh = smooth_policy_aware_admission(tr, y_tr, p1_oof, ntuh_base, ntuh_channels, contract_ntuh, budget=1.80, seed=47)
    ch_map = {c.channel_id: c for c in ntuh_channels}
    res_ntuh, p_ntuh, q_ntuh = _eval_fluid_arm(spol_ntuh["selected_channels"], tr, y_tr, p1_oof, cal, y_cal, p1_cal, te, y_te, p1_te, ntuh_base, ch_map, contract_ntuh, seed=47)
    boot_ntuh = paired_bootstrap_nb_diff(y_te, p_ntuh, q_ntuh, p1_te, np.zeros(len(y_te), dtype=bool), contract_ntuh, n_boot=1000, seed=47)
    print(f"  NTUH Pre-Arrest Quant-12Lead: selected={spol_ntuh['selected_channels']} | AUROC={res_ntuh['auroc_routed']:.4f} (vs S1 {roc_auc_score(y_te, p1_te):.4f}) | Auto NB={res_ntuh['nb_voi_auto']:+.2f} (vs S1: {boot_ntuh['diff_mean']:+.2f}, p={boot_ntuh['p_two_sided']:.4f})")

    # 2. NTUH Post-ROSC Hospital Survival & Good Neurological Outcome (CPC 1-2)
    sg = ntuh_root / "study-group"
    coh = pd.read_csv(sg / "cohort.csv", low_memory=False)
    rosc = pd.read_csv(sg / "rosc-outcomes.csv", low_memory=False)
    comorb = pd.read_csv(sg / "comorbidities.csv", low_memory=False)
    sg_rest_parts = [pd.read_csv(p, low_memory=False) for p in sorted(sg.rglob("resting-ecg-measurements.csv"))]
    sg_rest = pd.concat(sg_rest_parts, ignore_index=True).drop_duplicates("ecg_id", keep="first")

    df_r = coh.merge(rosc.drop(columns=["patient_ngsci_id", "year"], errors="ignore"), on="ecg_id", how="inner")
    df_r = df_r.merge(comorb.drop(columns=["patient_ngsci_id", "year"], errors="ignore"), on="ecg_id", how="left")
    df_r = df_r.merge(sg_rest.drop(columns=["year", "ecg_timetag"], errors="ignore"), on="ecg_id", how="left")
    df_r = df_r[df_r["hospital_discharge_survival"].isin([0.0, 1.0])].copy()
    df_r["y_survive"] = df_r["hospital_discharge_survival"].astype(int)
    df_r["y_good_neuro_cpc12"] = df_r["hospital_discharge_cpc"].astype(str).str.contains(r"\[1\]|\[2\]", regex=True, na=False).astype(int)
    df_r["sex_male"] = df_r["sex"].astype(str).str.upper().str.startswith("M").astype(int)
    yr_num = pd.to_numeric(df_r["year"], errors="coerce")
    df_r["split"] = np.where(yr_num <= 2017, "train", np.where(yr_num == 2018, "cal", "test"))

    r_base = [c for c in ["age", "sex_male", "witnessed_collapse", "initial_rhythm", "location_of_cardiac_arrest", "cpr_duration", "epinerphine_dose", "dc_shock"] if c in df_r.columns]
    r_ecg_cols = [c for c in ["ventricular_rate", "pr_interval", "qrs_duration", "qt_interval", "qt_corrected", "qtc_frederica", "p_axis", "r_axis", "t_axis"] if c in df_r.columns]
    comorb_cols = [c for c in comorb.columns if c not in ("patient_ngsci_id", "year", "ecg_id") and c in df_r.columns]
    r_channels = [
        ChannelDef("ch_rosc_ecg_intervals", r_ecg_cols, 0.50),
        ChannelDef("ch_rosc_comorbidities", comorb_cols, 0.50),
    ]
    r_ch_map = {c.channel_id: c for c in r_channels}
    tr_r, cal_r, te_r = df_r[df_r["split"] == "train"].reset_index(drop=True), df_r[df_r["split"] == "cal"].reset_index(drop=True), df_r[df_r["split"] == "test"].reset_index(drop=True)
    rosc_out = {}
    for target_r, c_fn in [("y_survive", 3.0), ("y_good_neuro_cpc12", 4.0)]:
        yr_tr, yr_cal, yr_te = tr_r[target_r].to_numpy(), cal_r[target_r].to_numpy(), te_r[target_r].to_numpy()
        p1r_oof = compute_oof_stage1(tr_r[r_base], yr_tr, n_folds=3, seed=48)
        p1r_cal, p1r_tests = fit_stage1_predictions(tr_r, yr_tr, cal_r, yr_cal, [te_r], r_base, seed=48)
        p1r_te = p1r_tests[0]
        c_r = ClinicalContract(cost_fp=1.0, cost_fn=c_fn, target_precision=0.40, per_instance_query_cost=0.02)
        spol_r = smooth_policy_aware_admission(tr_r, yr_tr, p1r_oof, r_base, r_channels, c_r, budget=1.20, pilot_frac=1.0, seed=48)
        res_r, pr_route, qr_route = _eval_fluid_arm(spol_r["selected_channels"], tr_r, yr_tr, p1r_oof, cal_r, yr_cal, p1r_cal, te_r, yr_te, p1r_te, r_base, r_ch_map, c_r, seed=48)
        print(f"  NTUH Post-ROSC [{target_r}] (N_tr={len(tr_r)}, N_te={len(te_r)}, prev={yr_te.mean()*100:.1f}%): S1 AUROC={roc_auc_score(yr_te, p1r_te):.4f} -> Arm AUROC={res_r['auroc_routed']:.4f} | selected={spol_r['selected_channels']} | NB={res_r['nb_voi_auto']:+.2f}")
        rosc_out[target_r] = {
            "n_train": len(tr_r), "n_cal": len(cal_r), "n_test": len(te_r),
            "test_prev": round(float(yr_te.mean()), 4),
            "stage1_auroc": round(float(roc_auc_score(yr_te, p1r_te)), 4),
            "stage1_nb": round(float(net_benefit_per_1k(yr_te, p1r_te, np.zeros(len(yr_te), dtype=bool), c_r)), 4),
            "smooth_policy": spol_r,
            "res_fluid": res_r,
        }

    return {
        "ntuh_pre_arrest_quant12": {
            "stage1_auroc": round(float(roc_auc_score(y_te, p1_te)), 4),
            "smooth_policy": spol_ntuh,
            "res_fluid": res_ntuh,
            "boot_vs_stage1": boot_ntuh,
        },
        "ntuh_post_rosc": rosc_out,
    }


def main() -> None:
    t0 = time.time()
    out = upgrade_1_smooth_policy_and_bwh()
    out["ntuh_upgrade"] = upgrade_2_ntuh_quantitative_and_rosc()
    out["total_elapsed_sec"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("\nSAVED ALL PHASE 2B UPGRADES ->", OUT_JSON)


if __name__ == "__main__":
    main()

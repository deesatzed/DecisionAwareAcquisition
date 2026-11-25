"""Step 2 (Paper 3 v2 Remediation): Unconfounded Ablations, 10,000-Rep Patient-Cluster Bootstraps,
Calibration/DCA/Subgroup Audits, 1,000-Rep Experiment E5 & 4D Operating-Envelope Grid (`R1`-`R24`).

Writes all results inside:
  results/

CPU Sharing Discipline:
  - Clinical phase runs strictly with `OMP_NUM_THREADS=4` and `n_jobs=4`.
  - Embarrassingly parallel synthetic phase (`1,000`-rep `E5` + `72`-cell x `50`-rep `4D` grid) runs with
    `ProcessPoolExecutor(max_workers=4)` where each worker enforces `OMP_NUM_THREADS=1` and `n_jobs=1`
    (4 workers x 1 thread = 4 CPU cores total).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

REBUILD_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REBUILD_ROOT / "data"
RES_DIR = REBUILD_ROOT / "results"

sys.path.insert(0, str(REBUILD_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "collision_zone_ed" / "src"))

from collision_zone.three_predicates import (
    AgnielDeconfounder,
    ChannelDef,
    ClinicalContract,
    FastFrameEncoder,
    FluidCollisionBackboneClassifier,
    apply_platt,
    compute_oof_stage1,
    fit_platt_calibrator,
    fit_stage1_predictions,
    holm_adjust,
    mean_centered_voi,
    sample_pilot,
    solve_martingale_offset,
)
from upca_v2_core import (
    adjust_probs_for_target_prevalence,
    case_control_prevalence_weights,
    check_within_stratum_pilot_independence,
    cochran_q_heterogeneity,
    compute_calibration_metrics,
    compute_weighted_decision_curve,
    discover_stage_a_ablation,
    fit_predict_prepurchase_voi,
    oof_policy_screening_stage_b,
    patient_cluster_bootstrap_diff,
    patient_cluster_bootstrap_scalar,
    skipped_case_disagreement_audit,
    standardized_net_benefit,
    unit_net_benefit_vec,
    weighted_net_benefit_per_1k,
)


def _clopper_pearson_ci95(k: int, n: int) -> list[float]:
    """Exact two-sided 95% Clopper-Pearson confidence interval for Binomial(n, p)."""
    if n <= 0:
        return [0.0, 1.0]
    lo = 0.0 if k <= 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return [round(lo, 5), round(hi, 5)]


# ===========================================================================
# PART A: CLINICAL COHORT UNCONFOUNDED EVALUATION & GOVERNANCE AUDITS
# ===========================================================================


def evaluate_clinical_task_full(
    task_id: str,
    task_label: str,
    df: pd.DataFrame,
    y_col: str,
    cluster_col: str,
    base_cols: list[str],
    channels: list[ChannelDef],
    contract: ClinicalContract,
    budget: float,
    dca_taus: list[float],
    *,
    pi_target_override: float | None = None,
    is_prespecified_primary: bool = True,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run all unconfounded comparators, 5 single-factor ablations, 10,000-rep patient-cluster bootstraps,
    calibration curves (intercept, slope, ICI), Vickers DCA, Skipped-Case Action-Disagreement Bounds,
    empirical VOI calibration, and within-stratum pilot independence check on a clinical task.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["null_1a"], df["null_1b"] = rng.normal(size=len(df)), rng.normal(size=len(df))
    df["null_2a"], df["null_2b"] = rng.normal(size=len(df)), rng.normal(size=len(df))
    ch_all = list(channels) + [
        ChannelDef("ch_null_noise_1", ("null_1a", "null_1b"), 0.50, is_null_stress=True),
        ChannelDef("ch_null_noise_2", ("null_2a", "null_2b"), 0.50, is_null_stress=True),
    ]
    ch_map = {c.channel_id: c for c in ch_all}

    tr = df[df["split"] == "train"].reset_index(drop=True)
    cal = df[df["split"] == "cal"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    y_tr = tr[y_col].astype(int).to_numpy()
    y_cal = cal[y_col].astype(int).to_numpy()
    y_te = te[y_col].astype(int).to_numpy()
    clusters_te = te[cluster_col].to_numpy()

    # Stage-1 baseline fit (sample prevalence first, then exact Bayes shift if case-control)
    p1_oof_raw = compute_oof_stage1(tr[base_cols], y_tr, n_folds=3, seed=seed)
    p1_cal_raw, p1_tests_raw = fit_stage1_predictions(tr, y_tr, cal, y_cal, [te], base_cols, seed=seed)
    p1_te_raw = p1_tests_raw[0]

    if pi_target_override is not None:
        w_tr = case_control_prevalence_weights(y_tr, pi_target_override)
        w_cal = case_control_prevalence_weights(y_cal, pi_target_override)
        w_te = case_control_prevalence_weights(y_te, pi_target_override)
        p1_oof_eval = adjust_probs_for_target_prevalence(p1_oof_raw, float(np.mean(y_tr)), pi_target_override)
        pi_cal_s = float(np.mean(y_cal))
        p1_cal_eval = adjust_probs_for_target_prevalence(p1_cal_raw, pi_cal_s, pi_target_override)
        p1_te_eval = adjust_probs_for_target_prevalence(p1_te_raw, pi_cal_s, pi_target_override)
        eval_prev = float(pi_target_override)
    else:
        w_tr = None
        w_cal = None
        w_te = None
        p1_oof_eval = p1_oof_raw
        p1_cal_eval = p1_cal_raw
        p1_te_eval = p1_te_raw
        eval_prev = float(np.mean(y_te))

    # --- Stage-A Discovery Comparators & Single-Factor Ablations ---
    disc_glob, _ = discover_stage_a_ablation(
        tr, y_tr, base_cols, ch_all, p1_oof_eval, contract, budget,
        mode="global_cmi", pilot_fraction=0.20, target_pop_weights=w_tr, raw_oof_p_base_for_fit=p1_oof_raw, seed=seed,
    )
    disc_wcz, _ = discover_stage_a_ablation(
        tr, y_tr, base_cols, ch_all, p1_oof_eval, contract, budget,
        mode="weighted_cz_005", pilot_fraction=0.20, target_pop_weights=w_tr, raw_oof_p_base_for_fit=p1_oof_raw, seed=seed,
    )
    disc_ipw, _ = discover_stage_a_ablation(
        tr, y_tr, base_cols, ch_all, p1_oof_eval, contract, budget,
        mode="ipw_cz_wht", pilot_fraction=0.20, target_pop_weights=w_tr, raw_oof_p_base_for_fit=p1_oof_raw, seed=seed,
    )
    disc_off_ll, _ = discover_stage_a_ablation(
        tr, y_tr, base_cols, ch_all, p1_oof_eval, contract, budget,
        mode="offset_cz_logloss", pilot_fraction=0.20, target_pop_weights=w_tr, raw_oof_p_base_for_fit=p1_oof_raw, seed=seed,
    )
    disc_off_cz, pilot_cz = discover_stage_a_ablation(
        tr, y_tr, base_cols, ch_all, p1_oof_eval, contract, budget,
        mode="offset_cz_schervish", pilot_fraction=0.20, target_pop_weights=w_tr, raw_oof_p_base_for_fit=p1_oof_raw, seed=seed,
    )

    # --- Stage-B Nested OOF Policy Screening (`UPCA-Fluid`) ---
    upca_sel, prune_log = oof_policy_screening_stage_b(
        disc_off_cz.selected_ids, tr, y_tr, p1_oof_raw, base_cols, ch_map, contract,
        pi_target_override=pi_target_override, seed=seed,
    )

    # Empirical check of I _|_ Z | (Y, b) on the Phase-2 pilot (`R15`)
    ind_check = check_within_stratum_pilot_independence(tr, ch_all, pilot_cz, y_tr, p1_oof_eval, n_bins=8)

    # Pre-encode base features once
    enc_b = FastFrameEncoder().fit(tr[base_cols])
    xb_tr, xb_cal, xb_te = enc_b.transform(tr[base_cols]), enc_b.transform(cal[base_cols]), enc_b.transform(te[base_cols])

    stage2_cache: dict[tuple[tuple[str, ...], bool], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _get_s2_probs(sel_ids: list[str], use_fluid: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (tuple(sel_ids), bool(use_fluid))
        if key in stage2_cache:
            return stage2_cache[key]
        if not sel_ids:
            res = (p1_oof_eval.copy(), p1_cal_eval.copy(), p1_te_eval.copy())
            stage2_cache[key] = res
            return res
        s2_cols = [c for cid in sel_ids for c in ch_map[cid].columns]
        enc_z = FastFrameEncoder().fit(tr[s2_cols])
        z_tr, z_cal, z_te = enc_z.transform(tr[s2_cols]), enc_z.transform(cal[s2_cols]), enc_z.transform(te[s2_cols])
        x2_tr, x2_cal, x2_te = np.hstack([xb_tr, z_tr]), np.hstack([xb_cal, z_cal]), np.hstack([xb_te, z_te])
        p1_oof_s = np.clip(p1_oof_raw, 1e-5, 1.0 - 1e-5)
        if use_fluid:
            tau_fit = (
                contract.tau_star
                if pi_target_override is None
                else float(adjust_probs_for_target_prevalence(np.array([contract.tau_star]), pi_target_override, float(np.mean(y_tr)))[0])
            )
            clf = FluidCollisionBackboneClassifier(
                tau_star=tau_fit, kappa_max=20.0, min_cz_ess=240.0, seed=seed + 7
            ).fit(x2_tr, z_tr, y_tr, logit(p1_oof_s))
            raw_tr = clf.decision_function(x2_tr, z_tr, logit(p1_oof_s))
            raw_c = clf.decision_function(x2_cal, z_cal, logit(p1_cal_raw))
            raw_t = clf.decision_function(x2_te, z_te, logit(p1_te_raw))
        else:
            clf_g = lgb.LGBMClassifier(
                n_estimators=140, learning_rate=0.05, num_leaves=31, min_child_samples=30,
                max_delta_step=1.0, subsample=0.85, colsample_bytree=0.85,
                random_state=seed + 7, n_jobs=4, verbose=-1,
            ).fit(x2_tr, y_tr)
            raw_tr = clf_g.predict(x2_tr, raw_score=True)
            raw_c = clf_g.predict(x2_cal, raw_score=True)
            raw_t = clf_g.predict(x2_te, raw_score=True)

        platt = fit_platt_calibrator(raw_c, y_cal)
        p2_tr = np.clip(apply_platt(raw_tr, platt), 1e-6, 1.0 - 1e-6)
        p2_c = np.clip(apply_platt(raw_c, platt), 1e-6, 1.0 - 1e-6)
        p2_t = np.clip(apply_platt(raw_t, platt), 1e-6, 1.0 - 1e-6)
        if pi_target_override is not None:
            pi_cal_s = float(np.mean(y_cal))
            p2_tr = adjust_probs_for_target_prevalence(p2_tr, pi_cal_s, pi_target_override)
            p2_c = adjust_probs_for_target_prevalence(p2_c, pi_cal_s, pi_target_override)
            p2_t = adjust_probs_for_target_prevalence(p2_t, pi_cal_s, pi_target_override)
        stage2_cache[key] = (p2_tr, p2_c, p2_t)
        return p2_tr, p2_c, p2_t

    gamma_q = contract.per_instance_query_cost / contract.cost_fn

    def _eval_arm(sel_ids: list[str], *, use_fluid: bool = True, use_2d_voi: bool = True) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p2_tr, p2_c, p2_t = _get_s2_probs(sel_ids, use_fluid=use_fluid)
        if not sel_ids:
            voi_t = np.zeros(len(y_te), dtype=float)
            q_t = np.zeros(len(y_te), dtype=bool)
        else:
            voi_t = fit_predict_prepurchase_voi(
                xb_tr, p1_oof_eval, p2_tr, xb_cal, p1_cal_eval, p2_c, xb_te, p1_te_eval,
                contract, use_dispersion_2d=use_2d_voi, seed=seed,
            )
            q_t = voi_t > contract.per_instance_query_cost

        p_rt = np.where(q_t, p2_t, p1_te_eval)
        nb_rt = weighted_net_benefit_per_1k(y_te, p_rt, q_t, contract, sample_weight=w_te)
        nb_100 = weighted_net_benefit_per_1k(y_te, p2_t, np.ones(len(y_te), dtype=bool), contract, sample_weight=w_te)
        q_rate = float(np.mean(q_t)) if w_te is None else float(np.sum(w_te * q_t) / np.sum(w_te))

        auprc_full = float(average_precision_score(y_te, p2_t, sample_weight=w_te))
        auprc_routed = float(average_precision_score(y_te, p_rt, sample_weight=w_te))
        auroc_full = float(roc_auc_score(y_te, p2_t, sample_weight=w_te))
        auroc_routed = float(roc_auc_score(y_te, p_rt, sample_weight=w_te))
        auprc_sample_unweighted = float(average_precision_score(y_te, p2_t))

        skip_audit = skipped_case_disagreement_audit(
            y_te, p1_te_eval, p2_t, q_t, clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed
        )
        cal_full = compute_calibration_metrics(y_te, p2_t, sample_weight=w_te)
        cal_rt = compute_calibration_metrics(y_te, p_rt, sample_weight=w_te)
        dca = compute_weighted_decision_curve(y_te, p_rt, q_t, dca_taus, gamma_q, sample_weight=w_te)

        spent = round(sum(ch_map[c].cost for c in sel_ids), 3)
        return (
            {
                "selected": list(sel_ids),
                "n_channels": len(sel_ids),
                "spent_budget": spent,
                "null_selected": any(c.startswith("ch_null") for c in sel_ids),
                "nb_routed": round(nb_rt, 4),
                "snb_routed": round(standardized_net_benefit(nb_rt, eval_prev), 4),
                "nb_100pct": round(nb_100, 4),
                "query_rate": round(q_rate, 4),
                "auprc_full": round(auprc_full, 4),
                "auprc_routed": round(auprc_routed, 4),
                "auprc_sample_unweighted": round(auprc_sample_unweighted, 4),
                "auroc_full_supplement": round(auroc_full, 4),
                "auroc_routed_supplement": round(auroc_routed, 4),
                "skip_disagreement": skip_audit,
                "calibration_full": cal_full,
                "calibration_routed": cal_rt,
                "decision_curve": dca,
            },
            p_rt,
            q_t,
            p2_t,
            voi_t,
        )

    r_s1, pr_s1, q_s1, _, _ = _eval_arm([], use_fluid=True, use_2d_voi=True)
    r_glob_gbdt, pr_gg, q_gg, _, _ = _eval_arm(disc_glob.selected_ids, use_fluid=False, use_2d_voi=True)
    r_glob_fluid, pr_gf, q_gf, _, _ = _eval_arm(disc_glob.selected_ids, use_fluid=True, use_2d_voi=True)
    r_wcz_fluid, pr_wcz, q_wcz, _, _ = _eval_arm(disc_wcz.selected_ids, use_fluid=True, use_2d_voi=True)
    r_ipw_fluid, pr_ipw, q_ipw, _, _ = _eval_arm(disc_ipw.selected_ids, use_fluid=True, use_2d_voi=True)
    r_off_ll_fluid, pr_oll, q_oll, _, _ = _eval_arm(disc_off_ll.selected_ids, use_fluid=True, use_2d_voi=True)
    r_off_cz_fluid, pr_ocz, q_ocz, _, _ = _eval_arm(disc_off_cz.selected_ids, use_fluid=True, use_2d_voi=True)
    r_upca_12b, pr_u12, q_u12, _, _ = _eval_arm(upca_sel, use_fluid=True, use_2d_voi=False)
    r_upca, pr_upca, q_upca, p2_upca, voi_upca = _eval_arm(upca_sel, use_fluid=True, use_2d_voi=True)

    # 10,000-Replicate Patient-Cluster Bootstraps (`R1, R14, R21`)
    boots = {
        "upca_vs_stage1": patient_cluster_bootstrap_diff(
            y_te, pr_upca, q_upca, pr_s1, q_s1, clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed + 1
        ),
        "upca_vs_global_cmi_fluid": patient_cluster_bootstrap_diff(
            y_te, pr_upca, q_upca, pr_gf, q_gf, clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed + 2
        ),
        "upca_vs_weighted_pilot_cz_fluid": patient_cluster_bootstrap_diff(
            y_te, pr_upca, q_upca, pr_wcz, q_wcz, clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed + 3
        ),
        "upca_vs_ipw_pilot_cz_fluid": patient_cluster_bootstrap_diff(
            y_te, pr_upca, q_upca, pr_ipw, q_ipw, clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed + 4
        ),
        "upca_vs_offset_stage_a_only": patient_cluster_bootstrap_diff(
            y_te, pr_upca, q_upca, pr_ocz, q_ocz, clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed + 5
        ),
        "upca_2d_vs_12bin_voi": patient_cluster_bootstrap_diff(
            y_te, pr_upca, q_upca, pr_u12, q_u12, clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed + 6
        ),
        "global_cmi_fluid_vs_global_gbdt": patient_cluster_bootstrap_diff(
            y_te, pr_gf, q_gf, pr_gg, q_gg, clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed + 7
        ),
        "offset_schervish_vs_offset_logloss": patient_cluster_bootstrap_diff(
            y_te, pr_ocz, q_ocz, pr_oll, q_oll, clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed + 8
        ),
        "upca_routed_vs_100pct_query": patient_cluster_bootstrap_diff(
            y_te, pr_upca, q_upca, p2_upca, np.ones(len(y_te), dtype=bool), clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed + 9
        ),
    }

    # Predicted VOI vs Held-Out Realized Decision Gain (`R23`)
    c_noq = ClinicalContract(contract.cost_fp, contract.cost_fn, contract.target_precision, 0.0)
    realized_gain_raw_cost = (
        unit_net_benefit_vec(y_te, p2_upca, np.zeros(len(y_te)), c_noq)
        - unit_net_benefit_vec(y_te, p1_te_eval, np.zeros(len(y_te)), c_noq)
    ) * (contract.cost_fn / 1000.0)
    voi_quartiles = []
    if len(upca_sel) > 0 and float(np.max(voi_upca)) > 0.0:
        q_cuts = np.quantile(voi_upca, [0.0, 0.25, 0.50, 0.75, 1.0])
        q_cuts[0], q_cuts[-1] = -1.0, np.inf
        vb = np.clip(np.digitize(voi_upca, q_cuts[1:-1]), 0, 3)
        for qi in range(4):
            m = vb == qi
            if int(np.sum(m)) > 0:
                wm = np.ones(int(np.sum(m))) if w_te is None else w_te[m]
                voi_quartiles.append({
                    "quartile": f"Q{qi+1}",
                    "n": int(np.sum(m)),
                    "mean_predicted_voi": round(float(np.sum(wm * voi_upca[m]) / np.sum(wm)), 5),
                    "mean_realized_decision_gain": round(float(np.sum(wm * realized_gain_raw_cost[m]) / np.sum(wm)), 5),
                    "query_cost_threshold": contract.per_instance_query_cost,
                    "queried_share": round(float(np.sum(wm * q_upca[m]) / np.sum(wm)), 4),
                })

    print(
        f"\n[{task_label}] (N_te={len(y_te):,}, Pats={len(np.unique(clusters_te)):,}, Prev={eval_prev*100:.2f}%, tau*={contract.tau_star:.4f})"
    )
    print(f"  Stage-1                 : NB={r_s1['nb_routed']:+7.2f} | AUPRC_full={r_s1['auprc_full']:.4f}")
    print(f"  Global-CMI + GlobalGBDT : NB={r_glob_gbdt['nb_routed']:+7.2f} | AUPRC_full={r_glob_gbdt['auprc_full']:.4f} | AUPRC_rt={r_glob_gbdt['auprc_routed']:.4f} | q={r_glob_gbdt['query_rate']*100:4.1f}% | sel={r_glob_gbdt['selected']}")
    print(f"  Global-CMI + Fluid      : NB={r_glob_fluid['nb_routed']:+7.2f} | AUPRC_full={r_glob_fluid['auprc_full']:.4f} | AUPRC_rt={r_glob_fluid['auprc_routed']:.4f} | q={r_glob_fluid['query_rate']*100:4.1f}% | sel={r_glob_fluid['selected']}")
    print(f"  Weighted-Pilot CZ+Fluid : NB={r_wcz_fluid['nb_routed']:+7.2f} | AUPRC_full={r_wcz_fluid['auprc_full']:.4f} | AUPRC_rt={r_wcz_fluid['auprc_routed']:.4f} | q={r_wcz_fluid['query_rate']*100:4.1f}% | sel={r_wcz_fluid['selected']}")
    print(f"  IPW-Pilot CZ (w_HT)+Fl  : NB={r_ipw_fluid['nb_routed']:+7.2f} | AUPRC_full={r_ipw_fluid['auprc_full']:.4f} | AUPRC_rt={r_ipw_fluid['auprc_routed']:.4f} | q={r_ipw_fluid['query_rate']*100:4.1f}% | sel={r_ipw_fluid['selected']}")
    print(f"  Offset-Pilot CZ (StgA)  : NB={r_off_cz_fluid['nb_routed']:+7.2f} | AUPRC_full={r_off_cz_fluid['auprc_full']:.4f} | AUPRC_rt={r_off_cz_fluid['auprc_routed']:.4f} | q={r_off_cz_fluid['query_rate']*100:4.1f}% | sel={r_off_cz_fluid['selected']}")
    print(f"  UPCA-Fluid (StgA+StgB)  : NB={r_upca['nb_routed']:+7.2f} | AUPRC_full={r_upca['auprc_full']:.4f} | AUPRC_rt={r_upca['auprc_routed']:.4f} | q={r_upca['query_rate']*100:4.1f}% | sel={r_upca['selected']}")
    print(f"  10k Cluster Boot UPCA vs Global-CMI+Fluid : {boots['upca_vs_global_cmi_fluid']['diff_mean']:+.2f} [{boots['upca_vs_global_cmi_fluid']['ci95_low']:+.2f}, {boots['upca_vs_global_cmi_fluid']['ci95_high']:+.2f}], p={boots['upca_vs_global_cmi_fluid']['p_two_sided']:.5f}")
    print(f"  10k Cluster Boot UPCA vs Weighted-CZ+Fluid: {boots['upca_vs_weighted_pilot_cz_fluid']['diff_mean']:+.2f} [{boots['upca_vs_weighted_pilot_cz_fluid']['ci95_low']:+.2f}, {boots['upca_vs_weighted_pilot_cz_fluid']['ci95_high']:+.2f}], p={boots['upca_vs_weighted_pilot_cz_fluid']['p_two_sided']:.5f}")
    print(f"  10k Cluster Boot UPCA vs Stage-1          : {boots['upca_vs_stage1']['diff_mean']:+.2f} [{boots['upca_vs_stage1']['ci95_low']:+.2f}, {boots['upca_vs_stage1']['ci95_high']:+.2f}], p={boots['upca_vs_stage1']['p_two_sided']:.5f}", flush=True)

    traces = {
        "y_te": y_te,
        "clusters_te": clusters_te,
        "w_te": np.ones(len(y_te), dtype=float) if w_te is None else w_te,
        "p1_te": p1_te_eval,
        "p_upca": pr_upca,
        "q_upca": q_upca,
        "p2_upca": p2_upca,
        "p_wcz": pr_wcz,
        "q_wcz": q_wcz,
        "p_gf": pr_gf,
        "q_gf": q_gf,
        "p1_oof_eval": p1_oof_eval,
        "p1_oof_raw": p1_oof_raw,
        "p1_cal_eval": p1_cal_eval,
        "p1_cal_raw": p1_cal_raw,
    }

    summary = {
        "task_id": task_id,
        "task_label": task_label,
        "is_prespecified_primary": is_prespecified_primary,
        "n_train": int(len(tr)),
        "n_cal": int(len(cal)),
        "n_test": int(len(te)),
        "n_test_patients": int(len(np.unique(clusters_te))),
        "test_events": int(np.sum(y_te == 1)),
        "sample_test_prevalence": round(float(np.mean(y_te)), 5),
        "target_eval_prevalence": round(eval_prev, 5),
        "contract": {
            "tau_star": round(contract.tau_star, 5),
            "cost_fp": contract.cost_fp,
            "cost_fn": contract.cost_fn,
            "c_query": contract.per_instance_query_cost,
            "gamma_cquery_over_cfn": round(gamma_q, 6),
            "budget": budget,
        },
        "arms": {
            "stage1": r_s1,
            "global_cmi_global_gbdt": r_glob_gbdt,
            "global_cmi_fluid": r_glob_fluid,
            "weighted_pilot_cz_fluid": r_wcz_fluid,
            "ipw_pilot_cz_fluid": r_ipw_fluid,
            "offset_pilot_logloss_fluid": r_off_ll_fluid,
            "offset_pilot_cz_stage_a_fluid": r_off_cz_fluid,
            "upca_12bin_voi": r_upca_12b,
            "upca_fluid": r_upca,
        },
        "stage_b_prune_log": prune_log,
        "within_stratum_pilot_independence": ind_check,
        "predicted_vs_realized_voi_quartiles": voi_quartiles,
        "bootstraps_10k_patient_cluster": boots,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    return summary, traces


def run_subgroup_and_missingness_audit(
    te_df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    traces: dict[str, np.ndarray],
    contract: ClinicalContract,
    seed: int = 42,
) -> dict[str, Any]:
    """Comprehensive Exploratory Subgroup Performance & Missingness Assessment (`R11`):
    Per-stratum N, events, prevalence, Stage-1 NB, UPCA NB, 10,000-rep patient-cluster bootstrap 95% CI on Delta-NB,
    query rate q_g%, group calibration intercept & slope, FNR & FPR at tau*, and Cochran Q heterogeneity test.
    """
    y_te = traces["y_te"]
    cl_te = traces["clusters_te"]
    w_te = traces["w_te"]
    p1 = traces["p1_te"]
    pu = traces["p_upca"]
    qu = traces["q_upca"]
    tau = contract.tau_star

    strata_out: dict[str, Any] = {}
    est_list: list[float] = []
    se_list: list[float] = []

    vals = te_df[group_col].astype(str).to_numpy()
    for idx_g, grp in enumerate(group_order):
        m = vals == grp
        n_g = int(np.sum(m))
        ev_g = int(np.sum(y_te[m] == 1))
        if n_g < 40 or ev_g < 2:
            continue
        yg = y_te[m]
        wg = w_te[m]
        cg = cl_te[m]
        p1g = p1[m]
        pug = pu[m]
        qug = qu[m]

        nb_s1 = weighted_net_benefit_per_1k(yg, p1g, np.zeros(n_g, dtype=bool), contract, sample_weight=wg)
        nb_up = weighted_net_benefit_per_1k(yg, pug, qug, contract, sample_weight=wg)
        boot_g = patient_cluster_bootstrap_diff(
            yg, pug, qug, p1g, np.zeros(n_g, dtype=bool), cg, contract,
            sample_weight=wg, n_boot=10_000, seed=seed + 50 + idx_g,
        )
        cal_g = compute_calibration_metrics(yg, pug, sample_weight=wg, n_bins=5)
        act_g = pug >= tau
        fnr_g = float(np.sum(wg * ((~act_g) & (yg == 1))) / max(1e-9, float(np.sum(wg * (yg == 1)))))
        fpr_g = float(np.sum(wg * (act_g & (yg == 0))) / max(1e-9, float(np.sum(wg * (yg == 0)))))
        q_rate_g = float(np.sum(wg * qug) / max(1e-9, float(np.sum(wg))))

        est_list.append(boot_g["diff_mean"])
        se_list.append(max(boot_g["boot_se"], 1e-4))
        strata_out[grp] = {
            "n_encounters": n_g,
            "n_patients": int(len(np.unique(cg))),
            "events": ev_g,
            "prevalence": round(float(np.sum(wg * yg) / np.sum(wg)), 5),
            "nb_stage1": round(nb_s1, 3),
            "nb_upca": round(nb_up, 3),
            "delta_nb": boot_g["diff_mean"],
            "delta_nb_ci95": [boot_g["ci95_low"], boot_g["ci95_high"]],
            "delta_nb_p_two_sided": boot_g["p_two_sided"],
            "query_rate": round(q_rate_g, 4),
            "cal_intercept": cal_g["cal_intercept"],
            "cal_slope": cal_g["cal_slope"],
            "ici": cal_g["ici"],
            "fnr_at_tau": round(fnr_g, 4),
            "fpr_at_tau": round(fpr_g, 4),
        }

    return {
        "strata": strata_out,
        "heterogeneity": cochran_q_heterogeneity(est_list, se_list),
    }


def run_two_phase_efficiency_comparison(
    df_mc: pd.DataFrame,
    base_cols: list[str],
    channels: list[ChannelDef],
    contract: ClinicalContract,
    n_reps: int = 100,
) -> dict[str, Any]:
    """R15 Efficiency Comparison: Weighted-Pilot (0.05 floor) vs IPW (w_HT) vs Parametric Two-Phase
    Offset Logistic MLE (Breslow-Cain / Tao-Zeng-Lin 2017) vs Unweighted Breslow-Cain Offset GBDT
    against the 100% Full-Cohort Oracle across 100 repeated 20% Phase-2 pilots on Stanford Critical ICU/Death.
    """
    print("\n=== R15: Two-Phase Pilot Efficiency Benchmark (100 Repeated 20% Pilots on Stanford ICU/Death) ===")
    tr = df_mc[df_mc["split"] == "train"].reset_index(drop=True)
    y_tr = tr["y_critical_icu_death"].astype(int).to_numpy()
    p1_oof = compute_oof_stage1(tr[base_cols], y_tr, n_folds=3, seed=44)
    tau = contract.tau_star

    # Compute 100% full-cohort oracle Step-1 Schervish proper-score gain for each of the 8 clinical channels
    oracle_outcome, _ = discover_stage_a_ablation(
        tr, y_tr, base_cols, channels, p1_oof, contract, budget=3.0,
        mode="offset_cz_schervish", pilot_fraction=1.0, seed=44,
    )
    oracle_vec = np.array([oracle_outcome.step1_scores[ch.channel_id] for ch in channels], dtype=float)
    oracle_top1 = channels[int(np.argmax(oracle_vec / np.array([ch.cost for ch in channels])))].channel_id

    # Also evaluate parametric two-phase offset MLE (Tao-Zeng-Lin 2017 / Breslow-Cain logistic sieve) vs IPW logistic
    enc_b = FastFrameEncoder().fit(tr[base_cols])
    xb_all = enc_b.transform(tr[base_cols])
    ch_mats = {ch.channel_id: FastFrameEncoder().fit_transform(tr[list(ch.columns)]) if hasattr(FastFrameEncoder(), "fit_transform") else FastFrameEncoder().fit(tr[list(ch.columns)]).transform(tr[list(ch.columns)]) for ch in channels}

    # Oracle parametric logistic coefficient norm on standardized first column of ch_lab_lactate_cardiac
    z_lac = (ch_mats["ch_lab_lactate_cardiac"][:, :1] - np.mean(ch_mats["ch_lab_lactate_cardiac"][:, :1])) / (
        np.std(ch_mats["ch_lab_lactate_cardiac"][:, :1]) + 1e-6
    )
    ell1 = logit(np.clip(p1_oof, 1e-5, 1.0 - 1e-5))
    x_param_all = np.column_stack([ell1, z_lac, z_lac ** 2])
    lr_full = LogisticRegression(C=100.0, solver="lbfgs", max_iter=500).fit(x_param_all, y_tr)
    beta_oracle = lr_full.coef_[0, 1]

    rmse_scores: dict[str, list[float]] = {"weighted_005": [], "ipw_wht": [], "offset_bc_gbdt": []}
    top1_match: dict[str, list[bool]] = {"weighted_005": [], "ipw_wht": [], "offset_bc_gbdt": []}
    beta_err_ipw, beta_err_tzl_offset = [], []

    quantiles = np.quantile(ell1, np.linspace(0.0, 1.0, 9))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    bin_all = np.clip(np.digitize(ell1, quantiles[1:-1], right=False), 0, 7)

    # Use 25 repeated full GBDT discovery pilots + 200 parametric Tao-Zeng-Lin vs IPW replicates for speed
    for rep in range(200):
        pilot = sample_pilot(y_tr, p1_oof, pilot_fraction=0.20, mode="collision_neyman", tau_star=tau, n_bins=8, seed=700 + rep)
        p_idx = pilot.indices
        yp = y_tr[p_idx]
        w_ht = pilot.ht_weights
        bin_p = bin_all[p_idx]
        delta_p = np.zeros(yp.size, dtype=float)
        for b_id in range(8):
            N1 = max(1, int(np.sum((y_tr == 1) & (bin_all == b_id))))
            N0 = max(1, int(np.sum((y_tr == 0) & (bin_all == b_id))))
            n1 = max(1, int(np.sum((yp == 1) & (bin_p == b_id))))
            n0 = max(1, int(np.sum((yp == 0) & (bin_p == b_id))))
            delta_p[bin_p == b_id] = float(np.log((n1 / float(N1)) / (n0 / float(N0))))

        # Parametric IPW vs Two-Phase Offset MLE (Breslow-Cain 1988 / Tao-Zeng-Lin 2017)
        xp = x_param_all[p_idx]
        lr_ipw = LogisticRegression(C=100.0, solver="lbfgs", max_iter=500).fit(xp, yp, sample_weight=w_ht)
        beta_err_ipw.append((float(lr_ipw.coef_[0, 1]) - float(beta_oracle)) ** 2)

        # Unweighted conditional likelihood with exact stratum offset delta_p via Newton-Raphson
        b_vec = np.zeros(4, dtype=float)  # [intercept, ell1, z_lac, z_lac^2]
        X_des = np.column_stack([np.ones(yp.size), xp])
        for _ in range(18):
            eta = np.clip(delta_p + X_des @ b_vec, -25.0, 25.0)
            q = expit(eta)
            grad = X_des.T @ (yp - q) - 0.01 * b_vec
            W_d = q * (1.0 - q)
            H = (X_des.T * W_d) @ X_des + 0.01 * np.eye(4)
            step = np.linalg.solve(H, grad)
            b_vec += step
            if float(np.max(np.abs(step))) < 1e-7:
                break
        beta_err_tzl_offset.append((float(b_vec[2]) - float(beta_oracle)) ** 2)

        if rep < 25:
            for key_m, mode_m in [
                ("weighted_005", "weighted_cz_005"),
                ("ipw_wht", "ipw_cz_wht"),
                ("offset_bc_gbdt", "offset_cz_schervish"),
            ]:
                out_m, _ = discover_stage_a_ablation(
                    tr, y_tr, base_cols, channels, p1_oof, contract, budget=3.0,
                    mode=mode_m, pilot_fraction=0.20, seed=700 + rep,
                )
                vec_m = np.array([out_m.step1_scores[ch.channel_id] for ch in channels], dtype=float)
                rmse_scores[key_m].append(float(np.sqrt(np.mean((vec_m - oracle_vec) ** 2))))
                top1_m = channels[int(np.argmax(vec_m / np.array([ch.cost for ch in channels])))].channel_id
                top1_match[key_m].append(top1_m == oracle_top1)

    res_eff = {
        "oracle_top1_channel": oracle_top1,
        "n_gbdt_pilot_reps": 25,
        "n_parametric_pilot_reps": 200,
        "gbdt_step1_score_rmse_vs_full_cohort": {k: round(float(np.mean(v)), 6) for k, v in rmse_scores.items()},
        "gbdt_top1_channel_concordance": {k: round(float(np.mean(v)), 4) for k, v in top1_match.items()},
        "parametric_lactate_slope_rmse": {
            "ipw_logistic_wht": round(float(np.sqrt(np.mean(beta_err_ipw))), 5),
            "two_phase_offset_mle_tao_zeng_lin": round(float(np.sqrt(np.mean(beta_err_tzl_offset))), 5),
            "relative_variance_reduction_pct": round(
                100.0 * (1.0 - float(np.mean(beta_err_tzl_offset)) / max(1e-12, float(np.mean(beta_err_ipw)))), 2
            ),
        },
    }
    print("  R15 Efficiency Results:", json.dumps(res_eff, indent=2), flush=True)
    return res_eff


def run_hyperparameter_and_cost_sensitivity(
    df_mc: pd.DataFrame,
    mc_base: list[str],
    mc_channels: list[ChannelDef],
    df_bwh: pd.DataFrame,
    bwh_base: list[str],
    bwh_channels: list[ChannelDef],
) -> dict[str, Any]:
    """R10 Sensitivity Sweeps over prespecified constants (`rho_pilot`, `B_score`, dispersion cut `q_disp`, `alpha_pol`)
    and per-query cost `c_query = gamma * C_FN` on `Stanford Critical ICU/Death` and `BWH 10d Stent/CABG`.
    """
    print("\n=== R10: Prespecified Constant & Query-Cost Sensitivity Sweeps ===")
    out_sens: dict[str, Any] = {}

    for label, df_s, y_col, cl_col, base_c, ch_list, c_fn, base_cq, budget, seed_s in [
        ("stanford_icu_death", df_mc, "y_critical_icu_death", "MRN", mc_base, mc_channels, 19.0, 0.030, 3.00, 44),
        ("bwh_stent_cabg_10d", df_bwh, "stent_or_cabg_010_day", "patient_ngsci_id", bwh_base, bwh_channels, 32.33, 0.020, 1.80, 43),
    ]:
        tr = df_s[df_s["split"] == "train"].reset_index(drop=True)
        cal = df_s[df_s["split"] == "cal"].reset_index(drop=True)
        te = df_s[df_s["split"] == "test"].reset_index(drop=True)
        y_tr, y_cal, y_te = tr[y_col].astype(int).to_numpy(), cal[y_col].astype(int).to_numpy(), te[y_col].astype(int).to_numpy()
        p1_oof = compute_oof_stage1(tr[base_c], y_tr, n_folds=3, seed=seed_s)
        p1_cal, p1_tests = fit_stage1_predictions(tr, y_tr, cal, y_cal, [te], base_c, seed=seed_s)
        p1_te = p1_tests[0]
        enc_b = FastFrameEncoder().fit(tr[base_c])
        xb_tr, xb_cal, xb_te = enc_b.transform(tr[base_c]), enc_b.transform(cal[base_c]), enc_b.transform(te[base_c])
        ch_map = {c.channel_id: c for c in ch_list}

        s2_fit_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        def _fit_cached(sel_ids: list[str], contract_k: ClinicalContract) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            key = tuple(sel_ids)
            if key in s2_fit_cache:
                return s2_fit_cache[key]
            if not sel_ids:
                return p1_oof, p1_cal, p1_te
            cols = [c for cid in sel_ids for c in ch_map[cid].columns]
            enc_z = FastFrameEncoder().fit(tr[cols])
            z_tr, z_cal, z_te = enc_z.transform(tr[cols]), enc_z.transform(cal[cols]), enc_z.transform(te[cols])
            clf = FluidCollisionBackboneClassifier(tau_star=contract_k.tau_star, kappa_max=20.0, min_cz_ess=240.0, seed=seed_s + 7).fit(
                np.hstack([xb_tr, z_tr]), z_tr, y_tr, logit(np.clip(p1_oof, 1e-5, 1 - 1e-5))
            )
            r_tr = clf.decision_function(np.hstack([xb_tr, z_tr]), z_tr, logit(np.clip(p1_oof, 1e-5, 1 - 1e-5)))
            r_c = clf.decision_function(np.hstack([xb_cal, z_cal]), z_cal, logit(p1_cal))
            r_t = clf.decision_function(np.hstack([xb_te, z_te]), z_te, logit(p1_te))
            pl = fit_platt_calibrator(r_c, y_cal)
            res = (
                np.clip(apply_platt(r_tr, pl), 1e-6, 1 - 1e-6),
                np.clip(apply_platt(r_c, pl), 1e-6, 1 - 1e-6),
                np.clip(apply_platt(r_t, pl), 1e-6, 1 - 1e-6),
            )
            s2_fit_cache[key] = res
            return res

        def _run_cfg(rho_p: float, b_bins: int, q_disp: float, alpha_p: float, cq: float) -> dict[str, Any]:
            ck = ClinicalContract(cost_fp=1.0, cost_fn=c_fn, target_precision=0.25, per_instance_query_cost=cq)
            disc, _ = discover_stage_a_ablation(
                tr, y_tr, base_c, ch_list, p1_oof, ck, budget,
                mode="offset_cz_schervish", pilot_fraction=rho_p, n_score_bins=b_bins, seed=seed_s,
            )
            sel, _ = oof_policy_screening_stage_b(
                disc.selected_ids, tr, y_tr, p1_oof, base_c, ch_map, ck,
                dispersion_quantile=q_disp, alpha_pol=alpha_p, seed=seed_s,
            )
            p2_tr, p2_c, p2_t = _fit_cached(sel, ck)
            if not sel:
                q_t = np.zeros(len(y_te), dtype=bool)
            else:
                voi_t = fit_predict_prepurchase_voi(
                    xb_tr, p1_oof, p2_tr, xb_cal, p1_cal, p2_c, xb_te, p1_te, ck,
                    use_dispersion_2d=True, dispersion_quantile=q_disp, seed=seed_s,
                )
                q_t = voi_t > cq
            p_rt = np.where(q_t, p2_t, p1_te)
            nb_s1 = weighted_net_benefit_per_1k(y_te, p1_te, np.zeros(len(y_te), dtype=bool), ck)
            nb_up = weighted_net_benefit_per_1k(y_te, p_rt, q_t, ck)
            nb_100 = weighted_net_benefit_per_1k(y_te, p2_t, np.ones(len(y_te), dtype=bool), ck)
            return {
                "rho_pilot": rho_p,
                "b_score": b_bins,
                "q_disp": q_disp,
                "alpha_pol": alpha_p,
                "c_query": cq,
                "gamma_cquery_over_cfn": round(cq / c_fn, 6),
                "selected": sel,
                "nb_stage1": round(nb_s1, 3),
                "nb_upca": round(nb_up, 3),
                "nb_100pct": round(nb_100, 3),
                "delta_nb_vs_s1": round(nb_up - nb_s1, 3),
                "query_rate": round(float(np.mean(q_t)), 4),
            }

        c_sweep = [_run_cfg(0.20, 8, 0.75, 0.10, cq) for cq in [0.005, 0.015, 0.030, 0.050, 0.080]]
        const_grid = [
            _run_cfg(0.15, 8, 0.75, 0.10, base_cq),
            _run_cfg(0.20, 8, 0.75, 0.10, base_cq),
            _run_cfg(0.30, 8, 0.75, 0.10, base_cq),
            _run_cfg(0.20, 6, 0.75, 0.10, base_cq),
            _run_cfg(0.20, 12, 0.75, 0.10, base_cq),
            _run_cfg(0.20, 8, 0.65, 0.10, base_cq),
            _run_cfg(0.20, 8, 0.85, 0.10, base_cq),
            _run_cfg(0.20, 8, 0.75, 0.05, base_cq),
            _run_cfg(0.20, 8, 0.75, 0.20, base_cq),
        ]
        out_sens[label] = {"c_query_sweep": c_sweep, "prespecified_constant_sensitivity": const_grid}
        print(f"  Completed sensitivity sweeps for {label}", flush=True)

    return out_sens


def run_clinical_phase() -> dict[str, Any]:
    t0 = time.time()
    print("\n" + "=" * 84)
    print("STEP 2A: AUDITED CLINICAL BENCHMARKS, 5 SINGLE-FACTOR ABLATIONS & 10K CLUSTER BOOTSTRAPS")
    print("=" * 84)

    # 1. Load Audited Stanford MC-MED Strict 2.0h Landmark Cohort (`N = 89,395`)
    mc = pd.read_parquet(DATA_DIR / "mcmed_stanford_2h_v2.parquet")
    mc["gender_male"] = mc["Gender"].astype(str).str.upper().str.startswith("M").astype(int)
    top_cc = mc.loc[mc["split"] == "train", "CC"].value_counts().head(25).index.tolist()
    mc["cc_clean"] = mc["CC"].where(mc["CC"].isin(top_cc), "OTHER_CC").astype(str)
    mc["arrival_clean"] = mc["Means_of_arrival"].fillna("Missing").astype(str)
    mc["payor_clean"] = mc["Payor_class"].fillna("Unspecified_or_Commercial").astype(str)
    mc["race_clean"] = np.where(
        mc["Race"].isin(["White", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander"]),
        mc["Race"].astype(str),
        "Other_or_Unknown",
    )

    mc_base = [
        "Age", "gender_male", "Triage_Temp", "Triage_HR", "Triage_RR",
        "Triage_SpO2", "Triage_SBP", "Triage_DBP", "esi_num", "cc_clean", "arrival_clean",
    ]
    mc_channels = [
        ChannelDef("ch_cdw_prior_pmh", tuple(c for c in mc.columns if c.startswith("cdw_pmh_")), 0.40),
        ChannelDef("ch_cdw_prior_meds", tuple(c for c in mc.columns if c.startswith("cdw_meds_")), 0.40),
        ChannelDef("ch_lab_cbc", ("lab_cbc_wbc", "lab_cbc_hgb", "lab_cbc_plt"), 0.78),
        ChannelDef("ch_lab_renal_elyte", ("lab_renal_cr", "lab_renal_bun", "lab_elyte_na", "lab_elyte_k", "lab_elyte_hco3", "lab_gluc"), 0.92),
        ChannelDef("ch_lab_lactate_cardiac", ("lab_lactate", "lab_troponin", "lab_bnp"), 1.15),
        ChannelDef("ch_lab_lft_coag", ("lab_lft_ast", "lab_lft_alt", "lab_lft_bili", "lab_coag_inr"), 0.95),
        ChannelDef("ch_rads_2h", tuple(c for c in mc.columns if c.startswith("rad_")), 1.50),
        ChannelDef("ch_monitor_numerics_2h", tuple(c for c in mc.columns if c.startswith("mon_")), 0.60),
    ]

    # Task 1 (Prespecified Primary Multi-Channel Rare-Event Regime): Stanford Critical ICU/Death
    res_mc_icu, tr_mc_icu = evaluate_clinical_task_full(
        "mcmed_critical_icu_death",
        "Stanford MC-MED Critical ICU/Death (<=2h)",
        mc, "y_critical_icu_death", "MRN", mc_base, mc_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.030),
        budget=3.00,
        dca_taus=list(np.round(np.linspace(0.01, 0.12, 23), 4)),
        is_prespecified_primary=True,
        seed=44,
    )

    # Task 2: Stanford Hospital Admission
    res_mc_adm, tr_mc_adm = evaluate_clinical_task_full(
        "mcmed_hospital_admission",
        "Stanford MC-MED Hospital Admission (<=2h)",
        mc, "y_admit", "MRN", mc_base, mc_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=5.67, target_precision=0.50, per_instance_query_cost=0.025),
        budget=3.00,
        dca_taus=list(np.round(np.linspace(0.05, 0.45, 21), 4)),
        is_prespecified_primary=True,
        seed=45,
    )

    # Task 3: Stanford 72h Discharged Bounceback w/ Hospitalization (Strict 2.0h Landmark)
    mc_disch_2h = mc[mc["is_discharged"] == 1].copy().reset_index(drop=True)
    res_mc_b72, tr_mc_b72 = evaluate_clinical_task_full(
        "mcmed_bounce72_admit_2h",
        "Stanford MC-MED 72h Bounceback Admit (<=2h Landmark)",
        mc_disch_2h, "y_bounce72_admit", "MRN", mc_base, mc_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=32.33, target_precision=0.15, per_instance_query_cost=0.025),
        budget=3.00,
        dca_taus=list(np.round(np.linspace(0.005, 0.08, 16), 4)),
        is_prespecified_primary=True,
        seed=46,
    )

    # Stanford Bounceback Sensitivities (`R11, R19`):
    # (a) Complete Prior-CDW Encounters (`has_prior_cdw_pmh == 1 & has_prior_cdw_meds == 1`)
    mc_disch_cdw_complete = mc_disch_2h[
        (mc_disch_2h["has_prior_cdw_pmh"] == 1) & (mc_disch_2h["has_prior_cdw_meds"] == 1)
    ].copy().reset_index(drop=True)
    res_mc_b72_cdw_comp, _ = evaluate_clinical_task_full(
        "mcmed_bounce72_complete_cdw_sensitivity",
        "Stanford 72h Bounceback (Complete Prior-CDW Subset)",
        mc_disch_cdw_complete, "y_bounce72_admit", "MRN", mc_base, mc_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=32.33, target_precision=0.15, per_instance_query_cost=0.025),
        budget=3.00,
        dca_taus=[0.01, 0.02, 0.03, 0.05],
        is_prespecified_primary=False,
        seed=146,
    )
    # (b) Arrival (`0h` CDW-only across all 63,060 discharged visits) & Discharge (`<= Dispo_time` labs + CDW)
    mc_dc_all = pd.read_parquet(DATA_DIR / "mcmed_discharged_sensitivity_v2.parquet")
    mc_dc_all["gender_male"] = mc_dc_all["Gender"].astype(str).str.upper().str.startswith("M").astype(int)
    mc_dc_all["cc_clean"] = mc_dc_all["CC"].where(mc_dc_all["CC"].isin(top_cc), "OTHER_CC").astype(str)
    mc_dc_all["arrival_clean"] = mc_dc_all["Means_of_arrival"].fillna("Missing").astype(str)
    ch_arrival_0h = [
        ChannelDef("ch_cdw_prior_pmh", tuple(c for c in mc_dc_all.columns if c.startswith("cdw_pmh_")), 0.40),
        ChannelDef("ch_cdw_prior_meds", tuple(c for c in mc_dc_all.columns if c.startswith("cdw_meds_")), 0.40),
    ]
    res_mc_b72_arr0h, _ = evaluate_clinical_task_full(
        "mcmed_bounce72_arrival_0h_sensitivity",
        "Stanford 72h Bounceback (All Discharged, 0h Arrival CDW Only)",
        mc_dc_all, "y_bounce72_admit", "MRN", mc_base, ch_arrival_0h,
        ClinicalContract(cost_fp=1.0, cost_fn=32.33, target_precision=0.15, per_instance_query_cost=0.025),
        budget=1.50,
        dca_taus=[0.01, 0.02, 0.03, 0.05],
        is_prespecified_primary=False,
        seed=246,
    )
    dc_lab_cols = [c for c in mc_dc_all.columns if c.endswith("_dc")]
    ch_discharge = ch_arrival_0h + (
        [ChannelDef("ch_labs_by_discharge", tuple(dc_lab_cols), 1.15)] if dc_lab_cols else []
    )
    res_mc_b72_disch_lm, _ = evaluate_clinical_task_full(
        "mcmed_bounce72_discharge_landmark_sensitivity",
        "Stanford 72h Bounceback (All Discharged, Discharge Landmark)",
        mc_dc_all, "y_bounce72_admit", "MRN", mc_base, ch_discharge,
        ClinicalContract(cost_fp=1.0, cost_fn=32.33, target_precision=0.15, per_instance_query_cost=0.025),
        budget=2.50,
        dca_taus=[0.01, 0.02, 0.03, 0.05],
        is_prespecified_primary=False,
        seed=346,
    )

    # 2. Load Audited BWH ED-ECG Cohort (`N = 51,158` Patient-Disjoint Holdout; Arrival-ECG Only -> ZERO Troponin Incorporation!)
    bwh = pd.read_parquet(DATA_DIR / "bwh_ed_ecg_v2.parquet")
    bwh_base = [
        c for c in [
            "age_at_admit", "sex_male", "vent_rate", "pr_interval",
            "qrs_duration", "qt_interval", "qtc_interval", "p_axes", "r_axes", "t_axes",
        ]
        if c in bwh.columns
    ]
    bwh_ecg_channels = [
        ChannelDef(
            "ch_ecg_ischemia_flags",
            tuple(c for c in ["has_st_eleva", "has_depress", "has_twave_inver", "has_twave_abnormal", "has_jpoint_eleva", "has_jpoint_repol", "has_st", "has_twave"] if c in bwh.columns),
            0.40,
        ),
        ChannelDef(
            "ch_ecg_rhythm_struct",
            tuple(c for c in ["has_afib", "has_bbb", "has_aberran_bbb", "has_lvh", "has_pacemaker", "has_prolonged_qt", "has_rhythm_disturbance", "has_normal_ecg", "has_normal_sinus"] if c in bwh.columns),
            0.40,
        ),
        ChannelDef(
            "ch_ecg_waveform",
            tuple(c for c in bwh.columns if c.startswith("wf_")),
            0.85,
        ),
    ]

    # Task 4: BWH 30d Mortality (Arrival ECG Only — Non-Incorporated Hard Endpoint!)
    c_bwh_death = ClinicalContract(cost_fp=1.0, cost_fn=32.33, target_precision=0.20, per_instance_query_cost=0.020)
    res_bwh_death30, tr_bwh_death30 = evaluate_clinical_task_full(
        "bwh_death_030_ecg_only",
        "BWH 30d Mortality (Arrival ECG Only, Zero Troponin)",
        bwh, "death_030_day", "patient_ngsci_id", bwh_base, bwh_ecg_channels,
        c_bwh_death, budget=1.80,
        dca_taus=list(np.round(np.linspace(0.005, 0.08, 16), 4)),
        is_prespecified_primary=True,
        seed=42,
    )

    # Task 5: BWH 10d Stent/CABG (Arrival ECG Only — Non-Incorporated Hard Endpoint!)
    c_bwh_revasc = ClinicalContract(cost_fp=1.0, cost_fn=32.33, target_precision=0.20, per_instance_query_cost=0.020)
    res_bwh_stent10, tr_bwh_stent10 = evaluate_clinical_task_full(
        "bwh_stent_cabg_010_ecg_only",
        "BWH 10d Stent/CABG (Arrival ECG Only, Zero Troponin)",
        bwh, "stent_or_cabg_010_day", "patient_ngsci_id", bwh_base, bwh_ecg_channels,
        c_bwh_revasc, budget=1.80,
        dca_taus=list(np.round(np.linspace(0.005, 0.07, 14), 4)),
        is_prespecified_primary=True,
        seed=43,
    )

    # Secondary BWH ECG-Only 30d MACE/Death (ZERO troponin predictor -> zero incorporation bias!)
    c_bwh_mace_ecg = ClinicalContract(cost_fp=1.0, cost_fn=15.0, target_precision=0.25, per_instance_query_cost=0.020)
    res_bwh_mace30_ecg, tr_bwh_mace30_ecg = evaluate_clinical_task_full(
        "bwh_mace_death_030_ecg_only_secondary",
        "BWH 30d MACE/Death (Secondary Arrival ECG Only, Zero Troponin)",
        bwh, "macetrop_pos_or_death_030", "patient_ngsci_id", bwh_base, bwh_ecg_channels,
        c_bwh_mace_ecg, budget=1.80,
        dca_taus=list(np.round(np.linspace(0.02, 0.20, 19), 4)),
        is_prespecified_primary=False,
        seed=142,
    )

    # BWH Subgroup Slice Audits (`R1, R18, R23`):
    # (i) Encounters without 10-day stress test or catheterization (`test_010_day == 0`, N_test = 5,299)
    # (ii) Encounters without same-day troponin (`trop_drawn_sameday == 0`, N_test = 4,104)
    te_bwh = bwh[bwh["split"] == "test"].reset_index(drop=True)
    tr_bwh_df = bwh[bwh["split"] == "train"].reset_index(drop=True)
    cal_bwh_df = bwh[bwh["split"] == "cal"].reset_index(drop=True)

    # Train a selective-label `Tested-Only` (trained exclusively on `test_010_day == 1`) ECG model for 30d mortality & 30d MACE/death
    all_ecg_cols = bwh_base + [c for ch in bwh_ecg_channels for c in ch.columns]
    enc_sl = FastFrameEncoder().fit(tr_bwh_df.loc[tr_bwh_df["test_010_day"] == 1, all_ecg_cols])

    def _slice_audit_bwh(y_name: str, tr_dict: dict[str, np.ndarray], ck: ClinicalContract) -> dict[str, Any]:
        tr_tested = tr_bwh_df[tr_bwh_df["test_010_day"] == 1]
        cal_tested = cal_bwh_df[cal_bwh_df["test_010_day"] == 1]
        clf_to = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.05, num_leaves=31, min_child_samples=20,
            random_state=42, n_jobs=4, verbose=-1,
        ).fit(enc_sl.transform(tr_tested[all_ecg_cols]), tr_tested[y_name].astype(int).to_numpy())
        pl_to = fit_platt_calibrator(
            clf_to.predict(enc_sl.transform(cal_tested[all_ecg_cols]), raw_score=True),
            cal_tested[y_name].astype(int).to_numpy(),
        )
        p_tested_only_te = np.clip(apply_platt(clf_to.predict(enc_sl.transform(te_bwh[all_ecg_cols]), raw_score=True), pl_to), 1e-6, 1 - 1e-6)

        out_slices = {}
        for s_name, mask_s in [
            ("no_10d_stress_or_cath", (te_bwh["test_010_day"] == 0).to_numpy()),
            ("no_sameday_troponin", (te_bwh["trop_drawn_sameday"] == 0).to_numpy()),
        ]:
            ys = tr_dict["y_te"][mask_s]
            cls = tr_dict["clusters_te"][mask_s]
            q0 = np.zeros(int(np.sum(mask_s)), dtype=bool)
            nb_to = weighted_net_benefit_per_1k(ys, p_tested_only_te[mask_s], q0, ck)
            nb_s1 = weighted_net_benefit_per_1k(ys, tr_dict["p1_te"][mask_s], q0, ck)
            nb_gf = weighted_net_benefit_per_1k(ys, tr_dict["p_gf"][mask_s], tr_dict["q_gf"][mask_s], ck)
            nb_wcz = weighted_net_benefit_per_1k(ys, tr_dict["p_wcz"][mask_s], tr_dict["q_wcz"][mask_s], ck)
            nb_up = weighted_net_benefit_per_1k(ys, tr_dict["p_upca"][mask_s], tr_dict["q_upca"][mask_s], ck)
            boot_vs_to = patient_cluster_bootstrap_diff(
                ys, tr_dict["p_upca"][mask_s], tr_dict["q_upca"][mask_s], p_tested_only_te[mask_s], q0, cls, ck, n_boot=10_000, seed=91
            )
            boot_vs_s1 = patient_cluster_bootstrap_diff(
                ys, tr_dict["p_upca"][mask_s], tr_dict["q_upca"][mask_s], tr_dict["p1_te"][mask_s], q0, cls, ck, n_boot=10_000, seed=92
            )
            flagged_up = tr_dict["p_upca"][mask_s] >= ck.tau_star
            flagged_to = p_tested_only_te[mask_s] >= ck.tau_star
            out_slices[s_name] = {
                "n_encounters": int(np.sum(mask_s)),
                "events": int(np.sum(ys == 1)),
                "prevalence": round(float(np.mean(ys)), 5),
                "nb_tested_only_model": round(nb_to, 3),
                "flag_rate_tested_only": round(float(np.mean(flagged_to)), 4),
                "nb_stage1": round(nb_s1, 3),
                "nb_global_cmi_fluid": round(nb_gf, 3),
                "nb_weighted_pilot_cz_fluid": round(nb_wcz, 3),
                "nb_upca_fluid": round(nb_up, 3),
                "flag_rate_upca": round(float(np.mean(flagged_up)), 4),
                "retrospectively_flagged_events_upca": int(np.sum(flagged_up & (ys == 1))),
                "sensitivity_at_tau_upca": round(float(np.sum(flagged_up & (ys == 1)) / max(1, int(np.sum(ys == 1)))), 4),
                "boot_upca_vs_tested_only": boot_vs_to,
                "boot_upca_vs_stage1": boot_vs_s1,
            }
        return out_slices

    bwh_slice_audits = {
        "death_030_day": _slice_audit_bwh("death_030_day", tr_bwh_death30, c_bwh_death),
        "macetrop_pos_or_death_030_ecg_only": _slice_audit_bwh("macetrop_pos_or_death_030", tr_bwh_mace30_ecg, c_bwh_mace_ecg),
    }

    # (iii) BWH Complete-Troponin Subgroup (`trop_drawn_sameday == 1`, N = 20,082) on NON-INCORPORATED endpoints!
    bwh_trop = bwh[bwh["trop_drawn_sameday"] == 1].copy().reset_index(drop=True)
    bwh_trop_channels = bwh_ecg_channels + [
        ChannelDef("ch_sameday_troponin", ("maxtrop_sameday", "log_maxtrop_sameday"), 1.15),
    ]
    res_bwh_trop_death30, _ = evaluate_clinical_task_full(
        "bwh_complete_troponin_death_030",
        "BWH Complete-Troponin Subgroup: 30d Mortality (Non-Incorporated)",
        bwh_trop, "death_030_day", "patient_ngsci_id", bwh_base, bwh_trop_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.025),
        budget=2.40,
        dca_taus=[0.01, 0.03, 0.05, 0.08, 0.12],
        is_prespecified_primary=False,
        seed=242,
    )
    res_bwh_trop_stent10, _ = evaluate_clinical_task_full(
        "bwh_complete_troponin_stent_cabg_010",
        "BWH Complete-Troponin Subgroup: 10d Stent/CABG (Non-Incorporated)",
        bwh_trop, "stent_or_cabg_010_day", "patient_ngsci_id", bwh_base, bwh_trop_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.025),
        budget=2.40,
        dca_taus=[0.01, 0.03, 0.05, 0.08, 0.12],
        is_prespecified_primary=False,
        seed=243,
    )

    # Also Agniel Healthcare-Process Deconfounding Audit on Stanford ICU/Death (where lab/monitor ordering indicators exist!)
    mc_tr_i = mc[mc["split"] == "train"].reset_index(drop=True)
    mc_cal_i = mc[mc["split"] == "cal"].reset_index(drop=True)
    mc_te_i = mc[mc["split"] == "test"].reset_index(drop=True)
    sel_icu = res_mc_icu["arms"]["upca_fluid"]["selected"]
    s2_icu_cols = [c for cid in sel_icu for c in {ch.channel_id: ch for ch in mc_channels}[cid].columns]
    agniel_audit = {}
    if s2_icu_cols:
        dec = AgnielDeconfounder().fit(mc_tr_i[mc_base], mc_tr_i[s2_icu_cols])
        tr_dec = dec.transform(mc_tr_i[mc_base], mc_tr_i[s2_icu_cols], seed=44)
        cal_dec = dec.transform(mc_cal_i[mc_base], mc_cal_i[s2_icu_cols], seed=45)
        te_dec = dec.transform(mc_te_i[mc_base], mc_te_i[s2_icu_cols], seed=46)
        eb_i = FastFrameEncoder().fit(mc_tr_i[mc_base])
        ez_i = FastFrameEncoder(include_missing_indicators=False).fit(tr_dec)
        xb_tr_i, xb_cal_i, xb_te_i = eb_i.transform(mc_tr_i[mc_base]), eb_i.transform(mc_cal_i[mc_base]), eb_i.transform(mc_te_i[mc_base])
        z_tr_i, z_cal_i, z_te_i = ez_i.transform(tr_dec), ez_i.transform(cal_dec), ez_i.transform(te_dec)
        ck_icu = ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.030)
        clf_ag = FluidCollisionBackboneClassifier(tau_star=ck_icu.tau_star, kappa_max=20.0, min_cz_ess=240.0, seed=51).fit(
            np.hstack([xb_tr_i, z_tr_i]), z_tr_i, mc_tr_i["y_critical_icu_death"].to_numpy(), logit(tr_mc_icu["p1_oof_raw"])
        )
        r_tr_ag = clf_ag.decision_function(np.hstack([xb_tr_i, z_tr_i]), z_tr_i, logit(tr_mc_icu["p1_oof_raw"]))
        r_c_ag = clf_ag.decision_function(np.hstack([xb_cal_i, z_cal_i]), z_cal_i, logit(tr_mc_icu["p1_cal_raw"]))
        r_t_ag = clf_ag.decision_function(np.hstack([xb_te_i, z_te_i]), z_te_i, logit(tr_mc_icu["p1_te"]))
        pl_ag = fit_platt_calibrator(r_c_ag, mc_cal_i["y_critical_icu_death"].to_numpy())
        p2_tr_ag = np.clip(apply_platt(r_tr_ag, pl_ag), 1e-6, 1 - 1e-6)
        p2_c_ag = np.clip(apply_platt(r_c_ag, pl_ag), 1e-6, 1 - 1e-6)
        p2_t_ag = np.clip(apply_platt(r_t_ag, pl_ag), 1e-6, 1 - 1e-6)
        voi_ag = fit_predict_prepurchase_voi(
            xb_tr_i, tr_mc_icu["p1_oof_eval"], p2_tr_ag, xb_cal_i, tr_mc_icu["p1_cal_eval"], p2_c_ag, xb_te_i, tr_mc_icu["p1_te"], ck_icu, seed=44
        )
        q_ag = voi_ag > ck_icu.per_instance_query_cost
        p_rt_ag = np.where(q_ag, p2_t_ag, tr_mc_icu["p1_te"])
        nb_ag = weighted_net_benefit_per_1k(tr_mc_icu["y_te"], p_rt_ag, q_ag, ck_icu)
        boot_ag = patient_cluster_bootstrap_diff(
            tr_mc_icu["y_te"], p_rt_ag, q_ag, tr_mc_icu["p1_te"], np.zeros(len(q_ag), dtype=bool), tr_mc_icu["clusters_te"], ck_icu, n_boot=10_000, seed=88
        )
        agniel_audit = {
            "task": "Stanford MC-MED Critical ICU/Death (Agniel-Deconfounded, Zero Missingness Indicators)",
            "selected_channels": sel_icu,
            "nb_agniel_routed": round(nb_ag, 4),
            "auprc_full_agniel": round(float(average_precision_score(tr_mc_icu["y_te"], p2_t_ag)), 4),
            "query_rate": round(float(np.mean(q_ag)), 4),
            "boot_vs_stage1": boot_ag,
        }

    # 3. Load Audited NTUH True Case-Control Pre-Arrest SCA Cohort (`2015-2019`, `N = 16,771`, `pi_target = 0.02` — `R17`)
    ntuh_cc = pd.read_parquet(DATA_DIR / "ntuh_pre_arrest_case_control_v2.parquet")
    ntuh_base = [
        c for c in [
            "age", "sex_male", "ventricular_rate", "atrial_rate", "pr_interval",
            "qrs_duration", "qt_interval", "qt_corrected", "p_axis", "r_axis", "t_axis", "qtc_frederica",
        ]
        if c in ntuh_cc.columns
    ]
    ntuh_channels = [
        ChannelDef("ch_ant_v1v4", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_ant")), 0.55),
        ChannelDef("ch_lat_i_avl_v5v6", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_lat")), 0.55),
        ChannelDef("ch_inf_ii_iii_avf", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_inf")), 0.55),
        ChannelDef("ch_avr", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_avr")), 0.45),
        ChannelDef("ch_ecg_diag_stmts", tuple(c for c in ntuh_cc.columns if c.startswith("dx_")), 0.40),
        ChannelDef("ch_waveform_12lead_median", tuple(c for c in ntuh_cc.columns if c.startswith("medwf_")), 0.85),
    ]

    # Task 6: NTUH Pre-Arrest SCA (Case-Control Prior-Odds Adjusted to Clinical Prevalence pi_target = 0.02)
    res_ntuh_cc_002, tr_ntuh_002 = evaluate_clinical_task_full(
        "ntuh_pre_arrest_sca_pi002",
        "NTUH Pre-Arrest SCA (2015-2019 Case-Control, pi_target=2.0%)",
        ntuh_cc, "y_pre_arrest", "patient_ngsci_id", ntuh_base, ntuh_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.020),
        budget=1.80,
        dca_taus=list(np.round(np.linspace(0.01, 0.12, 23), 4)),
        pi_target_override=0.02,
        is_prespecified_primary=True,
        seed=47,
    )

    # NTUH Case-Control Target Prevalence Sensitivity (`pi_target in {0.01, 0.05}`)
    res_ntuh_cc_001, _ = evaluate_clinical_task_full(
        "ntuh_pre_arrest_sca_pi001_sensitivity",
        "NTUH Pre-Arrest SCA (Sensitivity pi_target=1.0%)",
        ntuh_cc, "y_pre_arrest", "patient_ngsci_id", ntuh_base, ntuh_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=32.33, target_precision=0.20, per_instance_query_cost=0.020),
        budget=1.80,
        dca_taus=[0.01, 0.02, 0.03, 0.05],
        pi_target_override=0.01,
        is_prespecified_primary=False,
        seed=147,
    )
    res_ntuh_cc_005, _ = evaluate_clinical_task_full(
        "ntuh_pre_arrest_sca_pi005_sensitivity",
        "NTUH Pre-Arrest SCA (Sensitivity pi_target=5.0%)",
        ntuh_cc, "y_pre_arrest", "patient_ngsci_id", ntuh_base, ntuh_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=9.0, target_precision=0.35, per_instance_query_cost=0.020),
        budget=1.80,
        dca_taus=[0.03, 0.05, 0.10, 0.15],
        pi_target_override=0.05,
        is_prespecified_primary=False,
        seed=247,
    )

    # Supplementary NTUH Post-ROSC Cohort (`[1]ROSC` strictly, fixed hyphenated year parser — `R6`)
    ntuh_rosc = pd.read_parquet(DATA_DIR / "ntuh_post_rosc_v2.parquet")
    rosc_base = [
        c for c in [
            "age", "sex_male", "witnessed_collapse", "initial_rhythm",
            "location_of_cardiac_arrest", "cpr_duration", "epinerphine_dose", "dc_shock",
        ]
        if c in ntuh_rosc.columns
    ]
    rosc_comorb_cols = tuple(c for c in ["dm", "htn", "cad", "hf", "arrhythmia", "copd_asthma", "esrd", "cva"] if c in ntuh_rosc.columns)
    rosc_ecg_int_cols = tuple(
        c for c in ["ventricular_rate", "pr_interval", "qrs_duration", "qt_interval", "qt_corrected", "qtc_frederica", "p_axis", "r_axis", "t_axis"]
        if c in ntuh_rosc.columns
    )
    rosc_channels = [
        ChannelDef("ch_rosc_comorbidities", rosc_comorb_cols, 0.50),
        ChannelDef("ch_rosc_ecg_intervals", rosc_ecg_int_cols, 0.50),
        ChannelDef("ch_rosc_waveform_12lead", tuple(c for c in ntuh_rosc.columns if c.startswith("medwf_")), 0.70),
    ]
    res_rosc_surv, _ = evaluate_clinical_task_full(
        "ntuh_post_rosc_survival_supplement",
        "NTUH Post-ROSC Hospital Survival (`[1]ROSC` Fixed Split, Supplement)",
        ntuh_rosc, "y_survive", "patient_ngsci_id", rosc_base, rosc_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=4.0, target_precision=0.40, per_instance_query_cost=0.020),
        budget=1.20,
        dca_taus=[0.10, 0.15, 0.20, 0.25, 0.30],
        is_prespecified_primary=False,
        seed=48,
    )
    res_rosc_cpc, _ = evaluate_clinical_task_full(
        "ntuh_post_rosc_cpc12_supplement",
        "NTUH Post-ROSC Good Neuro CPC 1-2 (`[1]ROSC` Fixed Split, Supplement)",
        ntuh_rosc, "y_good_neuro_cpc12", "patient_ngsci_id", rosc_base, rosc_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=5.67, target_precision=0.35, per_instance_query_cost=0.020),
        budget=1.20,
        dca_taus=[0.08, 0.12, 0.15, 0.20, 0.25],
        is_prespecified_primary=False,
        seed=49,
    )

    # --- Holm Multiplicity Adjustment Across the 6 Primary Clinical Tasks (`R14`) ---
    primary_tasks = [
        res_mc_icu,
        res_mc_adm,
        res_mc_b72,
        res_bwh_death30,
        res_bwh_stent10,
        res_ntuh_cc_002,
    ]
    raw_p_vs_s1 = [t["bootstraps_10k_patient_cluster"]["upca_vs_stage1"]["p_two_sided"] for t in primary_tasks]
    holm_p_vs_s1 = holm_adjust(raw_p_vs_s1)
    raw_p_vs_glob = [t["bootstraps_10k_patient_cluster"]["upca_vs_global_cmi_fluid"]["p_two_sided"] for t in primary_tasks]
    holm_p_vs_glob = holm_adjust(raw_p_vs_glob)
    raw_p_vs_wcz = [t["bootstraps_10k_patient_cluster"]["upca_vs_weighted_pilot_cz_fluid"]["p_two_sided"] for t in primary_tasks]
    holm_p_vs_wcz = holm_adjust(raw_p_vs_wcz)
    for idx_t, t_obj in enumerate(primary_tasks):
        t_obj["holm_adjusted_p_ across_6_primary"] = {
            "upca_vs_stage1": round(float(holm_p_vs_s1[idx_t]), 6),
            "upca_vs_global_cmi_fluid": round(float(holm_p_vs_glob[idx_t]), 6),
            "upca_vs_weighted_pilot_cz_fluid": round(float(holm_p_vs_wcz[idx_t]), 6),
        }

    # --- Exploratory Subgroup Performance & Missingness Audits (`R11`) ---
    print("\n=== R11: Subgroup Performance, Calibration, FNR/FPR & Cochran-Q Heterogeneity Audits ===")
    subgroups = {
        "bwh_death30_by_agi_quartile": run_subgroup_and_missingness_audit(
            te_bwh, "agi_quartile",
            ["Q1_Lowest_Income", "Q2_LowMid", "Q3_HighMid", "Q4_Highest_Income"],
            tr_bwh_death30, c_bwh_death, seed=501,
        ),
        "bwh_death30_by_race": run_subgroup_and_missingness_audit(
            te_bwh, "race_str",
            ["White", "Black", "Hispanic", "Other"],
            tr_bwh_death30, c_bwh_death, seed=502,
        ),
        "mcmed_icu_by_payor": run_subgroup_and_missingness_audit(
            mc_te_i, "payor_clean",
            ["Medicaid", "Medicare", "Unspecified_or_Commercial"],
            tr_mc_icu, ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.030), seed=503,
        ),
        "mcmed_icu_by_race": run_subgroup_and_missingness_audit(
            mc_te_i, "race_clean",
            ["White", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander", "Other_or_Unknown"],
            tr_mc_icu, ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.030), seed=504,
        ),
        "mcmed_admission_by_payor": run_subgroup_and_missingness_audit(
            mc_te_i, "payor_clean",
            ["Medicaid", "Medicare", "Unspecified_or_Commercial"],
            tr_mc_adm, ClinicalContract(cost_fp=1.0, cost_fn=5.67, target_precision=0.50, per_instance_query_cost=0.025), seed=505,
        ),
        "mcmed_admission_by_race": run_subgroup_and_missingness_audit(
            mc_te_i, "race_clean",
            ["White", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander", "Other_or_Unknown"],
            tr_mc_adm, ClinicalContract(cost_fp=1.0, cost_fn=5.67, target_precision=0.50, per_instance_query_cost=0.025), seed=506,
        ),
    }

    # --- R15 Efficiency & R10 Sensitivity ---
    eff_r15 = run_two_phase_efficiency_comparison(mc, mc_base, mc_channels, ClinicalContract(1.0, 19.0, 0.25, 0.030))
    sens_r10 = run_hyperparameter_and_cost_sensitivity(mc, mc_base, mc_channels, bwh, bwh_base, bwh_ecg_channels)

    return {
        "prespecified_primary_regime_endpoint": "mcmed_critical_icu_death",
        "primary_tasks": {
            "mcmed_critical_icu_death": res_mc_icu,
            "mcmed_hospital_admission": res_mc_adm,
            "mcmed_bounce72_admit_2h": res_mc_b72,
            "bwh_death_030_ecg_only": res_bwh_death30,
            "bwh_stent_cabg_010_ecg_only": res_bwh_stent10,
            "ntuh_pre_arrest_sca_pi002": res_ntuh_cc_002,
        },
        "secondary_and_sensitivity_tasks": {
            "bwh_mace_death_030_ecg_only_secondary": res_bwh_mace30_ecg,
            "bwh_slice_audits_untested_and_no_troponin": bwh_slice_audits,
            "bwh_complete_troponin_death_030": res_bwh_trop_death30,
            "bwh_complete_troponin_stent_cabg_010": res_bwh_trop_stent10,
            "mcmed_agniel_deconfounding_icu": agniel_audit,
            "mcmed_bounce72_complete_cdw_sensitivity": res_mc_b72_cdw_comp,
            "mcmed_bounce72_arrival_0h_sensitivity": res_mc_b72_arr0h,
            "mcmed_bounce72_discharge_landmark_sensitivity": res_mc_b72_disch_lm,
            "ntuh_pre_arrest_sca_pi001_sensitivity": res_ntuh_cc_001,
            "ntuh_pre_arrest_sca_pi005_sensitivity": res_ntuh_cc_005,
            "ntuh_post_rosc_survival_supplement": res_rosc_surv,
            "ntuh_post_rosc_cpc12_supplement": res_rosc_cpc,
        },
        "subgroup_and_missingness_audits": subgroups,
        "two_phase_efficiency_r15": eff_r15,
        "sensitivity_sweeps_r10": sens_r10,
        "clinical_phase_elapsed_sec": round(time.time() - t0, 1),
    }


# ===========================================================================
# PART B: 1,000-REP EXPERIMENT E5 & 4D OPERATING-ENVELOPE GRID (MAX 4 CORES!)
# ===========================================================================


def _init_single_thread_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["UPCA_LGBM_JOBS"] = "1"
    # Enforce single-thread LightGBM inside child worker process without altering sklearn __init__ signature
    _orig_proc = lgb.LGBMModel._process_params

    def _patched_proc(self: Any, stage: str = "fit") -> dict[str, Any]:
        params = _orig_proc(self, stage=stage)
        params["n_jobs"] = 1
        params["num_threads"] = 1
        return params

    lgb.LGBMModel._process_params = _patched_proc  # type: ignore[method-assign]

    _orig_booster_init = lgb.Booster.__init__

    def _patched_booster_init(self: Any, params: dict[str, Any] | None = None, *args: Any, **kwargs: Any) -> None:
        if isinstance(params, dict):
            params = dict(params)
            params["n_jobs"] = 1
            params["num_threads"] = 1
        _orig_booster_init(self, params, *args, **kwargs)

    lgb.Booster.__init__ = _patched_booster_init  # type: ignore[method-assign]


def _e5_single_rep(rep: int) -> dict[str, Any]:
    """Single replicate of Experiment E5 running strictly on 1 CPU thread (`n_jobs=1`)."""
    contract = ClinicalContract(cost_fp=1.0, cost_fn=9.0, target_precision=0.35, per_instance_query_cost=0.09)
    base_cols = ["x_risk", "u_cue"]
    channels = [
        ChannelDef("ch_identifiable", ("z_ident",), 0.80),
        ChannelDef("ch_unidentifiable", ("z_unident",), 0.80),
        ChannelDef("ch_null_1", ("z_null1",), 0.50),
        ChannelDef("ch_null_2", ("z_null2",), 0.50),
    ]
    ch_map = {c.channel_id: c for c in channels}

    rng = np.random.default_rng(1000 + rep)
    n = 9000
    in_cz = rng.binomial(1, 0.40, size=n)
    u_cue = rng.binomial(1, 0.22, size=n)
    u_hidden = rng.binomial(1, 0.22, size=n)
    base_logit = np.where(
        in_cz == 1,
        logit(0.090) + rng.normal(0, 0.10, size=n),
        logit(0.010) + rng.normal(0, 0.20, size=n),
    )
    p0 = expit(base_logit)
    sig_id = 3.2 * u_cue * in_cz
    sig_unid = 3.2 * u_hidden * in_cz
    sig_tot = np.sqrt(sig_id ** 2 + sig_unid ** 2)
    mu_star = solve_martingale_offset(p0, sig_tot)
    z_ident_sig = sig_id * rng.normal(0, 1.0, size=n)
    z_unident_sig = sig_unid * rng.normal(0, 1.0, size=n)
    true_p = expit(base_logit + mu_star + z_ident_sig + z_unident_sig)
    y = rng.binomial(1, true_p)

    df = pd.DataFrame({
        "x_risk": (base_logit + rng.normal(0, 0.12, size=n)).astype(np.float32),
        "u_cue": u_cue.astype(np.float32),
        "z_ident": (z_ident_sig + rng.normal(0, 0.15, size=n)).astype(np.float32),
        "z_unident": (z_unident_sig + rng.normal(0, 0.15, size=n)).astype(np.float32),
        "z_null1": rng.normal(0, 1, size=n).astype(np.float32),
        "z_null2": rng.normal(0, 1, size=n).astype(np.float32),
    })
    tr, cal, te = df.iloc[:5400].reset_index(drop=True), df.iloc[5400:7200].reset_index(drop=True), df.iloc[7200:].reset_index(drop=True)
    y_tr, y_cal, y_te = y[:5400], y[5400:7200], y[7200:]

    # Fast single-thread OOF Stage 1 & test Stage 1
    p1_oof = compute_oof_stage1(tr[base_cols], y_tr, n_folds=3, seed=rep)
    p1_cal, p1_tests = fit_stage1_predictions(tr, y_tr, cal, y_cal, [te], base_cols, seed=rep)
    p1_te = p1_tests[0]

    disc_a, _ = discover_stage_a_ablation(
        tr, y_tr, base_cols, channels, p1_oof, contract, budget=2.0,
        mode="offset_cz_schervish", pilot_fraction=0.35, seed=rep,
    )
    upca_sel, p_log = oof_policy_screening_stage_b(disc_a.selected_ids, tr, y_tr, p1_oof, base_cols, ch_map, contract, seed=rep)

    cz_b1 = disc_a.selected_ids[:1]
    solo_scores = {e["tested_drop"]: e["solo_routed_gain"] / ch_map[e["tested_drop"]].cost for e in p_log}
    upca_b1 = sorted(upca_sel, key=lambda c: -solo_scores.get(c, 0.0))[:1]

    xb_tr = tr[base_cols].to_numpy(dtype=np.float32)
    xb_cal = cal[base_cols].to_numpy(dtype=np.float32)
    xb_te = te[base_cols].to_numpy(dtype=np.float32)
    p1_oof_c = np.clip(p1_oof, 1e-5, 1.0 - 1e-5)

    eval_cache: dict[tuple[str, ...], float] = {}

    def _eval_sel(sel_ids: list[str]) -> float:
        key = tuple(sel_ids)
        if key in eval_cache:
            return eval_cache[key]
        if not sel_ids:
            val = weighted_net_benefit_per_1k(y_te, p1_te, np.zeros(len(y_te), dtype=bool), contract)
            eval_cache[key] = val
            return val
        s2_cols = [c for cid in sel_ids for c in ch_map[cid].columns]
        z_tr = tr[s2_cols].to_numpy(dtype=np.float32)
        z_cal = cal[s2_cols].to_numpy(dtype=np.float32)
        z_te = te[s2_cols].to_numpy(dtype=np.float32)
        clf = FluidCollisionBackboneClassifier(tau_star=contract.tau_star, seed=rep).fit(
            np.hstack([xb_tr, z_tr]), z_tr, y_tr, logit(p1_oof_c)
        )
        raw_tr = clf.decision_function(np.hstack([xb_tr, z_tr]), z_tr, logit(p1_oof_c))
        raw_cal = clf.decision_function(np.hstack([xb_cal, z_cal]), z_cal, logit(p1_cal))
        raw_te = clf.decision_function(np.hstack([xb_te, z_te]), z_te, logit(p1_te))
        platt = fit_platt_calibrator(raw_cal, y_cal)
        p2_tr = np.clip(apply_platt(raw_tr, platt), 1e-6, 1.0 - 1e-6)
        p2_cal = np.clip(apply_platt(raw_cal, platt), 1e-6, 1.0 - 1e-6)
        p2_te = np.clip(apply_platt(raw_te, platt), 1e-6, 1.0 - 1e-6)
        voi_te = fit_predict_prepurchase_voi(
            xb_tr, p1_oof_c, p2_tr, xb_cal, p1_cal, p2_cal, xb_te, p1_te, contract, use_dispersion_2d=True, seed=rep
        )
        q_te = voi_te > contract.per_instance_query_cost
        p_rt = np.where(q_te, p2_te, p1_te)
        val = weighted_net_benefit_per_1k(y_te, p_rt, q_te, contract)
        eval_cache[key] = val
        return val

    nb_s1 = _eval_sel([])
    nb_a_b2 = _eval_sel(disc_a.selected_ids)
    nb_u_b2 = _eval_sel(upca_sel)
    nb_a_b1 = _eval_sel(cz_b1)
    nb_u_b1 = _eval_sel(upca_b1)

    return {
        "a_id": bool("ch_identifiable" in disc_a.selected_ids),
        "a_unid": bool("ch_unidentifiable" in disc_a.selected_ids),
        "a_null": bool(any("null" in c for c in disc_a.selected_ids)),
        "u_id": bool("ch_identifiable" in upca_sel),
        "u_unid": bool("ch_unidentifiable" in upca_sel),
        "u_null": bool(any("null" in c for c in upca_sel)),
        "u_b1_id": bool(upca_b1 == ["ch_identifiable"]),
        "a_b1_id": bool(cz_b1 == ["ch_identifiable"]),
        "nb_s1": nb_s1,
        "nb_a_b2": nb_a_b2,
        "nb_u_b2": nb_u_b2,
        "nb_a_b1": nb_a_b1,
        "nb_u_b1": nb_u_b1,
    }


def _envelope_cell_worker(args: tuple[float, float, float, float, int]) -> dict[str, Any]:
    """Evaluate one cell `(rho, sigma, cq_over_g, pi_cz)` across `n_cell_reps` replicates (`R13`).

    Maps exact empirical operating envelope where `u_hat(X_base)` 2D dispersion stratification
    and OOF policy screening separate `ch_identifiable` from `ch_unidentifiable` vs where it breaks down.
    """
    rho, sigma, cq_ratio, pi_cz, n_reps = args
    # Compute exact theoretical active point-tau* Bayes VOI g(sigma) at p0 = tau* = 0.10 (C_FP=1, C_FN=9)
    c_zero = ClinicalContract(cost_fp=1.0, cost_fn=9.0, target_precision=0.35, per_instance_query_cost=0.0)
    g_val = float(mean_centered_voi(np.array([0.10]), np.array([sigma]), c_zero)[0])
    cq = float(cq_ratio * g_val)
    contract = ClinicalContract(cost_fp=1.0, cost_fn=9.0, target_precision=0.35, per_instance_query_cost=cq)

    # Theoretical population gains (in NB per 1,000 scale: raw gain * 1000 / C_FN)
    theo_nb_id = (1000.0 / 9.0) * pi_cz * rho * max(0.0, g_val - cq)
    theo_nb_unid = (1000.0 / 9.0) * pi_cz * max(0.0, rho * g_val - cq)

    nb_2d_id_minus_s1 = []
    nb_12b_id_minus_s1 = []
    nb_2d_unid_minus_s1 = []
    z_solo_id_list = []
    z_solo_unid_list = []
    screen_prefers_id = []
    screen_retains_id = []
    screen_prunes_unid = []

    seed_base = int(round(100000 * rho + 1000 * sigma + 100 * cq_ratio + 10 * pi_cz))
    for rep in range(n_reps):
        rng = np.random.default_rng(seed_base + rep)
        n = 6000  # 3600 train / 1200 cal / 1200 test
        in_cz = rng.binomial(1, pi_cz, size=n)
        u_cue = rng.binomial(1, rho, size=n)
        u_hid = rng.binomial(1, rho, size=n)
        base_logit = np.where(
            in_cz == 1,
            logit(0.095) + rng.normal(0, 0.08, size=n),
            logit(0.010) + rng.normal(0, 0.20, size=n),
        )
        p0 = expit(base_logit)
        sig_id = sigma * u_cue * in_cz
        sig_un = sigma * u_hid * in_cz
        mu_star = solve_martingale_offset(p0, np.sqrt(sig_id ** 2 + sig_un ** 2))
        z_id = sig_id * rng.normal(0, 1.0, size=n)
        z_un = sig_un * rng.normal(0, 1.0, size=n)
        y = rng.binomial(1, expit(base_logit + mu_star + z_id + z_un))

        xb = np.column_stack([base_logit + rng.normal(0, 0.10, size=n), u_cue.astype(float)]).astype(np.float32)
        zi = (z_id + rng.normal(0, 0.15, size=n)).reshape(-1, 1).astype(np.float32)
        zu = (z_un + rng.normal(0, 0.15, size=n)).reshape(-1, 1).astype(np.float32)

        xb_tr, xb_cal, xb_te = xb[:3600], xb[3600:4800], xb[4800:]
        zi_tr, zi_cal, zi_te = zi[:3600], zi[3600:4800], zi[4800:]
        zu_tr, zu_cal, zu_te = zu[:3600], zu[3600:4800], zu[4800:]
        y_tr, y_cal, y_te = y[:3600], y[3600:4800], y[4800:]

        # Fit Stage 1
        clf1 = lgb.LGBMClassifier(
            n_estimators=80, learning_rate=0.05, num_leaves=15, min_child_samples=25,
            random_state=rep, n_jobs=1, verbose=-1,
        ).fit(xb_tr, y_tr)
        pl1 = fit_platt_calibrator(clf1.predict(xb_cal, raw_score=True), y_cal)
        p1_tr = np.clip(apply_platt(clf1.predict(xb_tr, raw_score=True), pl1), 1e-5, 1 - 1e-5)
        p1_cal = np.clip(apply_platt(clf1.predict(xb_cal, raw_score=True), pl1), 1e-5, 1 - 1e-5)
        p1_te = np.clip(apply_platt(clf1.predict(xb_te, raw_score=True), pl1), 1e-5, 1 - 1e-5)

        u_s1_te = unit_net_benefit_vec(y_te, p1_te, np.zeros(len(y_te)), contract)

        def _fit_channel(z_tr_c: np.ndarray, z_cal_c: np.ndarray, z_te_c: np.ndarray) -> tuple[float, float, float]:
            clf2 = lgb.LGBMClassifier(
                n_estimators=90, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                random_state=rep + 3, n_jobs=1, verbose=-1,
            ).fit(np.hstack([xb_tr, z_tr_c]), y_tr, init_score=logit(p1_tr))
            r_tr = logit(p1_tr) + clf2.predict(np.hstack([xb_tr, z_tr_c]), raw_score=True)
            r_cal = logit(p1_cal) + clf2.predict(np.hstack([xb_cal, z_cal_c]), raw_score=True)
            r_te = logit(p1_te) + clf2.predict(np.hstack([xb_te, z_te_c]), raw_score=True)
            pl2 = fit_platt_calibrator(r_cal, y_cal)
            p2_tr_c = np.clip(apply_platt(r_tr, pl2), 1e-6, 1 - 1e-6)
            p2_cal_c = np.clip(apply_platt(r_cal, pl2), 1e-6, 1 - 1e-6)
            p2_te_c = np.clip(apply_platt(r_te, pl2), 1e-6, 1 - 1e-6)

            # Evaluate 2D dispersion VOI vs 12-bin VOI on test, and OOF screening Z-stat on cal/tr split
            voi_2d = fit_predict_prepurchase_voi(
                xb_tr, p1_tr, p2_tr_c, xb_cal, p1_cal, p2_cal_c, xb_te, p1_te, contract, use_dispersion_2d=True, seed=rep
            )
            q_2d = voi_2d > cq
            u_2d = unit_net_benefit_vec(y_te, np.where(q_2d, p2_te_c, p1_te), q_2d, contract)
            diff_vec = u_2d - u_s1_te
            z_stat = float(np.mean(diff_vec) / max(1e-9, float(np.std(diff_vec, ddof=1) / np.sqrt(len(diff_vec)))))

            voi_12b = fit_predict_prepurchase_voi(
                xb_tr, p1_tr, p2_tr_c, xb_cal, p1_cal, p2_cal_c, xb_te, p1_te, contract, use_dispersion_2d=False, seed=rep
            )
            q_12b = voi_12b > cq
            u_12b = unit_net_benefit_vec(y_te, np.where(q_12b, p2_te_c, p1_te), q_12b, contract)
            return float(np.mean(diff_vec)), float(np.mean(u_12b - u_s1_te)), z_stat

        d2d_id, d12_id, z_id_s = _fit_channel(zi_tr, zi_cal, zi_te)
        d2d_un, _, z_un_s = _fit_channel(zu_tr, zu_cal, zu_te)

        nb_2d_id_minus_s1.append(d2d_id)
        nb_12b_id_minus_s1.append(d12_id)
        nb_2d_unid_minus_s1.append(d2d_un)
        z_solo_id_list.append(z_id_s)
        z_solo_unid_list.append(z_un_s)
        screen_prefers_id.append(d2d_id > d2d_un)
        screen_retains_id.append(z_id_s >= 1.282)
        screen_prunes_unid.append(z_un_s < 1.282)

    regime = (
        "unconditional_profitable"
        if cq_ratio < rho
        else ("selective_separation" if cq_ratio < 1.0 else "never_profitable")
    )
    return {
        "rho": rho,
        "sigma": sigma,
        "cquery_over_g": cq_ratio,
        "pi_cz": pi_cz,
        "g_active_voi": round(g_val, 5),
        "c_query": round(cq, 5),
        "theoretical_regime": regime,
        "theo_nb_gain_identifiable": round(theo_nb_id, 3),
        "theo_nb_gain_unidentifiable": round(theo_nb_unid, 3),
        "emp_nb_gain_2d_identifiable": round(float(np.mean(nb_2d_id_minus_s1)), 3),
        "emp_nb_gain_12bin_identifiable": round(float(np.mean(nb_12b_id_minus_s1)), 3),
        "emp_2d_minus_12bin_gain": round(float(np.mean(np.array(nb_2d_id_minus_s1) - np.array(nb_12b_id_minus_s1))), 3),
        "emp_nb_gain_2d_unidentifiable": round(float(np.mean(nb_2d_unid_minus_s1)), 3),
        "mean_z_identifiable": round(float(np.mean(z_solo_id_list)), 3),
        "mean_z_unidentifiable": round(float(np.mean(z_solo_unid_list)), 3),
        "retain_identifiable_rate": round(float(np.mean(screen_retains_id)), 4),
        "prune_unidentifiable_rate": round(float(np.mean(screen_prunes_unid)), 4),
        "prefer_identifiable_rate": round(float(np.mean(screen_prefers_id)), 4),
        "n_reps": n_reps,
    }


def run_synthetic_phase(n_e5_reps: int = 1000, n_grid_reps: int = 50) -> dict[str, Any]:
    t0 = time.time()
    print("\n" + "=" * 84)
    print(f"STEP 2B: 1,000-REP EXPERIMENT E5 + 4D OPERATING-ENVELOPE GRID ({n_grid_reps} REPS/CELL, 4 WORKERS x 1 THREAD)")
    print("=" * 84, flush=True)

    e5_rows: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=4, initializer=_init_single_thread_worker) as pool:
        for idx, r_out in enumerate(pool.map(_e5_single_rep, range(n_e5_reps), chunksize=5), start=1):
            e5_rows.append(r_out)
            if idx % 100 == 0 or idx == n_e5_reps:
                k_null = sum(1 for x in e5_rows if x["u_null"])
                k_id = sum(1 for x in e5_rows if x["u_id"])
                d_b1 = float(np.mean([x["nb_u_b1"] - x["nb_a_b1"] for x in e5_rows]))
                print(
                    f"  [E5 {idx:4d}/{n_e5_reps}] null_FWER={k_null}/{idx} ({k_null/idx*100:.2f}%), "
                    f"retain_id={k_id}/{idx} ({k_id/idx*100:.2f}%), mean_dNB(B=1)={d_b1:+.3f}",
                    flush=True,
                )

    n_e5 = len(e5_rows)
    k_a_null = sum(1 for x in e5_rows if x["a_null"])
    k_u_null = sum(1 for x in e5_rows if x["u_null"])
    k_a_id = sum(1 for x in e5_rows if x["a_id"])
    k_a_unid = sum(1 for x in e5_rows if x["a_unid"])
    k_u_id = sum(1 for x in e5_rows if x["u_id"])
    k_u_unid = sum(1 for x in e5_rows if x["u_unid"])
    k_u_b1_id = sum(1 for x in e5_rows if x["u_b1_id"])
    k_a_b1_id = sum(1 for x in e5_rows if x["a_b1_id"])

    nb_s1_arr = np.array([x["nb_s1"] for x in e5_rows], dtype=float)
    nb_a_b2_arr = np.array([x["nb_a_b2"] for x in e5_rows], dtype=float)
    nb_u_b2_arr = np.array([x["nb_u_b2"] for x in e5_rows], dtype=float)
    nb_a_b1_arr = np.array([x["nb_a_b1"] for x in e5_rows], dtype=float)
    nb_u_b1_arr = np.array([x["nb_u_b1"] for x in e5_rows], dtype=float)
    diff_b1 = nb_u_b1_arr - nb_a_b1_arr
    diff_b2 = nb_u_b2_arr - nb_a_b2_arr

    se_b1 = float(np.std(diff_b1, ddof=1) / np.sqrt(n_e5))
    se_b2 = float(np.std(diff_b2, ddof=1) / np.sqrt(n_e5))
    from scipy.stats import t as t_dist
    t_b1 = float(np.mean(diff_b1) / max(1e-12, se_b1))
    p_b1 = float(2.0 * t_dist.sf(abs(t_b1), df=n_e5 - 1))

    e5_summary = {
        "n_reps": n_e5,
        "stage_a_null_fwer": round(k_a_null / n_e5, 5),
        "stage_a_null_fwer_count": k_a_null,
        "stage_a_null_fwer_ci95_clopper_pearson": _clopper_pearson_ci95(k_a_null, n_e5),
        "upca_null_fwer": round(k_u_null / n_e5, 5),
        "upca_null_fwer_count": k_u_null,
        "upca_null_fwer_ci95_clopper_pearson": _clopper_pearson_ci95(k_u_null, n_e5),
        "stage_a_pick_identifiable_rate": round(k_a_id / n_e5, 5),
        "stage_a_pick_identifiable_ci95": _clopper_pearson_ci95(k_a_id, n_e5),
        "stage_a_pick_unidentifiable_rate": round(k_a_unid / n_e5, 5),
        "stage_a_pick_unidentifiable_ci95": _clopper_pearson_ci95(k_a_unid, n_e5),
        "upca_retain_identifiable_rate": round(k_u_id / n_e5, 5),
        "upca_retain_identifiable_count": k_u_id,
        "upca_retain_identifiable_ci95_clopper_pearson": _clopper_pearson_ci95(k_u_id, n_e5),
        "upca_retain_unidentifiable_rate": round(k_u_unid / n_e5, 5),
        "upca_retain_unidentifiable_count": k_u_unid,
        "upca_retain_unidentifiable_ci95_clopper_pearson": _clopper_pearson_ci95(k_u_unid, n_e5),
        "budget_b1_top1_identifiable_rate_stage_a": round(k_a_b1_id / n_e5, 5),
        "budget_b1_top1_identifiable_rate_upca": round(k_u_b1_id / n_e5, 5),
        "nb_stage1_mean": round(float(np.mean(nb_s1_arr)), 4),
        "nb_stage_a_b2_mean": round(float(np.mean(nb_a_b2_arr)), 4),
        "nb_upca_b2_mean": round(float(np.mean(nb_u_b2_arr)), 4),
        "paired_delta_nb_b2_mean": round(float(np.mean(diff_b2)), 4),
        "paired_delta_nb_b2_ci95": [round(float(np.mean(diff_b2) - 1.96 * se_b2), 4), round(float(np.mean(diff_b2) + 1.96 * se_b2), 4)],
        "nb_stage_a_b1_mean": round(float(np.mean(nb_a_b1_arr)), 4),
        "nb_upca_b1_mean": round(float(np.mean(nb_u_b1_arr)), 4),
        "paired_delta_nb_b1_mean": round(float(np.mean(diff_b1)), 4),
        "paired_delta_nb_b1_se": round(se_b1, 4),
        "paired_delta_nb_b1_ci95": [round(float(np.mean(diff_b1) - 1.96 * se_b1), 4), round(float(np.mean(diff_b1) + 1.96 * se_b1), 4)],
        "paired_delta_nb_b1_t_stat": round(t_b1, 3),
        "paired_delta_nb_b1_p_two_sided": p_b1,
    }
    print("  1,000-Rep E5 Complete:", json.dumps(e5_summary, indent=2), flush=True)

    # 4D Operating-Envelope Grid (`3 x 3 x 4 x 2 = 72` cells x `50` reps = `3,600` evaluations)
    grid_tasks = [
        (rho, sig, cq_r, pi_c, n_grid_reps)
        for rho in [0.10, 0.22, 0.40]
        for sig in [1.5, 2.5, 3.5]
        for cq_r in [0.15, 0.30, 0.60, 0.90]
        for pi_c in [0.20, 0.40]
    ]
    grid_cells: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=4, initializer=_init_single_thread_worker) as pool:
        for idx_c, cell_res in enumerate(pool.map(_envelope_cell_worker, grid_tasks), start=1):
            grid_cells.append(cell_res)
            if idx_c % 12 == 0 or idx_c == len(grid_tasks):
                print(
                    f"  [4D Grid {idx_c:2d}/{len(grid_tasks)}] last=({cell_res['rho']}, {cell_res['sigma']}, "
                    f"{cell_res['cquery_over_g']}, {cell_res['pi_cz']}) -> dNB_2D_id={cell_res['emp_nb_gain_2d_identifiable']:+.2f}, "
                    f"retain_id={cell_res['retain_identifiable_rate']*100:.0f}%, prune_unid={cell_res['prune_unidentifiable_rate']*100:.0f}%",
                    flush=True,
                )

    return {
        "experiment_e5_1000_reps": e5_summary,
        "operating_envelope_4d_grid": {
            "n_cells": len(grid_cells),
            "n_reps_per_cell": n_grid_reps,
            "axes": {
                "rho": [0.10, 0.22, 0.40],
                "sigma": [1.5, 2.5, 3.5],
                "cquery_over_g": [0.15, 0.30, 0.60, 0.90],
                "pi_cz": [0.20, 0.40],
            },
            "cells": grid_cells,
        },
        "synthetic_phase_elapsed_sec": round(time.time() - t0, 1),
    }


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    phase = "all"
    if "--phase" in sys.argv:
        phase = sys.argv[sys.argv.index("--phase") + 1]

    clin_json = RES_DIR / "checkpoint_clinical_v2.json"
    synth_json = RES_DIR / "checkpoint_synthetic_v2.json"
    final_json = RES_DIR / "paper3_v2_remediation_results.json"

    if phase in ("all", "clinical"):
        clin_res = run_clinical_phase()
        clin_json.write_text(json.dumps(clin_res, indent=2))
        print("SAVED CLINICAL CHECKPOINT ->", clin_json, flush=True)
    else:
        clin_res = json.loads(clin_json.read_text()) if clin_json.exists() else {}

    if phase in ("all", "synthetic"):
        synth_res = run_synthetic_phase(n_e5_reps=1000, n_grid_reps=50)
        synth_json.write_text(json.dumps(synth_res, indent=2))
        print("SAVED SYNTHETIC CHECKPOINT ->", synth_json, flush=True)
    else:
        synth_res = json.loads(synth_json.read_text()) if synth_json.exists() else {}

    merged = {
        "schema_version": "paper3_v2_complete_remediation",
        "cohort_audit_path": str(DATA_DIR / "cohort_provenance_and_flow_audit.json"),
        "clinical": clin_res,
        "synthetic": synth_res,
    }
    final_json.write_text(json.dumps(merged, indent=2))
    print("\nALL PAPER 3 V2 REMEDIATION RESULTS SAVED ->", final_json, flush=True)


if __name__ == "__main__":
    main()

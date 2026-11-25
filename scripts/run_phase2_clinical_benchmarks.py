"""Phase 2 Multi-Hospital Clinical Acquisition Benchmarks (Exact Published Comparator Outcomes & Splits):
  1. BWH ED-ECG (Mullainathan & Obermeyer / Schubert et al.: 30d MACE/Death All+Untested & 10d Stent/CABG)
  2. Stanford MC-MED (Kansal et al.: Official split_chrono Critical ICU/Death, Admission, 72h Bounceback Admit)
  3. NTUH Arrest-ECG (2015-2019 chronological Impending Sudden Cardiac Arrest)

Compares:
  - Stage-1 Zero-Cost Base
  - Global-CMI + Global S2
  - CZ-Fluid (Schervish + HSFP Neyman + KL-Gated Residual Backbone)
  - Track 1 Policy-Aware Channel Admission Delta_m(S) + Fluid Backbone
  - Selective-Label (Tested-Only) Comparator on BWH ED-ECG
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, "/Users/o2satz/antigravity/collision_zone_ed/src")

from collision_zone.three_predicates import (
    ChannelDef,
    ClinicalContract,
    FastFrameEncoder,
    FluidCollisionBackboneClassifier,
    apply_platt,
    compute_oof_stage1,
    empirical_martingale_voi,
    fit_platt_calibrator,
    fit_stage1_predictions,
    net_benefit_per_1k,
    paired_bootstrap_nb_diff,
    run_sequential_discovery,
)
from decision_aware_acq.policy_admission import (
    compute_murphy_diagram,
    compute_policy_delta_stats,
    per_record_net_utility,
    selective_label_audit,
    skipped_case_safety_certificate,
)

PROC_DIR = Path("data/processed_clinical_cohorts")
OUT_JSON = REPO_ROOT / "results" / "phase2_clinical_results.json"


def _fit_global_gbdt_cal(
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
    p_cal = np.clip(apply_platt(raw_cal, platt), 1e-6, 1.0 - 1e-6)
    p_te = np.clip(apply_platt(raw_te, platt), 1e-6, 1.0 - 1e-6)
    return p_cal, p_te


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


def _policy_aware_channel_admission(
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    p1_tr_oof: np.ndarray,
    base_cols: list[str],
    channels: list[ChannelDef],
    contract: ClinicalContract,
    budget: float,
    seed: int = 42,
) -> dict[str, Any]:
    """Track 1 Policy-Aware Channel Admission strictly inside D_train (leaves D_cal 100% untouched!)."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    idx_fit, idx_pol = next(sss.split(df_tr, y_tr))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    i_vcal, i_peval = next(sss2.split(df_tr.iloc[idx_pol], y_tr[idx_pol]))
    idx_vcal = idx_pol[i_vcal]
    idx_peval = idx_pol[i_peval]

    y_fit, y_vcal, y_peval = y_tr[idx_fit], y_tr[idx_vcal], y_tr[idx_peval]
    p1_vcal = np.clip(p1_tr_oof[idx_vcal], 1e-5, 1.0 - 1e-5)
    p1_peval = np.clip(p1_tr_oof[idx_peval], 1e-5, 1.0 - 1e-5)

    enc_b = FastFrameEncoder().fit(df_tr[base_cols].iloc[idx_fit])
    xb_fit = enc_b.transform(df_tr[base_cols].iloc[idx_fit])
    xb_vcal = enc_b.transform(df_tr[base_cols].iloc[idx_vcal])
    xb_peval = enc_b.transform(df_tr[base_cols].iloc[idx_peval])

    ch_mats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ch in channels:
        enc_c = FastFrameEncoder().fit(df_tr[ch.columns].iloc[idx_fit])
        ch_mats[ch.channel_id] = (
            enc_c.transform(df_tr[ch.columns].iloc[idx_fit]),
            enc_c.transform(df_tr[ch.columns].iloc[idx_vcal]),
            enc_c.transform(df_tr[ch.columns].iloc[idx_peval]),
        )

    selected: list[str] = []
    spent = 0.0
    history: list[dict[str, Any]] = []
    curr_nb_i = per_record_net_utility(
        y_peval, p1_peval, np.zeros(len(y_peval)), contract.tau_star, contract.cost_fp, contract.cost_fn, 0.0
    )
    ch_by_id = {ch.channel_id: ch for ch in channels}

    for step in range(len(channels)):
        affordable = [ch.channel_id for ch in channels if ch.channel_id not in selected and spent + ch.cost <= budget + 1e-9]
        if not affordable:
            break
        step_cands = []
        for cid in affordable:
            parts_fit = [xb_fit] + [ch_mats[s][0] for s in selected] + [ch_mats[cid][0]]
            parts_vcal = [xb_vcal] + [ch_mats[s][1] for s in selected] + [ch_mats[cid][1]]
            parts_peval = [xb_peval] + [ch_mats[s][2] for s in selected] + [ch_mats[cid][2]]
            p2_vcal, p2_peval = _fit_global_gbdt_cal(
                np.hstack(parts_fit), y_fit, np.hstack(parts_vcal), y_vcal, np.hstack(parts_peval), seed=seed + 100 + step
            )
            voi_peval = _compute_emp_voi(p1_vcal, p2_vcal, p1_peval, contract)
            q_mask = voi_peval > contract.per_instance_query_cost
            p_route = np.where(q_mask, p2_peval, p1_peval)
            nb_cand_i = per_record_net_utility(
                y_peval, p_route, q_mask.astype(float), contract.tau_star, contract.cost_fp, contract.cost_fn, contract.per_instance_query_cost
            )
            delta, se, z, pval = compute_policy_delta_stats(nb_cand_i, curr_nb_i)
            step_cands.append(
                {
                    "channel": cid,
                    "delta_nb": delta,
                    "se": se,
                    "z": z,
                    "p_value": pval,
                    "query_rate": float(q_mask.mean()),
                    "cost": ch_by_id[cid].cost,
                    "gain_per_cost": delta / max(1e-6, ch_by_id[cid].cost),
                    "nb_i": nb_cand_i,
                }
            )
        order = sorted(range(len(step_cands)), key=lambda idx: step_cands[idx]["p_value"])
        m_cnt = len(step_cands)
        passing = []
        for rank, idx in enumerate(order):
            thresh = 0.05 / (m_cnt - rank)
            if step_cands[idx]["p_value"] < thresh and step_cands[idx]["delta_nb"] > 0.0:
                passing.append(step_cands[idx])
            else:
                break
        if not passing:
            history.append({"step": step, "action": "stop", "best_delta": round(max(c["delta_nb"] for c in step_cands), 4)})
            break
        winner = max(passing, key=lambda c: c["gain_per_cost"])
        selected.append(winner["channel"])
        spent += winner["cost"]
        curr_nb_i = winner["nb_i"]
        history.append(
            {
                "step": step,
                "selected": winner["channel"],
                "delta_nb": round(winner["delta_nb"], 4),
                "se": round(winner["se"], 4),
                "p_value": round(winner["p_value"], 5),
                "query_rate": round(winner["query_rate"], 4),
                "cost": winner["cost"],
            }
        )

    return {"selected_channels": selected, "spent_budget": round(spent, 3), "history": history}


def _evaluate_arm(
    arm_name: str,
    selected_ids: list[str],
    use_fluid: bool,
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
    seed: int = 42,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    s2_cols = [c for cid in selected_ids for c in ch_by_id[cid].columns]
    if len(s2_cols) == 0:
        p2_cal, p2_te, beta_cz = p1_cal.copy(), p1_te.copy(), 0.0
    else:
        enc_b = FastFrameEncoder().fit(df_tr[base_cols])
        enc_z = FastFrameEncoder().fit(df_tr[s2_cols])
        xb_tr, xb_cal, xb_te = enc_b.transform(df_tr[base_cols]), enc_b.transform(df_cal[base_cols]), enc_b.transform(df_te[base_cols])
        z_tr, z_cal, z_te = enc_z.transform(df_tr[s2_cols]), enc_z.transform(df_cal[s2_cols]), enc_z.transform(df_te[s2_cols])
        x2_tr, x2_cal, x2_te = np.hstack([xb_tr, z_tr]), np.hstack([xb_cal, z_cal]), np.hstack([xb_te, z_te])
        if use_fluid:
            ell1_tr = logit(np.clip(p1_tr_oof, 1e-5, 1.0 - 1e-5))
            clf = FluidCollisionBackboneClassifier(
                tau_star=contract.tau_star, kappa_max=20.0, min_cz_ess=240.0, seed=seed + 7
            ).fit(x2_tr, z_tr, y_tr, ell1_tr)
            raw_cal = clf.decision_function(x2_cal, z_cal, logit(p1_cal))
            raw_te = clf.decision_function(x2_te, z_te, logit(p1_te))
            platt = fit_platt_calibrator(raw_cal, y_cal)
            p2_cal = np.clip(apply_platt(raw_cal, platt), 1e-6, 1.0 - 1e-6)
            p2_te = np.clip(apply_platt(raw_te, platt), 1e-6, 1.0 - 1e-6)
            beta_cz = float(clf.beta_cz_)
        else:
            p2_cal, p2_te = _fit_global_gbdt_cal(x2_tr, y_tr, x2_cal, y_cal, x2_te, seed=seed + 7)
            beta_cz = 0.0

    voi_te = _compute_emp_voi(p1_cal, p2_cal, p1_te, contract)
    q_auto = voi_te > contract.per_instance_query_cost
    p_route = np.where(q_auto, p2_te, p1_te)

    nb_base = float(net_benefit_per_1k(y_te, p1_te, np.zeros(len(y_te), dtype=bool), contract))
    nb_auto = float(net_benefit_per_1k(y_te, p_route, q_auto, contract))
    nb_100 = float(net_benefit_per_1k(y_te, p2_te, np.ones(len(y_te), dtype=bool), contract))
    cert = skipped_case_safety_certificate(
        y_te, p1_te, p2_te, q_auto, contract.tau_star, contract.cost_fp, contract.cost_fn
    )
    res = {
        "arm": arm_name,
        "selected_channels": selected_ids,
        "beta_cz": round(beta_cz, 4),
        "nb_stage1_0pct": round(nb_base, 4),
        "nb_voi_auto": round(nb_auto, 4),
        "nb_100pct_query": round(nb_100, 4),
        "auto_query_rate": round(float(q_auto.mean()), 4),
        "auroc_s2_full": round(float(roc_auc_score(y_te, p2_te)), 4),
        "auprc_s2_full": round(float(average_precision_score(y_te, p2_te)), 4),
        "auroc_routed": round(float(roc_auc_score(y_te, p_route)), 4),
        "auprc_routed": round(float(average_precision_score(y_te, p_route)), 4),
        "skipped_case_certificate": cert,
    }
    return res, p_route, q_auto, p2_te


def run_cohort_suite(
    name: str,
    df: pd.DataFrame,
    y_col: str,
    base_cols: list[str],
    channels: list[ChannelDef],
    contract: ClinicalContract,
    budget: float,
    tested_col: str | None = None,
    fairness_cols: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    t0 = time.time()
    print(f"\n{'='*78}\nCOHORT: {name} | target={y_col} | tau*={contract.tau_star:.3f} | B={budget}\n{'='*78}")

    rng = np.random.default_rng(seed)
    df = df.copy()
    df["null_noise_1a"] = rng.normal(0, 1, size=len(df))
    df["null_noise_1b"] = rng.normal(0, 1, size=len(df))
    df["null_noise_2a"] = rng.normal(0, 1, size=len(df))
    df["null_noise_2b"] = rng.normal(0, 1, size=len(df))
    ch_all = list(channels) + [
        ChannelDef("ch_null_noise_1", ["null_noise_1a", "null_noise_1b"], 0.50),
        ChannelDef("ch_null_noise_2", ["null_noise_2a", "null_noise_2b"], 0.50),
    ]
    ch_by_id = {ch.channel_id: ch for ch in ch_all}

    tr = df[df["split"] == "train"].reset_index(drop=True)
    cal = df[df["split"] == "cal"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    y_tr = tr[y_col].astype(int).to_numpy()
    y_cal = cal[y_col].astype(int).to_numpy()
    y_te = te[y_col].astype(int).to_numpy()
    print(f"  Splits: train={len(tr):,} ({y_tr.mean()*100:.2f}%) | cal={len(cal):,} ({y_cal.mean()*100:.2f}%) | test={len(te):,} ({y_te.mean()*100:.2f}%)")

    p1_tr_oof = compute_oof_stage1(tr[base_cols], y_tr, n_folds=3, seed=seed)
    p1_cal, p1_tests = fit_stage1_predictions(tr, y_tr, cal, y_cal, [te], base_cols, seed=seed)
    p1_te = p1_tests[0]
    nb_s1 = float(net_benefit_per_1k(y_te, p1_te, np.zeros(len(y_te), dtype=bool), contract))
    print(f"  Stage-1 Base: Test AUROC={roc_auc_score(y_te, p1_te):.4f} | AUPRC={average_precision_score(y_te, p1_te):.4f} | NB={nb_s1:+.2f}")

    disc_glob = run_sequential_discovery(
        tr, y_tr, base_cols, ch_all, p1_tr_oof, contract,
        budget=budget, pilot_fraction=0.20, method="global_cmi", seed=seed
    )
    disc_cz = run_sequential_discovery(
        tr, y_tr, base_cols, ch_all, p1_tr_oof, contract,
        budget=budget, pilot_fraction=0.20, method="collision_zone", seed=seed
    )
    disc_pol = _policy_aware_channel_admission(
        tr, y_tr, p1_tr_oof, base_cols, ch_all, contract, budget=budget, seed=seed
    )
    print(f"  Global-CMI selected   : {disc_glob.selected_ids}")
    print(f"  CZ-Schervish selected : {disc_cz.selected_ids}")
    print(f"  Policy-Aware selected : {disc_pol['selected_channels']}")

    res_glob, p_route_glob, q_glob, _ = _evaluate_arm(
        "Global-CMI + Global S2", disc_glob.selected_ids, False,
        tr, y_tr, p1_tr_oof, cal, y_cal, p1_cal, te, y_te, p1_te, base_cols, ch_by_id, contract, seed=seed
    )
    res_cz, p_route_cz, q_cz, _ = _evaluate_arm(
        "CZ-Fluid (Schervish + KL Backbone)", disc_cz.selected_ids, True,
        tr, y_tr, p1_tr_oof, cal, y_cal, p1_cal, te, y_te, p1_te, base_cols, ch_by_id, contract, seed=seed
    )
    res_pol, p_route_pol, q_pol, _ = _evaluate_arm(
        "Policy-Aware Delta_m + Fluid Backbone", disc_pol["selected_channels"], True,
        tr, y_tr, p1_tr_oof, cal, y_cal, p1_cal, te, y_te, p1_te, base_cols, ch_by_id, contract, seed=seed
    )

    for r in (res_glob, res_cz, res_pol):
        print(
            f"  [{r['arm']:38s}] Auto NB={r['nb_voi_auto']:+7.2f} (q={r['auto_query_rate']*100:4.1f}%) | "
            f"100% NB={r['nb_100pct_query']:+7.2f} | AUROC={r['auroc_routed']:.4f} | "
            f"SkipFlipUB95={r['skipped_case_certificate']['flip_rate_ub95']*100:.2f}%"
        )

    boot_cz_vs_glob = paired_bootstrap_nb_diff(y_te, p_route_cz, q_cz, p_route_glob, q_glob, contract, n_boot=1000, seed=seed)
    boot_pol_vs_glob = paired_bootstrap_nb_diff(y_te, p_route_pol, q_pol, p_route_glob, q_glob, contract, n_boot=1000, seed=seed + 1)
    boot_pol_vs_base = paired_bootstrap_nb_diff(y_te, p_route_pol, q_pol, p1_te, np.zeros(len(y_te), dtype=bool), contract, n_boot=1000, seed=seed + 2)
    print(f"  Paired Boot CZ-Fluid vs Global-CMI     : {boot_cz_vs_glob['diff_mean']:+.2f} [{boot_cz_vs_glob['ci95_low']:+.2f}, {boot_cz_vs_glob['ci95_high']:+.2f}], p={boot_cz_vs_glob['p_two_sided']:.4f}")
    print(f"  Paired Boot Policy-Aware vs Global-CMI : {boot_pol_vs_glob['diff_mean']:+.2f} [{boot_pol_vs_glob['ci95_low']:+.2f}, {boot_pol_vs_glob['ci95_high']:+.2f}], p={boot_pol_vs_glob['p_two_sided']:.4f}")
    print(f"  Paired Boot Policy-Aware vs Stage-1    : {boot_pol_vs_base['diff_mean']:+.2f} [{boot_pol_vs_base['ci95_low']:+.2f}, {boot_pol_vs_base['ci95_high']:+.2f}], p={boot_pol_vs_base['p_two_sided']:.4f}")

    taus = [round(t, 4) for t in np.linspace(max(0.01, contract.tau_star * 0.4), min(0.35, contract.tau_star * 2.5), 9)]
    murphy_pol_vs_glob = compute_murphy_diagram(y_te, p_route_pol, p_route_glob, taus)
    murphy_cz_vs_glob = compute_murphy_diagram(y_te, p_route_cz, p_route_glob, taus)

    sel_audit = None
    if tested_col is not None and tested_col in te.columns:
        t_tr = tr[tested_col].astype(bool).to_numpy()
        t_cal = cal[tested_col].astype(bool).to_numpy()
        t_te = te[tested_col].astype(bool).to_numpy()
        sel_cols = base_cols + [c for cid in disc_pol["selected_channels"] for c in ch_by_id[cid].columns]
        enc_s = FastFrameEncoder().fit(tr.loc[t_tr, sel_cols])
        _, p_sel_te = _fit_global_gbdt_cal(
            enc_s.transform(tr.loc[t_tr, sel_cols]),
            y_tr[t_tr],
            enc_s.transform(cal.loc[t_cal, sel_cols]),
            y_cal[t_cal],
            enc_s.transform(te[sel_cols]),
            seed=seed + 77,
        )
        sel_audit = {
            "selective_label_tested_only_model": selective_label_audit(y_te, p_sel_te, t_te, contract.tau_star, contract.cost_fp, contract.cost_fn),
            "stage1_base": selective_label_audit(y_te, p1_te, t_te, contract.tau_star, contract.cost_fp, contract.cost_fn),
            "global_cmi": selective_label_audit(y_te, p_route_glob, t_te, contract.tau_star, contract.cost_fp, contract.cost_fn, q_glob, contract.per_instance_query_cost),
            "cz_fluid": selective_label_audit(y_te, p_route_cz, t_te, contract.tau_star, contract.cost_fp, contract.cost_fn, q_cz, contract.per_instance_query_cost),
            "policy_aware_fluid": selective_label_audit(y_te, p_route_pol, t_te, contract.tau_star, contract.cost_fp, contract.cost_fn, q_pol, contract.per_instance_query_cost),
        }
        print("  --- Selective-Label Audit (Clinician-Untested / Discharged Test Slice) ---")
        for k_arm, d_arm in sel_audit.items():
            u = d_arm["clinician_untested"]
            t = d_arm["clinician_tested"]
            t_str = f"{t['auroc']:.4f}" if t["auroc"] is not None else "N/A"
            if u["auroc"] is not None:
                u_str = f"Untested AUROC={u['auroc']:.4f}, NB={u['nb_per_1k']:+6.2f}, Rescued={u['rescued_events_at_tau']}/{u['events']} ({u['sensitivity_at_tau']*100:.1f}%)"
            else:
                u_str = f"Untested events={u['events']}/{u['n']} (single-class)"
            print(f"    {k_arm:34s} | Tested AUROC={t_str} | {u_str}")

    fairness = {}
    if fairness_cols:
        for fcol in fairness_cols:
            if fcol not in te.columns:
                continue
            vals = te[fcol].fillna("Missing").astype(str)
            for grp, cnt in vals.value_counts().items():
                if cnt < 100:
                    continue
                m = (vals == grp).to_numpy()
                if y_te[m].sum() < 5:
                    continue
                nb0_g = float(net_benefit_per_1k(y_te[m], p1_te[m], np.zeros(m.sum(), dtype=bool), contract))
                nb_pol_g = float(net_benefit_per_1k(y_te[m], p_route_pol[m], q_pol[m], contract))
                fairness[f"{fcol}:{grp}"] = {
                    "n": int(m.sum()),
                    "prevalence": round(float(y_te[m].mean()), 4),
                    "nb_stage1": round(nb0_g, 3),
                    "nb_policy_fluid": round(nb_pol_g, 3),
                    "delta_nb_gain": round(nb_pol_g - nb0_g, 3),
                    "query_rate": round(float(q_pol[m].mean()), 4),
                }

    return {
        "cohort": name,
        "target": y_col,
        "tau_star": contract.tau_star,
        "budget": budget,
        "n_train": int(len(tr)),
        "n_cal": int(len(cal)),
        "n_test": int(len(te)),
        "test_prevalence": round(float(y_te.mean()), 5),
        "stage1_test_auroc": round(float(roc_auc_score(y_te, p1_te)), 4),
        "stage1_test_auprc": round(float(average_precision_score(y_te, p1_te)), 4),
        "discovery": {
            "global_cmi": disc_glob.selected_ids,
            "cz_schervish": disc_cz.selected_ids,
            "policy_aware": disc_pol,
            "channel_ranking_changed_vs_global": disc_pol["selected_channels"] != disc_glob.selected_ids,
            "null_selected_global": any("null_noise" in c for c in disc_glob.selected_ids),
            "null_selected_cz": any("null_noise" in c for c in disc_cz.selected_ids),
            "null_selected_policy": any("null_noise" in c for c in disc_pol["selected_channels"]),
        },
        "arms": {
            "global_cmi": res_glob,
            "cz_fluid": res_cz,
            "policy_aware_fluid": res_pol,
        },
        "paired_bootstrap": {
            "cz_fluid_vs_global_cmi": boot_cz_vs_glob,
            "policy_aware_vs_global_cmi": boot_pol_vs_glob,
            "policy_aware_vs_stage1": boot_pol_vs_base,
        },
        "murphy_diagrams": {
            "policy_aware_vs_global_cmi": murphy_pol_vs_glob,
            "cz_fluid_vs_global_cmi": murphy_cz_vs_glob,
        },
        "selective_label_audit": sel_audit,
        "fairness_audit": fairness,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main() -> None:
    results: dict[str, Any] = {}

    # 1. BWH ED-ECG
    bwh = pd.read_parquet(PROC_DIR / "bwh_ed_ecg.parquet")
    bwh["race_str"] = np.where(
        bwh["race_white"] == 1, "White",
        np.where(bwh["race_black"] == 1, "Black",
        np.where(bwh["race_hispanic"] == 1, "Hispanic", "Other"))
    )
    bwh["agi_str"] = np.where(
        bwh["agi_under_25k"] == 1, "<25k",
        np.where(bwh["agi_25k_to_50k"] == 1, "25k-50k",
        np.where(bwh["agi_50k_to_75k"] == 1, "50k-75k",
        np.where(bwh["agi_75k_to_100k"] == 1, "75k-100k",
        np.where(bwh["agi_100k_to_200k"] == 1, "100k-200k", ">200k"))))
    )
    bwh_base = [c for c in ["age_at_admit", "sex_male", "vent_rate", "pr_interval", "qrs_duration", "qt_interval", "qtc_interval", "p_axes", "r_axes", "t_axes"] if c in bwh.columns]
    bwh_channels = [
        ChannelDef("ch_ecg_ischemia_flags", [c for c in ["has_st_eleva", "has_depress", "has_twave_inver", "has_twave_abnormal", "has_jpoint_eleva", "has_jpoint_repol", "has_st", "has_twave"] if c in bwh.columns], 0.40),
        ChannelDef("ch_ecg_rhythm_struct", [c for c in ["has_afib", "has_bbb", "has_aberran_bbb", "has_lvh", "has_pacemaker", "has_prolonged_qt", "has_rhythm_disturbance", "has_normal_ecg", "has_normal_sinus"] if c in bwh.columns], 0.40),
        ChannelDef("ch_ecg_waveform_3lead", [c for c in bwh.columns if c.startswith("wf_")], 0.80),
        ChannelDef("ch_sameday_troponin", ["maxtrop_sameday", "log_maxtrop_sameday"], 1.15),
    ]
    results["bwh_mace_death_30"] = run_cohort_suite(
        "BWH ED-ECG: 30-Day MACE or Death (All + Untested)",
        bwh, "macetrop_pos_or_death_030",
        bwh_base, bwh_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=15.0, target_precision=0.25, per_instance_query_cost=0.03),
        budget=2.40, tested_col="test_010_day", fairness_cols=["race_str", "agi_str"], seed=42,
    )
    results["bwh_stent_cabg_10"] = run_cohort_suite(
        "BWH ED-ECG: 10-Day Stent or CABG (Revascularization)",
        bwh, "stent_or_cabg_010_day",
        bwh_base, bwh_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=32.33, target_precision=0.20, per_instance_query_cost=0.03),
        budget=2.40, tested_col="test_010_day", fairness_cols=["race_str", "agi_str"], seed=43,
    )

    # 2. STANFORD MC-MED
    mc = pd.read_parquet(PROC_DIR / "mcmed_stanford.parquet")
    mc["gender_male"] = mc["Gender"].astype(str).str.upper().str.startswith("M").astype(int)
    top_cc = mc.loc[mc["split"] == "train", "CC"].value_counts().head(25).index.tolist()
    mc["cc_clean"] = mc["CC"].where(mc["CC"].isin(top_cc), "OTHER_CC").astype(str)
    mc["arrival_clean"] = mc["Means_of_arrival"].fillna("Missing").astype(str)
    mc_base = [
        "Age", "gender_male", "Triage_Temp", "Triage_HR", "Triage_RR",
        "Triage_SpO2", "Triage_SBP", "Triage_DBP", "esi_num", "cc_clean", "arrival_clean",
    ]
    mc_channels = [
        ChannelDef("ch_hie_pmh", [c for c in mc.columns if c.startswith("pmh_")], 0.40),
        ChannelDef("ch_hie_meds", [c for c in mc.columns if c.startswith("meds_")], 0.40),
        ChannelDef("ch_lab_cbc", [c for c in ["lab_cbc_wbc", "lab_cbc_hgb", "lab_cbc_plt"] if c in mc.columns], 0.78),
        ChannelDef("ch_lab_renal_elyte", [c for c in ["lab_renal_cr", "lab_renal_bun", "lab_elyte_na", "lab_elyte_k", "lab_elyte_hco3", "lab_gluc"] if c in mc.columns], 0.92),
        ChannelDef("ch_lab_lactate_cardiac", [c for c in ["lab_lactate", "lab_troponin", "lab_bnp"] if c in mc.columns], 1.15),
        ChannelDef("ch_lab_lft_coag", [c for c in ["lab_lft_ast", "lab_lft_alt", "lab_lft_bili", "lab_coag_inr"] if c in mc.columns], 0.95),
        ChannelDef("ch_rads_2h", [c for c in mc.columns if c.startswith("rad_")], 1.50),
        ChannelDef("ch_monitor_numerics_2h", [c for c in mc.columns if c.startswith("mon_") or c.startswith("wf_dur_")], 0.60),
    ]
    mc_channels = [ch for ch in mc_channels if len(ch.columns) > 0]

    results["mcmed_critical_icu_death"] = run_cohort_suite(
        "Stanford MC-MED: Critical Decompensation (ICU or In-Hospital Death)",
        mc, "y_critical_icu_death", mc_base, mc_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.03),
        budget=3.0, fairness_cols=["Race", "Payor_class"], seed=44,
    )
    results["mcmed_admission"] = run_cohort_suite(
        "Stanford MC-MED: Hospital Admission (Inpatient or ICU)",
        mc, "y_admit", mc_base, mc_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=5.67, target_precision=0.50, per_instance_query_cost=0.025),
        budget=3.0, fairness_cols=["Race", "Payor_class"], seed=45,
    )
    mc_disch = mc[mc["is_discharged"] == 1].copy()
    results["mcmed_bounce72_admit"] = run_cohort_suite(
        "Stanford MC-MED: 72h Discharged ED Bounceback w/ Hospitalization",
        mc_disch, "y_bounce72_admit", mc_base, mc_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=32.33, target_precision=0.15, per_instance_query_cost=0.025),
        budget=3.0, fairness_cols=["Race", "Payor_class"], seed=46,
    )

    # 3. NTUH ARREST-ECG
    ntuh = pd.read_parquet(PROC_DIR / "ntuh_arrest_ecg.parquet")
    ntuh_base = [
        c for c in [
            "age", "sex_male", "ventricular_rate", "atrial_rate", "pr_interval",
            "qrs_duration", "qt_interval", "qt_corrected", "p_axis", "r_axis", "t_axis", "qtc_frederica",
        ]
        if c in ntuh.columns
    ]
    ntuh_channels = [
        ChannelDef("ch_ant_v1v4", [c for c in ntuh.columns if c.startswith("pl_") and c.endswith("_ant")], 0.60),
        ChannelDef("ch_lat_i_avl_v5v6", [c for c in ntuh.columns if c.startswith("pl_") and c.endswith("_lat")], 0.60),
        ChannelDef("ch_inf_ii_iii_avf", [c for c in ntuh.columns if c.startswith("pl_") and c.endswith("_inf")], 0.60),
        ChannelDef("ch_avr", [c for c in ntuh.columns if c.startswith("pl_") and c.endswith("_avr")], 0.50),
        ChannelDef("ch_ecg_diag_stmts", [c for c in ntuh.columns if c.startswith("dx_")], 0.40),
    ]
    ntuh_channels = [ch for ch in ntuh_channels if len(ch.columns) > 0]
    results["ntuh_pre_arrest"] = run_cohort_suite(
        "NTUH Arrest-ECG: Impending Sudden Cardiac Arrest (2015-2017 -> 2019)",
        ntuh, "y_pre_arrest", ntuh_base, ntuh_channels,
        ClinicalContract(cost_fp=1.0, cost_fn=5.67, target_precision=0.50, per_instance_query_cost=0.02),
        budget=1.80, seed=47,
    )

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print("\nSAVED ALL PHASE 2 CLINICAL RESULTS ->", OUT_JSON)


if __name__ == "__main__":
    main()

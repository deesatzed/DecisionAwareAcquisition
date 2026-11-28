"""Paper 3 (`DecisionAwareAcquisition` v2.1) Comprehensive Remediation & Cross-Paper Upgrade Suite.

Addresses all 10 Round-2 Reviewer Action Items + all 7 Cross-Impact Upgrades from `collision-zone-ed`:
  1. `500`-Pilot Repeated Two-Phase Discovery Benchmark (`seed = 700..1199`) on Stanford ICU/Death
     at rho_pilot = 0.20 (plus `200`-pilot sensitivity at rho_pilot in {0.15, 0.30} and `500` parametric
     Tao-Zeng-Lin / Breslow-Cain offset MLE vs IPW replicates), with exact paired McNemar binomial tests
     and Clopper-Pearson / paired bootstrap 95% CIs (`Actions #2, #7`).
  2. Ported `R = 10` Repeated Patient-Grouped Development Split Estimator Triad (`Bagged VOI` primary
     deployable binary decision, `Sym. Ens.` expected policy value, and `Per-Split` single-model mean)
     on untouched held-out test sets across all 6 primary tasks, recording per-task learned `beta_CZ`
     and `kappa*` (`Cross-Impact #1, #6`).
  3. Stanford ICU Clinical Alert-to-Case Exchange-Rate Table across tau in {3.0%, 5.0%, 8.0%, 10.0%}
     with `10,000`-rep patient-cluster bootstrap 95% CIs on `Delta TP/1k` and `False Alerts Spared/1k`
     (`Cross-Impact #3, #4`).
  4. Published Sequential Instancewise Greedy-CMI AFA (`CovertGreedyCMIAFA`, Covert et al. 2023 policy
     class) across all 6 primary tasks with `10,000`-rep patient-cluster bootstrap CIs (`Action #8`).
  5. Complete `10,000`-rep patient-cluster bootstrap CIs against `Global-CMI + Global GBDT` across all
     6 primary tasks + `[1]ROSC` supplement (`Action #3`).
  6. BWH 10d Revascularization `AUPRC_full` vs `NB` divergence decomposition (`Delta TP`, `Delta FP`,
     query-cost drag, local calibration near tau*=0.030) + label-free marginal intercept alignment check
     on 72h Bounceback and NTUH (`Action #5`, `Cross-Impact #e`).
  7. Stanford ICU `seed=44` within-stratum pilot independence deep-dive: exact outlier feature trace,
     correlation-standardized full-rank Mahalanobis + 1,000-draw exact permutation p-value, 500-seed
     null uniformity check, and outlier-feature exclusion sensitivity (`Action #6`).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_N_THR = os.environ.get("PILOT_WORKER_OMP", "8")
os.environ["OMP_NUM_THREADS"] = _N_THR
os.environ["OPENBLAS_NUM_THREADS"] = _N_THR
os.environ["MKL_NUM_THREADS"] = _N_THR
os.environ["VECLIB_MAXIMUM_THREADS"] = _N_THR

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

FALLBACK_DATA_DIR = Path(
    "/Users/o2satz/.gemini/antigravity/brain/5be51496-dafb-4e38-a543-75823c062544/scratch/paper3_v2_rebuild/data"
)
DATA_DIR = (REPO_ROOT / "data") if (REPO_ROOT / "data" / "mcmed_stanford_2h_v2.parquet").exists() else FALLBACK_DATA_DIR
RES_JSON = REPO_ROOT / "results" / "paper3_v2_remediation_results.json"

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist
from scipy.stats import binomtest, chi2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from collision_zone.three_predicates import (
    ChannelDef,
    ClinicalContract,
    FastFrameEncoder,
    FluidCollisionBackboneClassifier,
    apply_platt,
    bernoulli_kl_from_tau,
    collision_zone_relevance_weights,
    compute_oof_stage1,
    empirical_martingale_voi,
    fit_platt_calibrator,
    fit_stage1_predictions,
    holm_adjust,
    sample_pilot,
    schervish_collision_proper_score,
    solve_ess_kappa,
    two_phase_ht_mean_and_se,
)
from decision_aware_acq.upca_v2_core import (
    adjust_probs_for_target_prevalence,
    case_control_prevalence_weights,
    compute_calibration_metrics,
    discover_stage_a_ablation,
    fit_predict_prepurchase_voi,
    oof_policy_screening_stage_b,
    patient_cluster_bootstrap_diff,
    standardized_net_benefit,
    unit_net_benefit_vec,
    weighted_net_benefit_per_1k,
)


def _clopper_pearson_ci95(k: int, n: int) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    lo = 0.0 if k <= 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return [round(lo, 5), round(hi, 5)]


def _paired_mcnemar_summary(win_a: list[bool], win_b: list[bool], seed: int = 42) -> dict[str, Any]:
    a = np.asarray(win_a, dtype=bool)
    b = np.asarray(win_b, dtype=bool)
    n = int(a.size)
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & (~b)))  # A right, B wrong
    n01 = int(np.sum((~a) & b))  # A wrong, B right
    n00 = int(np.sum((~a) & (~b)))
    disc = n10 + n01
    p_mcnemar = float(binomtest(n10, disc, 0.5, alternative="two-sided").pvalue) if disc > 0 else 1.0

    diff_vec = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(10000, n))
    b_diffs = np.mean(diff_vec[idx], axis=1)
    lo, hi = np.quantile(b_diffs, [0.025, 0.975])
    return {
        "n_pilots": n,
        "rate_a": round(float(np.mean(a)), 5),
        "rate_b": round(float(np.mean(b)), 5),
        "diff_rate": round(float(np.mean(diff_vec)), 5),
        "diff_ci95": [round(float(lo), 5), round(float(hi), 5)],
        "both_correct_n11": n11,
        "a_only_correct_n10": n10,
        "b_only_correct_n01": n01,
        "both_wrong_n00": n00,
        "mcnemar_exact_p_two_sided": round(p_mcnemar, 6),
    }


# ---------------------------------------------------------------------------
# Worker globals for 500-Pilot Parallel Discovery Benchmark (Pre-Encoded Float32)
# ---------------------------------------------------------------------------
_W_XB: np.ndarray | None = None
_W_Z_DICT: dict[str, np.ndarray] | None = None
_W_Y_TR: np.ndarray | None = None
_W_P1_OOF: np.ndarray | None = None
_W_CHS: list[ChannelDef] | None = None
_W_CONTRACT: ClinicalContract | None = None


def _fast_stage_a_preencoded(
    xb_all: np.ndarray,
    z_dict: dict[str, np.ndarray],
    y_train: np.ndarray,
    channels: list[ChannelDef],
    oof_p_base_eval: np.ndarray,
    contract: ClinicalContract,
    budget: float,
    *,
    mode: str,
    pilot_fraction: float,
    seed: int,
    max_steps: int = 8,
    n_jobs: int = 1,
) -> tuple[dict[str, float], list[str]]:
    yv = np.asarray(y_train, dtype=int)
    tau = contract.tau_star
    p_eval_all = np.clip(np.asarray(oof_p_base_eval, dtype=float), 1e-5, 1.0 - 1e-5)
    pilot = sample_pilot(
        yv, p_eval_all, pilot_fraction=pilot_fraction, mode="collision_neyman",
        tau_star=tau, kappa=24.0, n_bins=8, seed=seed,
    )
    p_idx = pilot.indices
    yp = yv[p_idx]
    p_eval_p = p_eval_all[p_idx]
    init_logit_fit = logit(p_eval_p)
    w_ht = pilot.ht_weights
    kappa_eff = solve_ess_kappa(
        bernoulli_kl_from_tau(p_eval_p, tau),
        kappa_max=24.0,
        min_effective_cz_samples=max(160.0, 0.24 * float(yp.size)),
    )

    quantiles = np.quantile(logit(p_eval_all), np.linspace(0.0, 1.0, 9))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    bin_all = np.clip(np.digitize(logit(p_eval_all), quantiles[1:-1], right=False), 0, 7)
    bin_p = bin_all[p_idx]

    log_odds_shift_p = np.zeros(yp.size, dtype=float)
    for b_id in range(8):
        N1 = max(1, int(np.sum((yv == 1) & (bin_all == b_id))))
        N0 = max(1, int(np.sum((yv == 0) & (bin_all == b_id))))
        n1 = max(1, int(np.sum((yp == 1) & (bin_p == b_id))))
        n0 = max(1, int(np.sum((yp == 0) & (bin_p == b_id))))
        log_odds_shift_p[bin_p == b_id] = float(np.log((n1 / float(N1)) / (n0 / float(N0))))

    if mode == "offset_cz_schervish":
        fit_init = init_logit_fit + log_odds_shift_p
        fit_weights = np.ones(yp.size, dtype=float)
    elif mode == "weighted_cz_005":
        fit_init = init_logit_fit.copy()
        fit_weights = w_ht * collision_zone_relevance_weights(p_eval_p, tau, kappa=kappa_eff, floor=0.05)
    elif mode == "ipw_cz_wht":
        fit_init = init_logit_fit.copy()
        fit_weights = w_ht.copy()
    else:
        raise ValueError(mode)

    rng = np.random.default_rng(seed + 17)
    pos = rng.permutation(np.flatnonzero(yp == 1))
    neg = rng.permutation(np.flatnonzero(yp == 0))
    n_folds = 4
    folds = [np.concatenate([pos[k::n_folds], neg[k::n_folds]]) for k in range(n_folds)]
    trs = [np.concatenate([folds[j] for j in range(n_folds) if j != k]) for k in range(n_folds)]

    xb_p = xb_all[p_idx]
    zp_dict = {ch.channel_id: z_dict[ch.channel_id][p_idx] for ch in channels}

    def _oof_probs(sel_ids: list[str], step_seed: int) -> np.ndarray:
        x_p = np.hstack([xb_p] + [zp_dict[cid] for cid in sel_ids]) if sel_ids else xb_p
        oof = np.empty(yp.size, dtype=float)
        for k in range(n_folds):
            va, tr = folds[k], trs[k]
            clf = lgb.LGBMClassifier(
                n_estimators=85, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                max_delta_step=1.0, subsample=0.85, colsample_bytree=0.85,
                random_state=step_seed + k, n_jobs=n_jobs, verbose=-1,
            )
            clf.fit(x_p[tr], yp[tr], sample_weight=fit_weights[tr], init_score=fit_init[tr])
            raw_va = init_logit_fit[va] + clf.predict(x_p[va], raw_score=True)
            oof[va] = np.clip(expit(np.clip(raw_va, -22.0, 22.0)), 1e-6, 1.0 - 1e-6)
        return oof

    p_curr = _oof_probs([], seed + 100)
    sc_curr = schervish_collision_proper_score(yp, p_curr, tau, kappa=kappa_eff)
    selected: list[str] = []
    spent = 0.0
    step1_scores: dict[str, float] = {}

    for step in range(1, min(max_steps, len(channels)) + 1):
        affordable = [ch for ch in channels if ch.channel_id not in selected and spent + ch.cost <= budget + 1e-9]
        if not affordable:
            break
        cand_stats = []
        raw_pvals = []
        for ch in affordable:
            p_cand = _oof_probs(selected + [ch.channel_id], seed + 100 * step)
            sc_cand = schervish_collision_proper_score(yp, p_cand, tau, kappa=kappa_eff)
            psi = sc_cand - sc_curr
            mean_g, se_g = two_phase_ht_mean_and_se(psi, pilot)
            z = mean_g / max(se_g, 1e-12)
            raw_pvals.append(float(1.0 - norm.cdf(z)))
            if step == 1:
                step1_scores[ch.channel_id] = float(mean_g)
            cand_stats.append((ch, mean_g, sc_cand))
        p_holm = holm_adjust(raw_pvals)
        passing = [
            (cand_stats[i][0], cand_stats[i][1] / cand_stats[i][0].cost, cand_stats[i][2])
            for i in range(len(affordable))
            if p_holm[i] < 0.05 and cand_stats[i][1] > 0.0
        ]
        if not passing:
            break
        best_ch, _, best_sc = max(passing, key=lambda item: item[1])
        selected.append(best_ch.channel_id)
        spent += best_ch.cost
        sc_curr = best_sc
    return step1_scores, selected


def _init_pilot_worker(
    xb_all: np.ndarray,
    z_dict: dict[str, np.ndarray],
    y_tr: np.ndarray,
    p1_oof: np.ndarray,
    channels: list[ChannelDef],
    contract: ClinicalContract,
) -> None:
    global _W_XB, _W_Z_DICT, _W_Y_TR, _W_P1_OOF, _W_CHS, _W_CONTRACT
    _W_XB = xb_all
    _W_Z_DICT = z_dict
    _W_Y_TR = y_tr
    _W_P1_OOF = p1_oof
    _W_CHS = channels
    _W_CONTRACT = contract


def _run_one_pilot_rep(args: tuple[int, float]) -> dict[str, Any]:
    rep, rho_p = args
    assert _W_XB is not None and _W_Z_DICT is not None and _W_Y_TR is not None and _W_P1_OOF is not None
    assert _W_CHS is not None and _W_CONTRACT is not None
    do_full = abs(rho_p - 0.20) < 1e-6 and rep < 150
    out_rep: dict[str, Any] = {"rep": rep, "rho_pilot": rho_p, "full_bundle": do_full}
    costs = np.array([ch.cost for ch in _W_CHS], dtype=float)
    for key_m, mode_m in [
        ("weighted_005", "weighted_cz_005"),
        ("ipw_wht", "ipw_cz_wht"),
        ("offset_bc_gbdt", "offset_cz_schervish"),
    ]:
        s1_scores, sel_ids = _fast_stage_a_preencoded(
            _W_XB, _W_Z_DICT, _W_Y_TR, _W_CHS, _W_P1_OOF, _W_CONTRACT, budget=3.0,
            mode=mode_m, pilot_fraction=rho_p, seed=700 + rep,
            max_steps=8 if do_full else 1, n_jobs=1,
        )
        vec_m = np.array([s1_scores[ch.channel_id] for ch in _W_CHS], dtype=float)
        top1_m = _W_CHS[int(np.argmax(vec_m / costs))].channel_id
        out_rep[key_m] = {
            "step1_vec": [float(x) for x in vec_m],
            "top1_id": top1_m,
            "selected": list(sel_ids),
        }
    return out_rep


def run_500_pilot_discovery_suite(
    mc_tr: pd.DataFrame,
    mc_base: list[str],
    mc_channels: list[ChannelDef],
) -> dict[str, Any]:
    t0 = time.time()
    print("\n=== [1/6] Running 500-Pilot Repeated Two-Phase Discovery Suite on Stanford ICU/Death ===")
    y_tr = mc_tr["y_critical_icu_death"].astype(int).to_numpy()
    p1_oof = compute_oof_stage1(mc_tr[mc_base], y_tr, n_folds=3, seed=44)
    contract = ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.030)

    # Pre-encode full training matrices once (float32) so 10 workers execute pure NumPy + single-thread LightGBM
    xb_all = FastFrameEncoder().fit(mc_tr[mc_base]).transform(mc_tr[mc_base]).astype(np.float32)
    z_dict = {
        ch.channel_id: FastFrameEncoder().fit(mc_tr[list(ch.columns)]).transform(mc_tr[list(ch.columns)]).astype(np.float32)
        for ch in mc_channels
    }
    s1_ora, sel_ora = _fast_stage_a_preencoded(
        xb_all, z_dict, y_tr, mc_channels, p1_oof, contract, budget=3.0,
        mode="offset_cz_schervish", pilot_fraction=1.0, seed=44, n_jobs=8,
    )
    oracle_vec = np.array([s1_ora[ch.channel_id] for ch in mc_channels], dtype=float)
    costs = np.array([ch.cost for ch in mc_channels], dtype=float)
    oracle_top1 = mc_channels[int(np.argmax(oracle_vec / costs))].channel_id
    oracle_set = set(sel_ora)

    # 1. Parametric 500-pilot Tao-Zeng-Lin / Breslow-Cain Offset MLE vs IPW
    enc_lac = FastFrameEncoder().fit(mc_tr[list(mc_channels[4].columns)])
    z_lac_raw = enc_lac.transform(mc_tr[list(mc_channels[4].columns)])[:, :1]
    z_lac = (z_lac_raw - np.mean(z_lac_raw)) / (np.std(z_lac_raw) + 1e-6)
    ell1 = logit(np.clip(p1_oof, 1e-5, 1.0 - 1e-5))
    x_param_all = np.column_stack([ell1, z_lac, z_lac ** 2])
    lr_full = LogisticRegression(C=100.0, solver="lbfgs", max_iter=500).fit(x_param_all, y_tr)
    beta_oracle = float(lr_full.coef_[0, 1])

    quantiles = np.quantile(ell1, np.linspace(0.0, 1.0, 9))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    bin_all = np.clip(np.digitize(ell1, quantiles[1:-1], right=False), 0, 7)

    beta_err_ipw, beta_err_tzl = [], []
    for rep in range(500):
        pilot = sample_pilot(y_tr, p1_oof, pilot_fraction=0.20, mode="collision_neyman", tau_star=contract.tau_star, n_bins=8, seed=700 + rep)
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

        xp = x_param_all[p_idx]
        lr_ipw = LogisticRegression(C=100.0, solver="lbfgs", max_iter=500).fit(xp, yp, sample_weight=w_ht)
        beta_err_ipw.append((float(lr_ipw.coef_[0, 1]) - beta_oracle) ** 2)

        b_vec = np.zeros(4, dtype=float)
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
        beta_err_tzl.append((float(b_vec[2]) - beta_oracle) ** 2)

    # 2. Parallel GBDT Pilot Discovery: 500 reps at rho=0.20 + 200 reps each at rho=0.15 and rho=0.30
    tasks_pool = (
        [(r, 0.20) for r in range(500)]
        + [(r, 0.15) for r in range(200)]
        + [(r, 0.30) for r in range(200)]
    )
    by_rho: dict[float, list[dict[str, Any]]] = {0.15: [], 0.20: [], 0.30: []}
    os.environ["PILOT_WORKER_OMP"] = "1"
    os.environ["UPCA_LGBM_JOBS"] = "1"
    with cf.ProcessPoolExecutor(
        max_workers=10,
        initializer=_init_pilot_worker,
        initargs=(xb_all, z_dict, y_tr, p1_oof, mc_channels, contract),
    ) as ex:
        for idx_done, item in enumerate(ex.map(_run_one_pilot_rep, tasks_pool, chunksize=3)):
            by_rho[float(item["rho_pilot"])].append(item)
            if (idx_done + 1) % 100 == 0:
                print(f"  [500-Pilot Suite] Completed {idx_done + 1}/{len(tasks_pool)} pilot runs ({time.time()-t0:.1f}s)...", flush=True)
    os.environ["PILOT_WORKER_OMP"] = "8"
    os.environ["UPCA_LGBM_JOBS"] = "8"

    rho_summaries: dict[str, Any] = {}
    for rho_val, rows_r in by_rho.items():
        rows_r.sort(key=lambda d: int(d["rep"]))
        n_r = len(rows_r)
        top1_m: dict[str, list[bool]] = {"weighted_005": [], "ipw_wht": [], "offset_bc_gbdt": []}
        mon_rec: dict[str, list[bool]] = {"weighted_005": [], "ipw_wht": [], "offset_bc_gbdt": []}
        jacc_m: dict[str, list[float]] = {"weighted_005": [], "ipw_wht": [], "offset_bc_gbdt": []}
        rmse_m: dict[str, list[float]] = {"weighted_005": [], "ipw_wht": [], "offset_bc_gbdt": []}
        for row in rows_r:
            is_fb = bool(row.get("full_bundle", False))
            for km in ("weighted_005", "ipw_wht", "offset_bc_gbdt"):
                d_km = row[km]
                v_km = np.array(d_km["step1_vec"], dtype=float)
                s_km = set(d_km["selected"])
                top1_m[km].append(d_km["top1_id"] == oracle_top1)
                rmse_m[km].append(float(np.sqrt(np.mean((v_km - oracle_vec) ** 2))))
                if is_fb:
                    mon_rec[km].append("ch_monitor_numerics_2h" in s_km)
                    union_sz = max(1, len(s_km | oracle_set))
                    jacc_m[km].append(float(len(s_km & oracle_set)) / float(union_sz))

        n_fb = len(mon_rec["offset_bc_gbdt"])
        sum_entry: dict[str, Any] = {
            "n_pilots": n_r,
            "n_full_bundle_pilots": n_fb,
            "top1_concordance": {
                km: {
                    "k": int(np.sum(top1_m[km])),
                    "rate": round(float(np.mean(top1_m[km])), 4),
                    "ci95_clopper_pearson": _clopper_pearson_ci95(int(np.sum(top1_m[km])), n_r),
                }
                for km in ("weighted_005", "ipw_wht", "offset_bc_gbdt")
            },
            "step1_score_rmse_mean": {km: round(float(np.mean(rmse_m[km])), 6) for km in rmse_m},
            "mcnemar_top1_offset_vs_weighted_005": _paired_mcnemar_summary(top1_m["offset_bc_gbdt"], top1_m["weighted_005"], seed=91),
            "mcnemar_top1_offset_vs_ipw_wht": _paired_mcnemar_summary(top1_m["offset_bc_gbdt"], top1_m["ipw_wht"], seed=92),
        }
        if n_fb > 0:
            sum_entry["monitor_numerics_bundle_recovery"] = {
                km: {
                    "k": int(np.sum(mon_rec[km])),
                    "rate": round(float(np.mean(mon_rec[km])), 4),
                    "ci95_clopper_pearson": _clopper_pearson_ci95(int(np.sum(mon_rec[km])), n_fb),
                }
                for km in ("weighted_005", "ipw_wht", "offset_bc_gbdt")
            }
            sum_entry["oracle_bundle_jaccard_mean"] = {km: round(float(np.mean(jacc_m[km])), 4) for km in jacc_m}
            sum_entry["mcnemar_monitor_recovery_offset_vs_weighted_005"] = _paired_mcnemar_summary(
                mon_rec["offset_bc_gbdt"], mon_rec["weighted_005"], seed=93
            )
            sum_entry["mcnemar_monitor_recovery_offset_vs_ipw_wht"] = _paired_mcnemar_summary(
                mon_rec["offset_bc_gbdt"], mon_rec["ipw_wht"], seed=94
            )
        rho_summaries[f"rho_{rho_val:.2f}"] = sum_entry

    primary_r20 = rho_summaries["rho_0.20"]
    out = {
        "oracle_top1_channel": oracle_top1,
        "oracle_selected_bundle": list(sel_ora),
        "n_gbdt_pilot_reps": 500,
        "n_parametric_pilot_reps": 500,
        "gbdt_step1_score_rmse_vs_full_cohort": primary_r20["step1_score_rmse_mean"],
        "gbdt_top1_channel_concordance": {k: v["rate"] for k, v in primary_r20["top1_concordance"].items()},
        "primary_rho_020_detailed": primary_r20,
        "rho_sensitivity_repeated_pilots": rho_summaries,
        "parametric_lactate_slope_rmse": {
            "ipw_logistic_wht": round(float(np.sqrt(np.mean(beta_err_ipw))), 5),
            "two_phase_offset_mle_tao_zeng_lin": round(float(np.sqrt(np.mean(beta_err_tzl))), 5),
            "relative_variance_reduction_pct": round(
                100.0 * (1.0 - float(np.mean(beta_err_tzl)) / max(1e-12, float(np.mean(beta_err_ipw)))), 2
            ),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("  [500-Pilot Suite Summary]:", json.dumps(out["primary_rho_020_detailed"], indent=2), flush=True)
    return out


# ---------------------------------------------------------------------------
# Published Sequential Instancewise Greedy-CMI AFA (`CovertGreedyCMIAFA`)
# Adapted cleanly for Case-Control Prior-Odds Target Prevalence (`w_cal, w_te`)
# ---------------------------------------------------------------------------
def evaluate_covert_afa_clinical_task(
    tr: pd.DataFrame,
    y_tr: np.ndarray,
    cal: pd.DataFrame,
    y_cal: np.ndarray,
    te: pd.DataFrame,
    y_te: np.ndarray,
    clusters_te: np.ndarray,
    base_cols: list[str],
    ch_all: list[ChannelDef],
    p1_oof_raw: np.ndarray,
    p1_cal_raw: np.ndarray,
    p1_te_raw: np.ndarray,
    p1_cal_eval: np.ndarray,
    p1_te_eval: np.ndarray,
    contract: ClinicalContract,
    budget: float,
    *,
    pi_target_override: float | None = None,
    w_cal: np.ndarray | None = None,
    w_te: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    m_ch = len(ch_all)
    costs = np.array([float(ch.cost) for ch in ch_all], dtype=float)
    enc_b = FastFrameEncoder().fit(tr[base_cols])
    xb_tr = enc_b.transform(tr[base_cols]).astype(np.float32)
    xb_ca = enc_b.transform(cal[base_cols]).astype(np.float32)
    xb_te = enc_b.transform(te[base_cols]).astype(np.float32)

    ch_slices: list[tuple[int, int]] = []
    z_tr_blks, z_ca_blks, z_te_blks = [], [], []
    cur = 0
    for ch in ch_all:
        cols_c = list(ch.columns)
        enc_c = FastFrameEncoder().fit(tr[cols_c])
        b_tr = enc_c.transform(tr[cols_c]).astype(np.float32)
        b_ca = enc_c.transform(cal[cols_c]).astype(np.float32)
        b_te = enc_c.transform(te[cols_c]).astype(np.float32)
        w_c = b_tr.shape[1]
        ch_slices.append((cur, cur + w_c))
        cur += w_c
        z_tr_blks.append(b_tr)
        z_ca_blks.append(b_ca)
        z_te_blks.append(b_te)

    zf_tr = np.hstack(z_tr_blks)
    zf_ca = np.hstack(z_ca_blks)
    zf_te = np.hstack(z_te_blks)
    sentinel = np.float32(-999.0)

    def _masked(xb: np.ndarray, zf: np.ndarray, mm: np.ndarray) -> np.ndarray:
        zm = np.full_like(zf, sentinel)
        for j, (c0, c1) in enumerate(ch_slices):
            obs = mm[:, j] > 0.5
            if np.any(obs):
                zm[obs, c0:c1] = zf[obs, c0:c1]
        return np.hstack([xb, zm, mm.astype(np.float32)])

    rng = np.random.default_rng(seed + 303)

    def _sample_masks(n_rows: int) -> np.ndarray:
        masks = np.zeros((n_rows, m_ch), dtype=np.float32)
        for i in range(n_rows):
            perm = rng.permutation(m_ch)
            tk = int(rng.integers(1, min(4, m_ch) + 1))
            rem_b = budget
            picked = 0
            for j in perm:
                if picked >= tk:
                    break
                if costs[j] <= rem_b + 1e-9:
                    masks[i, j] = 1.0
                    rem_b -= costs[j]
                    picked += 1
        return masks

    n_tr = y_tr.size
    ell1_tr = logit(np.clip(p1_oof_raw, 1e-5, 1.0 - 1e-5))
    x_reps, y_reps, i_reps = [], [], []
    n_masks = 4 if n_tr > 25000 else 5
    for _ in range(n_masks):
        mm = _sample_masks(n_tr)
        x_reps.append(_masked(xb_tr, zf_tr, mm))
        y_reps.append(y_tr)
        i_reps.append(ell1_tr)

    clf_m = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.05, num_leaves=31, min_child_samples=30,
        max_delta_step=1.0, subsample=0.85, colsample_bytree=0.85,
        random_state=seed + 41, n_jobs=4, verbose=-1,
    ).fit(np.vstack(x_reps), np.concatenate(y_reps), init_score=np.concatenate(i_reps))

    m_ca_r = _sample_masks(y_cal.size)
    raw_ca_m = logit(np.clip(p1_cal_raw, 1e-5, 1 - 1e-5)) + clf_m.predict(_masked(xb_ca, zf_ca, m_ca_r), raw_score=True)
    platt_m = fit_platt_calibrator(raw_ca_m, y_cal)
    pi_cal_s = float(np.mean(y_cal))

    def _pred_eval_p(xb: np.ndarray, zf: np.ndarray, p1_raw_v: np.ndarray, p1_eval_v: np.ndarray, mm: np.ndarray) -> np.ndarray:
        has_any = np.any(mm > 0.5, axis=1)
        out = p1_eval_v.copy()
        if np.any(has_any):
            xm = _masked(xb[has_any], zf[has_any], mm[has_any])
            raw = logit(np.clip(p1_raw_v[has_any], 1e-5, 1 - 1e-5)) + clf_m.predict(xm, raw_score=True)
            p_cal_sc = np.clip(apply_platt(raw, platt_m), 1e-6, 1.0 - 1e-6)
            if pi_target_override is not None:
                p_cal_sc = adjust_probs_for_target_prevalence(p_cal_sc, pi_cal_s, pi_target_override)
            out[has_any] = p_cal_sc
        return out

    sub_n = min(n_tr, 6000)
    sub_idx = rng.choice(n_tr, size=sub_n, replace=False) if sub_n < n_tr else np.arange(n_tr)
    xb_s, zf_s = xb_tr[sub_idx], zf_tr[sub_idx]
    p1_s_raw = np.clip(p1_oof_raw[sub_idx], 1e-5, 1 - 1e-5)
    p1_s_eval = p1_s_raw if pi_target_override is None else adjust_probs_for_target_prevalence(p1_s_raw, float(np.mean(y_tr)), pi_target_override)

    m_zero = np.zeros((sub_n, m_ch), dtype=np.float32)
    m_part = np.zeros((sub_n, m_ch), dtype=np.float32)
    for i in range(sub_n):
        j0 = int(rng.integers(0, m_ch))
        if costs[j0] < budget - 0.35:
            m_part[i, j0] = 1.0

    st_m = np.vstack([m_zero, m_part])
    st_xb = np.vstack([xb_s, xb_s])
    st_zf = np.vstack([zf_s, zf_s])
    st_p1r = np.concatenate([p1_s_raw, p1_s_raw])
    st_p1e = np.concatenate([p1_s_eval, p1_s_eval])
    p_cur_st = _pred_eval_p(st_xb, st_zf, st_p1r, st_p1e, st_m)
    x_st_feat = np.column_stack([logit(p_cur_st).astype(np.float32), _masked(st_xb, st_zf, st_m)])

    cmi_regs: list[lgb.LGBMRegressor | None] = []
    for j in range(m_ch):
        valid = (st_m[:, j] < 0.5) & ((st_m @ costs) + costs[j] <= budget + 1e-9)
        if int(np.sum(valid)) < 40:
            cmi_regs.append(None)
            continue
        m_plus = st_m[valid].copy()
        m_plus[:, j] = 1.0
        p_plus = _pred_eval_p(st_xb[valid], st_zf[valid], st_p1r[valid], st_p1e[valid], m_plus)
        pb = p_cur_st[valid]
        cmi_t = np.maximum(0.0, p_plus * np.log(p_plus / pb) + (1.0 - p_plus) * np.log((1.0 - p_plus) / (1.0 - pb)))
        reg_j = lgb.LGBMRegressor(
            n_estimators=55, learning_rate=0.06, num_leaves=15, min_child_samples=25,
            subsample=0.85, colsample_bytree=0.85, random_state=seed + 200 + j, n_jobs=4, verbose=-1,
        ).fit(x_st_feat[valid], cmi_t)
        cmi_regs.append(reg_j)

    def _rollout(xb: np.ndarray, zf: np.ndarray, p1r: np.ndarray, p1e: np.ndarray, lam: float, max_steps: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = p1e.size
        m_cur = np.zeros((n, m_ch), dtype=np.float32)
        sp_cur = np.zeros(n, dtype=float)
        p_cur = p1e.copy()
        active = np.ones(n, dtype=bool)
        s1_best = np.zeros(n, dtype=float)
        for step in range(max_steps):
            act_i = np.flatnonzero(active)
            if act_i.size == 0:
                break
            x_st = np.column_stack([logit(p_cur[act_i]).astype(np.float32), _masked(xb[act_i], zf[act_i], m_cur[act_i])])
            ratios = np.full((act_i.size, m_ch), -np.inf, dtype=float)
            for j, rj in enumerate(cmi_regs):
                if rj is None:
                    continue
                can = (m_cur[act_i, j] < 0.5) & (sp_cur[act_i] + costs[j] <= budget + 1e-9)
                if np.any(can):
                    ratios[can, j] = np.maximum(0.0, rj.predict(x_st[can])) / costs[j]
            best_j = np.argmax(ratios, axis=1)
            best_r = ratios[np.arange(act_i.size), best_j]
            if step == 0:
                s1_best[act_i] = np.where(np.isfinite(best_r), best_r, 0.0)
            go = np.isfinite(best_r) & (best_r > lam)
            if not np.any(go):
                break
            go_rows = act_i[go]
            go_ch = best_j[go]
            m_cur[go_rows, go_ch] = 1.0
            sp_cur[go_rows] += costs[go_ch]
            p_cur[go_rows] = _pred_eval_p(xb[go_rows], zf[go_rows], p1r[go_rows], p1e[go_rows], m_cur[go_rows])
            active[~np.isin(np.arange(n), go_rows)] = False
        q_any = np.any(m_cur > 0.5, axis=1)
        return p_cur, q_any, m_cur, s1_best

    _, _, _, cal_r1 = _rollout(xb_ca, zf_ca, p1_cal_raw, p1_cal_eval, lam=1e9, max_steps=1)
    pos_r = cal_r1[cal_r1 > 0.0]
    cand_lams = np.unique(np.concatenate([np.quantile(pos_r, [0.10, 0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 0.96]), [1e9]])) if pos_r.size >= 10 else np.array([1e-4, 1e-3, 1e-2, 1e9])

    best_lam = float(cand_lams[-1])
    best_nb = -np.inf
    for lam in cand_lams:
        pc_t, qc_t, _, _ = _rollout(xb_ca, zf_ca, p1_cal_raw, p1_cal_eval, float(lam))
        nb_t = weighted_net_benefit_per_1k(y_cal, pc_t, qc_t, contract, sample_weight=w_cal)
        if nb_t > best_nb:
            best_nb = nb_t
            best_lam = float(lam)

    p_rt, q_t, m_te, _ = _rollout(xb_te, zf_te, p1_te_raw, p1_te_eval, best_lam)
    p_full, _, m_full, _ = _rollout(xb_te, zf_te, p1_te_raw, p1_te_eval, 0.0)
    eval_prev = float(pi_target_override) if pi_target_override is not None else float(np.mean(y_te))

    nb_rt = weighted_net_benefit_per_1k(y_te, p_rt, q_t, contract, sample_weight=w_te)
    q_rate = float(np.mean(q_t)) if w_te is None else float(np.sum(w_te * q_t) / np.sum(w_te))
    null_idxs = [j for j, ch in enumerate(ch_all) if ch.channel_id.startswith("ch_null")]
    null_q_rate = float(np.mean(np.any(m_te[:, null_idxs] > 0.5, axis=1))) if null_idxs else 0.0
    top_ch_freqs = {
        ch_all[j].channel_id: round(float(np.mean(m_te[:, j] > 0.5)), 4)
        for j in range(m_ch)
        if float(np.mean(m_te[:, j] > 0.5)) > 0.005
    }
    mean_spent_when_q = float(np.mean((m_te @ costs)[q_t])) if np.any(q_t) else 0.0

    arm_dict = {
        "policy_type": "instancewise_sequential_greedy_cmi",
        "selected": sorted(top_ch_freqs.keys(), key=lambda k: top_ch_freqs[k], reverse=True)[:3],
        "channel_query_frequencies": top_ch_freqs,
        "mean_spent_when_queried": round(mean_spent_when_q, 3),
        "null_channel_query_rate": round(null_q_rate, 4),
        "nb_routed": round(nb_rt, 4),
        "snb_routed": round(standardized_net_benefit(nb_rt, eval_prev), 4),
        "query_rate": round(q_rate, 4),
        "auprc_full": round(float(average_precision_score(y_te, p_full, sample_weight=w_te)), 4),
        "auprc_routed": round(float(average_precision_score(y_te, p_rt, sample_weight=w_te)), 4),
        "auroc_full_supplement": round(float(roc_auc_score(y_te, p_full, sample_weight=w_te)), 4),
        "auroc_routed_supplement": round(float(roc_auc_score(y_te, p_rt, sample_weight=w_te)), 4),
    }
    return arm_dict, p_rt, q_t


# ---------------------------------------------------------------------------
# Patient-Grouped Development Split Helper + R=10 Estimator Triad
# ---------------------------------------------------------------------------
def _patient_grouped_dev_split(
    clusters_dev: np.ndarray,
    y_dev: np.ndarray,
    cal_frac: float = 0.20,
    seed: int = 2100,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    df_g = pd.DataFrame({"cid": clusters_dev, "y": y_dev})
    c_pos = df_g.groupby("cid", sort=False)["y"].max()
    pos_c = rng.permutation(c_pos.index[c_pos == 1].to_numpy())
    neg_c = rng.permutation(c_pos.index[c_pos == 0].to_numpy())
    n_pos_tr = max(1, round((1.0 - cal_frac) * len(pos_c)))
    n_neg_tr = max(1, round((1.0 - cal_frac) * len(neg_c)))
    tr_set = set(pos_c[:n_pos_tr]).union(set(neg_c[:n_neg_tr]))
    in_tr = pd.Series(clusters_dev).isin(tr_set).to_numpy()
    return np.flatnonzero(in_tr), np.flatnonzero(~in_tr)


def _cluster_exchange_table(
    y_te: np.ndarray,
    p_upca: np.ndarray,
    q_upca: np.ndarray,
    p_ref: np.ndarray,
    q_ref: np.ndarray,
    clusters_te: np.ndarray,
    taus: list[float],
    c_query: float,
) -> list[dict[str, Any]]:
    yv = np.asarray(y_te, dtype=float)
    _, inv = np.unique(clusters_te, return_inverse=True)
    n_c = int(inv.max()) + 1
    c_den = np.bincount(inv, minlength=n_c).astype(float)
    probs = np.full(n_c, 1.0 / float(n_c))

    rows = []
    for tau in taus:
        odds = tau / (1.0 - tau)
        c_fn_t = (1.0 - tau) / tau
        gamma_q = c_query / c_fn_t

        act_u = (p_upca >= tau).astype(float)
        act_r = (p_ref >= tau).astype(float)
        tp_u = 1000.0 * float(np.mean(act_u * yv))
        fp_u = 1000.0 * float(np.mean(act_u * (1.0 - yv)))
        tp_r = 1000.0 * float(np.mean(act_r * yv))
        fp_r = 1000.0 * float(np.mean(act_r * (1.0 - yv)))
        ppv_u = float(tp_u / max(1e-9, tp_u + fp_u))
        nb_u = tp_u - odds * fp_u - gamma_q * 1000.0 * float(np.mean(q_upca))
        nb_r = tp_r - odds * fp_r - gamma_q * 1000.0 * float(np.mean(q_ref))

        dtp_i = 1000.0 * ((act_u - act_r) * yv)
        sp_i = 1000.0 * ((act_r - act_u) * (1.0 - yv))
        c_dtp = np.bincount(inv, weights=dtp_i, minlength=n_c)
        c_sp = np.bincount(inv, weights=sp_i, minlength=n_c)
        rng = np.random.default_rng(int(round(tau * 100000)) + 701)
        b_dtp = np.empty(10000, dtype=float)
        b_sp = np.empty(10000, dtype=float)
        for st in range(0, 10000, 2500):
            en = min(10000, st + 2500)
            cnts = rng.multinomial(n_c, probs, size=en - st).astype(float)
            den = np.maximum(1e-12, cnts @ c_den)
            b_dtp[st:en] = (cnts @ c_dtp) / den
            b_sp[st:en] = (cnts @ c_sp) / den

        def _ci_p(b_arr: np.ndarray, diff_obs: float) -> tuple[list[float], float]:
            if abs(diff_obs) <= 1e-12 and float(np.std(b_arr)) <= 1e-12:
                return [0.0, 0.0], 1.0
            lo, hi = np.quantile(b_arr, [0.025, 0.975])
            kle = int(np.sum(b_arr <= 0.0))
            kge = int(np.sum(b_arr >= 0.0))
            p = min(1.0, 2.0 * float(min(kle, kge) + 1) / 10001.0)
            return [round(float(lo), 2), round(float(hi), 2)], round(float(p), 4)

        dtp_ci, dtp_p = _ci_p(b_dtp, tp_u - tp_r)
        sp_ci, sp_p = _ci_p(b_sp, fp_r - fp_u)
        rows.append({
            "tau": round(tau, 4),
            "indifference_fp_per_tp": round(c_fn_t, 2),
            "upca_tp_per_1k": round(tp_u, 2),
            "ref_tp_per_1k": round(tp_r, 2),
            "delta_tp_per_1k": round(tp_u - tp_r, 2),
            "delta_tp_ci95": dtp_ci,
            "delta_tp_p_dh": dtp_p,
            "upca_fp_per_1k": round(fp_u, 2),
            "ref_fp_per_1k": round(fp_r, 2),
            "upca_ppv_pct": round(100.0 * ppv_u, 2),
            "alerts_spared_per_1k": round(fp_r - fp_u, 2),
            "alerts_spared_ci95": sp_ci,
            "alerts_spared_p_dh": sp_p,
            "upca_nb_per_1k": round(nb_u, 2),
            "ref_nb_per_1k": round(nb_r, 2),
            "delta_nb_per_1k": round(nb_u - nb_r, 2),
        })
    return rows


def run_r10_triad_and_task_upgrades(
    task_id: str,
    df: pd.DataFrame,
    y_col: str,
    cluster_col: str,
    base_cols: list[str],
    channels: list[ChannelDef],
    contract: ClinicalContract,
    budget: float,
    *,
    pi_target_override: float | None = None,
    canonical_seed: int = 42,
    n_splits: int = 10,
) -> dict[str, Any]:
    t0 = time.time()
    rng0 = np.random.default_rng(canonical_seed)
    df_w = df.copy()
    df_w["null_1a"], df_w["null_1b"] = rng0.normal(size=len(df_w)), rng0.normal(size=len(df_w))
    df_w["null_2a"], df_w["null_2b"] = rng0.normal(size=len(df_w)), rng0.normal(size=len(df_w))
    ch_all = list(channels) + [
        ChannelDef("ch_null_noise_1", ("null_1a", "null_1b"), 0.50, is_null_stress=True),
        ChannelDef("ch_null_noise_2", ("null_2a", "null_2b"), 0.50, is_null_stress=True),
    ]
    ch_map = {c.channel_id: c for c in ch_all}

    tr0 = df_w[df_w["split"] == "train"].reset_index(drop=True)
    cal0 = df_w[df_w["split"] == "cal"].reset_index(drop=True)
    te = df_w[df_w["split"] == "test"].reset_index(drop=True)
    dev = pd.concat([tr0, cal0], ignore_index=True)

    y_tr0 = tr0[y_col].astype(int).to_numpy()
    y_cal0 = cal0[y_col].astype(int).to_numpy()
    y_te = te[y_col].astype(int).to_numpy()
    clusters_te = te[cluster_col].to_numpy()
    n_te = y_te.size

    w_cal0 = case_control_prevalence_weights(y_cal0, pi_target_override) if pi_target_override is not None else None
    w_te = case_control_prevalence_weights(y_te, pi_target_override) if pi_target_override is not None else None
    eval_prev = float(pi_target_override) if pi_target_override is not None else float(np.mean(y_te))

    # --- A. Canonical Single-Split Supplements: Covert-AFA + CMI+GBDT Cluster Bootstraps + beta_CZ ---
    p1_oof0_raw = compute_oof_stage1(tr0[base_cols], y_tr0, n_folds=3, seed=canonical_seed)
    p1_cal0_raw, p1_tes0_raw = fit_stage1_predictions(tr0, y_tr0, cal0, y_cal0, [te], base_cols, seed=canonical_seed)
    p1_te0_raw = p1_tes0_raw[0]
    if pi_target_override is not None:
        w_tr0 = case_control_prevalence_weights(y_tr0, pi_target_override)
        p1_oof0_eval = adjust_probs_for_target_prevalence(p1_oof0_raw, float(np.mean(y_tr0)), pi_target_override)
        p1_cal0_eval = adjust_probs_for_target_prevalence(p1_cal0_raw, float(np.mean(y_cal0)), pi_target_override)
        p1_te0_eval = adjust_probs_for_target_prevalence(p1_te0_raw, float(np.mean(y_cal0)), pi_target_override)
    else:
        w_tr0 = None
        p1_oof0_eval, p1_cal0_eval, p1_te0_eval = p1_oof0_raw, p1_cal0_raw, p1_te0_raw

    afa_arm, pr_afa, q_afa = evaluate_covert_afa_clinical_task(
        tr0, y_tr0, cal0, y_cal0, te, y_te, clusters_te, base_cols, ch_all,
        p1_oof0_raw, p1_cal0_raw, p1_te0_raw, p1_cal0_eval, p1_te0_eval,
        contract, budget, pi_target_override=pi_target_override, w_cal=w_cal0, w_te=w_te, seed=canonical_seed,
    )

    # Re-extract canonical split predictions for UPCA-Fluid and Global-CMI + Global GBDT + learned beta_CZ
    disc_glob0, _ = discover_stage_a_ablation(
        tr0, y_tr0, base_cols, ch_all, p1_oof0_eval, contract, budget,
        mode="global_cmi", pilot_fraction=0.20, target_pop_weights=w_tr0, raw_oof_p_base_for_fit=p1_oof0_raw, seed=canonical_seed,
    )
    disc_off0, _ = discover_stage_a_ablation(
        tr0, y_tr0, base_cols, ch_all, p1_oof0_eval, contract, budget,
        mode="offset_cz_schervish", pilot_fraction=0.20, target_pop_weights=w_tr0, raw_oof_p_base_for_fit=p1_oof0_raw, seed=canonical_seed,
    )
    upca_sel0, _ = oof_policy_screening_stage_b(
        disc_off0.selected_ids, tr0, y_tr0, p1_oof0_raw, base_cols, ch_map, contract,
        pi_target_override=pi_target_override, seed=canonical_seed,
    )

    enc_b0 = FastFrameEncoder().fit(tr0[base_cols])
    xb_tr0, xb_ca0, xb_te0 = enc_b0.transform(tr0[base_cols]), enc_b0.transform(cal0[base_cols]), enc_b0.transform(te[base_cols])

    def _fit_eval_split(
        tr_s: pd.DataFrame, y_tr_s: np.ndarray, p1_oof_r: np.ndarray, p1_oof_e: np.ndarray,
        cal_s: pd.DataFrame, y_ca_s: np.ndarray, p1_ca_r: np.ndarray, p1_ca_e: np.ndarray,
        xb_tr_s: np.ndarray, xb_ca_s: np.ndarray, xb_te_s: np.ndarray, p1_te_r: np.ndarray, p1_te_e: np.ndarray,
        sel_ids: list[str], use_fluid: bool, seed_s: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        if not sel_ids:
            return p1_te_e.copy(), np.zeros(n_te, dtype=bool), p1_te_e.copy(), p1_ca_e.copy(), 0.0, 0.0
        s2_cols = [c for cid in sel_ids for c in ch_map[cid].columns]
        enc_z = FastFrameEncoder().fit(tr_s[s2_cols])
        z_tr, z_ca, z_te = enc_z.transform(tr_s[s2_cols]), enc_z.transform(cal_s[s2_cols]), enc_z.transform(te[s2_cols])
        x2_tr, x2_ca, x2_te = np.hstack([xb_tr_s, z_tr]), np.hstack([xb_ca_s, z_ca]), np.hstack([xb_te_s, z_te])
        p1_s = np.clip(p1_oof_r, 1e-5, 1.0 - 1e-5)
        beta_cz_val, kappa_val = 0.0, 0.0
        if use_fluid:
            tau_fit = (
                contract.tau_star if pi_target_override is None
                else float(adjust_probs_for_target_prevalence(np.array([contract.tau_star]), pi_target_override, float(np.mean(y_tr_s)))[0])
            )
            clf = FluidCollisionBackboneClassifier(
                tau_star=tau_fit, kappa_max=20.0, min_cz_ess=240.0, seed=seed_s + 7
            ).fit(x2_tr, z_tr, y_tr_s, logit(p1_s))
            beta_cz_val = float(clf.beta_cz_)
            kappa_val = float(clf.kappa_)
            r_tr = clf.decision_function(x2_tr, z_tr, logit(p1_s))
            r_c = clf.decision_function(x2_ca, z_ca, logit(np.clip(p1_ca_r, 1e-5, 1 - 1e-5)))
            r_t = clf.decision_function(x2_te, z_te, logit(np.clip(p1_te_r, 1e-5, 1 - 1e-5)))
        else:
            clf_g = lgb.LGBMClassifier(
                n_estimators=140, learning_rate=0.05, num_leaves=31, min_child_samples=30,
                max_delta_step=1.0, subsample=0.85, colsample_bytree=0.85,
                random_state=seed_s + 7, n_jobs=4, verbose=-1,
            ).fit(x2_tr, y_tr_s)
            r_tr = clf_g.predict(x2_tr, raw_score=True)
            r_c = clf_g.predict(x2_ca, raw_score=True)
            r_t = clf_g.predict(x2_te, raw_score=True)

        pl = fit_platt_calibrator(r_c, y_ca_s)
        p2_tr = np.clip(apply_platt(r_tr, pl), 1e-6, 1.0 - 1e-6)
        p2_c = np.clip(apply_platt(r_c, pl), 1e-6, 1.0 - 1e-6)
        p2_t = np.clip(apply_platt(r_t, pl), 1e-6, 1.0 - 1e-6)
        if pi_target_override is not None:
            pi_cs = float(np.mean(y_ca_s))
            p2_tr = adjust_probs_for_target_prevalence(p2_tr, pi_cs, pi_target_override)
            p2_c = adjust_probs_for_target_prevalence(p2_c, pi_cs, pi_target_override)
            p2_t = adjust_probs_for_target_prevalence(p2_t, pi_cs, pi_target_override)
        voi_t = fit_predict_prepurchase_voi(
            xb_tr_s, p1_oof_e, p2_tr, xb_ca_s, p1_ca_e, p2_c, xb_te_s, p1_te_e,
            contract, use_dispersion_2d=True, seed=seed_s,
        )
        q_t = voi_t > contract.per_instance_query_cost
        p_rt = np.where(q_t, p2_t, p1_te_e)
        return p_rt, q_t, p2_t, p2_c, beta_cz_val, kappa_val

    pr_gg0, q_gg0, _, _, _, _ = _fit_eval_split(
        tr0, y_tr0, p1_oof0_raw, p1_oof0_eval, cal0, y_cal0, p1_cal0_raw, p1_cal0_eval,
        xb_tr0, xb_ca0, xb_te0, p1_te0_raw, p1_te0_eval, disc_glob0.selected_ids, False, canonical_seed,
    )
    pr_gf0, q_gf0, _, _, bcz_gf0, _ = _fit_eval_split(
        tr0, y_tr0, p1_oof0_raw, p1_oof0_eval, cal0, y_cal0, p1_cal0_raw, p1_cal0_eval,
        xb_tr0, xb_ca0, xb_te0, p1_te0_raw, p1_te0_eval, disc_glob0.selected_ids, True, canonical_seed,
    )
    pr_up0, q_up0, p2_up0, _, bcz_up0, kap_up0 = _fit_eval_split(
        tr0, y_tr0, p1_oof0_raw, p1_oof0_eval, cal0, y_cal0, p1_cal0_raw, p1_cal0_eval,
        xb_tr0, xb_ca0, xb_te0, p1_te0_raw, p1_te0_eval, upca_sel0, True, canonical_seed,
    )

    canonical_new_boots = {
        "upca_vs_covert_afa": patient_cluster_bootstrap_diff(
            y_te, pr_up0, q_up0, pr_afa, q_afa, clusters_te, contract, sample_weight=w_te, n_boot=10000, seed=canonical_seed + 21
        ),
        "upca_vs_global_cmi_global_gbdt": patient_cluster_bootstrap_diff(
            y_te, pr_up0, q_up0, pr_gg0, q_gg0, clusters_te, contract, sample_weight=w_te, n_boot=10000, seed=canonical_seed + 22
        ),
    }

    # --- B. R = 10 Repeated Patient-Grouped Development Splits Estimator Triad (`Cross-Impact #1`) ---
    y_dev = dev[y_col].astype(int).to_numpy()
    cl_dev = dev[cluster_col].to_numpy()
    cal_frac_dev = float(len(cal0)) / float(len(dev))

    triad_arms = ["stage1", "global_cmi_gbdt", "global_cmi_fluid", "weighted_cz_fluid", "ipw_cz_fluid", "upca_fluid"]
    ens_p_sum = {a: np.zeros(n_te, dtype=float) for a in triad_arms}
    ens_q_sum = {a: np.zeros(n_te, dtype=float) for a in triad_arms}
    ens_p2_sum = {a: np.zeros(n_te, dtype=float) for a in triad_arms}
    split_u_sum = {a: np.zeros(n_te, dtype=float) for a in triad_arms}
    split_nbs: dict[str, list[float]] = {a: [] for a in triad_arms}
    cal_p1_pool: list[np.ndarray] = []
    cal_p2_pool: dict[str, list[np.ndarray]] = {a: [] for a in triad_arms if a != "stage1"}
    bcz_upca_splits: list[float] = []
    sel_counts_r10: dict[str, dict[str, int]] = {"weighted_cz": {}, "upca": {}, "global_cmi": {}}

    print(f"  [{task_id}] Running R={n_splits} patient-grouped dev splits -> untouched test (N_te={n_te:,})...", flush=True)
    for r_i in range(n_splits):
        seed_r = 2100 + 100 * (canonical_seed % 10) + r_i
        tr_idx, ca_idx = _patient_grouped_dev_split(cl_dev, y_dev, cal_frac=cal_frac_dev, seed=seed_r)
        tr_r = dev.iloc[tr_idx].reset_index(drop=True)
        ca_r = dev.iloc[ca_idx].reset_index(drop=True)
        y_tr_r = tr_r[y_col].astype(int).to_numpy()
        y_ca_r = ca_r[y_col].astype(int).to_numpy()

        p1_oof_r = compute_oof_stage1(tr_r[base_cols], y_tr_r, n_folds=3, seed=seed_r)
        p1_ca_r, p1_tes_r = fit_stage1_predictions(tr_r, y_tr_r, ca_r, y_ca_r, [te], base_cols, seed=seed_r)
        p1_te_r = p1_tes_r[0]

        if pi_target_override is not None:
            w_tr_r = case_control_prevalence_weights(y_tr_r, pi_target_override)
            p1_oof_e = adjust_probs_for_target_prevalence(p1_oof_r, float(np.mean(y_tr_r)), pi_target_override)
            p1_ca_e = adjust_probs_for_target_prevalence(p1_ca_r, float(np.mean(y_ca_r)), pi_target_override)
            p1_te_e = adjust_probs_for_target_prevalence(p1_te_r, float(np.mean(y_ca_r)), pi_target_override)
        else:
            w_tr_r = None
            p1_oof_e, p1_ca_e, p1_te_e = p1_oof_r, p1_ca_r, p1_te_r

        d_gl, _ = discover_stage_a_ablation(
            tr_r, y_tr_r, base_cols, ch_all, p1_oof_e, contract, budget,
            mode="global_cmi", pilot_fraction=0.20, target_pop_weights=w_tr_r, raw_oof_p_base_for_fit=p1_oof_r, seed=seed_r,
        )
        d_wc, _ = discover_stage_a_ablation(
            tr_r, y_tr_r, base_cols, ch_all, p1_oof_e, contract, budget,
            mode="weighted_cz_005", pilot_fraction=0.20, target_pop_weights=w_tr_r, raw_oof_p_base_for_fit=p1_oof_r, seed=seed_r,
        )
        d_ip, _ = discover_stage_a_ablation(
            tr_r, y_tr_r, base_cols, ch_all, p1_oof_e, contract, budget,
            mode="ipw_cz_wht", pilot_fraction=0.20, target_pop_weights=w_tr_r, raw_oof_p_base_for_fit=p1_oof_r, seed=seed_r,
        )
        d_oc, _ = discover_stage_a_ablation(
            tr_r, y_tr_r, base_cols, ch_all, p1_oof_e, contract, budget,
            mode="offset_cz_schervish", pilot_fraction=0.20, target_pop_weights=w_tr_r, raw_oof_p_base_for_fit=p1_oof_r, seed=seed_r,
        )
        sel_up_r, _ = oof_policy_screening_stage_b(
            d_oc.selected_ids, tr_r, y_tr_r, p1_oof_r, base_cols, ch_map, contract,
            pi_target_override=pi_target_override, seed=seed_r,
        )

        for cid in d_gl.selected_ids:
            sel_counts_r10["global_cmi"][cid] = sel_counts_r10["global_cmi"].get(cid, 0) + 1
        for cid in d_wc.selected_ids:
            sel_counts_r10["weighted_cz"][cid] = sel_counts_r10["weighted_cz"].get(cid, 0) + 1
        for cid in sel_up_r:
            sel_counts_r10["upca"][cid] = sel_counts_r10["upca"].get(cid, 0) + 1

        eb_r = FastFrameEncoder().fit(tr_r[base_cols])
        xb_tr_r, xb_ca_r, xb_te_r = eb_r.transform(tr_r[base_cols]), eb_r.transform(ca_r[base_cols]), eb_r.transform(te[base_cols])

        # Stage-1 accumulate
        q0 = np.zeros(n_te, dtype=bool)
        ens_p_sum["stage1"] += p1_te_e
        ens_p2_sum["stage1"] += p1_te_e
        u_s1 = unit_net_benefit_vec(y_te, p1_te_e, q0, contract)
        split_u_sum["stage1"] += u_s1
        split_nbs["stage1"].append(weighted_net_benefit_per_1k(y_te, p1_te_e, q0, contract, sample_weight=w_te))
        cal_p1_pool.append(p1_ca_e)

        split_cache: dict[tuple[tuple[str, ...], bool], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]] = {}
        for arm_k, sel_k, fl_k in [
            ("global_cmi_gbdt", d_gl.selected_ids, False),
            ("global_cmi_fluid", d_gl.selected_ids, True),
            ("weighted_cz_fluid", d_wc.selected_ids, True),
            ("ipw_cz_fluid", d_ip.selected_ids, True),
            ("upca_fluid", sel_up_r, True),
        ]:
            ckey = (tuple(sel_k), fl_k)
            if ckey not in split_cache:
                split_cache[ckey] = _fit_eval_split(
                    tr_r, y_tr_r, p1_oof_r, p1_oof_e, ca_r, y_ca_r, p1_ca_r, p1_ca_e,
                    xb_tr_r, xb_ca_r, xb_te_r, p1_te_r, p1_te_e, sel_k, fl_k, seed_r,
                )
            pr_k, q_k, p2_k, p2_ca_k, bcz_k, _ = split_cache[ckey]
            ens_p_sum[arm_k] += pr_k
            ens_q_sum[arm_k] += q_k.astype(float)
            ens_p2_sum[arm_k] += p2_k
            split_u_sum[arm_k] += unit_net_benefit_vec(y_te, pr_k, q_k, contract)
            split_nbs[arm_k].append(weighted_net_benefit_per_1k(y_te, pr_k, q_k, contract, sample_weight=w_te))
            cal_p2_pool[arm_k].append(p2_ca_k)
            if arm_k == "upca_fluid":
                bcz_upca_splits.append(bcz_k)

    # Aggregate R=10 Estimator Triad (`Bagged VOI`, `Sym. Ens.`, `Per-Split`)
    p1_bag_te = ens_p_sum["stage1"] / float(n_splits)
    p1_cal_cat = np.concatenate(cal_p1_pool)

    sym_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    bag_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {"stage1": (p1_bag_te, np.zeros(n_te, dtype=bool))}
    triad_summary: dict[str, Any] = {}

    for a in triad_arms:
        pa_sym = ens_p_sum[a] / float(n_splits)
        qa_sym = ens_q_sum[a] / float(n_splits)
        sym_preds[a] = (pa_sym, qa_sym)
        nb_sym = weighted_net_benefit_per_1k(y_te, pa_sym, qa_sym, contract, sample_weight=w_te)

        if a == "stage1":
            pa_bag, qa_bag = p1_bag_te, np.zeros(n_te, dtype=bool)
        else:
            p2_cal_cat = np.concatenate(cal_p2_pool[a])
            d_cal = logit(np.clip(p2_cal_cat, 1e-6, 1 - 1e-6)) - logit(np.clip(p1_cal_cat, 1e-6, 1 - 1e-6))
            if float(np.max(np.abs(d_cal))) >= 1e-9:
                edges_c = np.quantile(p1_cal_cat, np.linspace(0.0, 1.0, 13))
                edges_c[0], edges_c[-1] = 0.0, 1.0
                b_cal_c = np.clip(np.digitize(p1_cal_cat, edges_c[1:-1], right=False), 0, 11)
                qs_c = np.linspace(0.02, 0.98, 33)
                gq_c = np.quantile(d_cal, qs_c)
                bq_c = np.array([
                    np.quantile(d_cal[b_cal_c == b], qs_c) if int(np.sum(b_cal_c == b)) >= 15 else gq_c
                    for b in range(12)
                ])
                b_te_c = np.clip(np.digitize(p1_bag_te, edges_c[1:-1], right=False), 0, 11)
                voi_bag = empirical_martingale_voi(p1_bag_te, b_te_c, bq_c, contract)
                qa_bag = voi_bag > contract.per_instance_query_cost
            else:
                qa_bag = np.zeros(n_te, dtype=bool)
            p2_bag_te = ens_p2_sum[a] / float(n_splits)
            pa_bag = np.where(qa_bag, p2_bag_te, p1_bag_te)
            bag_preds[a] = (pa_bag, qa_bag)

        nb_bag = weighted_net_benefit_per_1k(y_te, pa_bag, qa_bag, contract, sample_weight=w_te)
        q_sym_rate = float(np.mean(qa_sym)) if w_te is None else float(np.sum(w_te * qa_sym) / np.sum(w_te))
        q_bag_rate = float(np.mean(qa_bag)) if w_te is None else float(np.sum(w_te * qa_bag) / np.sum(w_te))
        sp_arr = np.array(split_nbs[a], dtype=float)

        triad_summary[a] = {
            "bagged_voi_nb": round(nb_bag, 4),
            "bagged_voi_snb": round(standardized_net_benefit(nb_bag, eval_prev), 4),
            "bagged_voi_query_rate": round(q_bag_rate, 4),
            "sym_ens_nb": round(nb_sym, 4),
            "sym_ens_snb": round(standardized_net_benefit(nb_sym, eval_prev), 4),
            "sym_ens_query_rate": round(q_sym_rate, 4),
            "per_split_mean_nb": round(float(np.mean(sp_arr)), 4),
            "per_split_se_nb": round(float(np.std(sp_arr, ddof=1) / np.sqrt(len(sp_arr))), 4),
        }

    # 10,000-rep Patient-Cluster Bootstraps for BOTH Bagged VOI (Deployable Primary) & Symmetric Ensemble
    pb_up, qb_up = bag_preds["upca_fluid"]
    pb_wc, qb_wc = bag_preds["weighted_cz_fluid"]
    pb_gf, qb_gf = bag_preds["global_cmi_fluid"]
    pb_gg, qb_gg = bag_preds["global_cmi_gbdt"]
    pb_s1, qb_s1 = bag_preds["stage1"]

    ps_up, qs_up = sym_preds["upca_fluid"]
    ps_wc, qs_wc = sym_preds["weighted_cz_fluid"]
    ps_gf, qs_gf = sym_preds["global_cmi_fluid"]
    ps_gg, qs_gg = sym_preds["global_cmi_gbdt"]
    ps_s1, qs_s1 = sym_preds["stage1"]

    r10_boots = {
        "bagged_upca_vs_weighted_cz": patient_cluster_bootstrap_diff(
            y_te, pb_up, qb_up, pb_wc, qb_wc, clusters_te, contract, sample_weight=w_te, n_boot=10000, seed=31
        ),
        "bagged_upca_vs_global_cmi_fluid": patient_cluster_bootstrap_diff(
            y_te, pb_up, qb_up, pb_gf, qb_gf, clusters_te, contract, sample_weight=w_te, n_boot=10000, seed=32
        ),
        "bagged_upca_vs_global_cmi_gbdt": patient_cluster_bootstrap_diff(
            y_te, pb_up, qb_up, pb_gg, qb_gg, clusters_te, contract, sample_weight=w_te, n_boot=10000, seed=33
        ),
        "bagged_upca_vs_stage1": patient_cluster_bootstrap_diff(
            y_te, pb_up, qb_up, pb_s1, qb_s1, clusters_te, contract, sample_weight=w_te, n_boot=10000, seed=34
        ),
        "sym_ens_upca_vs_weighted_cz": patient_cluster_bootstrap_diff(
            y_te, ps_up, qs_up, ps_wc, qs_wc, clusters_te, contract, sample_weight=w_te, n_boot=10000, seed=35
        ),
        "sym_ens_upca_vs_global_cmi_fluid": patient_cluster_bootstrap_diff(
            y_te, ps_up, qs_up, ps_gf, qs_gf, clusters_te, contract, sample_weight=w_te, n_boot=10000, seed=36
        ),
        "sym_ens_upca_vs_global_cmi_gbdt": patient_cluster_bootstrap_diff(
            y_te, ps_up, qs_up, ps_gg, qs_gg, clusters_te, contract, sample_weight=w_te, n_boot=10000, seed=37
        ),
        "sym_ens_upca_vs_stage1": patient_cluster_bootstrap_diff(
            y_te, ps_up, qs_up, ps_s1, qs_s1, clusters_te, contract, sample_weight=w_te, n_boot=10000, seed=38
        ),
    }

    exchange_out: dict[str, Any] = {}
    if task_id == "mcmed_critical_icu_death":
        exchange_out = {
            "canonical_split_vs_stage1": _cluster_exchange_table(
                y_te, pr_up0, q_up0, p1_te0_eval, np.zeros(n_te, dtype=bool), clusters_te,
                [0.03, 0.05, 0.08, 0.10], contract.per_instance_query_cost,
            ),
            "r10_bagged_vs_weighted_cz": _cluster_exchange_table(
                y_te, pb_up, qb_up, pb_wc, qb_wc, clusters_te,
                [0.03, 0.05, 0.08, 0.10], contract.per_instance_query_cost,
            ),
            "r10_sym_ens_vs_weighted_cz": _cluster_exchange_table(
                y_te, ps_up, qs_up, ps_wc, qs_wc, clusters_te,
                [0.03, 0.05, 0.08, 0.10], contract.per_instance_query_cost,
            ),
        }

    # Diagnostic for BWH 10d Stent/CABG (`Action #5`) and Intercept Alignment (`Cross-Impact #e`)
    extra_diag: dict[str, Any] = {}
    if task_id == "bwh_stent_cabg_010_ecg_only":
        tau_s = contract.tau_star
        odds_s = tau_s / (1.0 - tau_s)
        gamma_s = contract.per_instance_query_cost / contract.cost_fn
        tp_up = 1000.0 * float(np.mean((pr_up0 >= tau_s) & (y_te == 1)))
        fp_up = 1000.0 * float(np.mean((pr_up0 >= tau_s) & (y_te == 0)))
        tp_gf = 1000.0 * float(np.mean((pr_gf0 >= tau_s) & (y_te == 1)))
        fp_gf = 1000.0 * float(np.mean((pr_gf0 >= tau_s) & (y_te == 0)))
        q_drag = -1000.0 * gamma_s * (float(np.mean(q_up0)) - float(np.mean(q_gf0)))
        fp_drag = -odds_s * (fp_up - fp_gf)
        tp_diff = tp_up - tp_gf
        near_m = (pr_up0 >= 0.015) & (pr_up0 <= 0.060)
        extra_diag["auprc_vs_nb_divergence_decomposition"] = {
            "global_cmi_selected": disc_glob0.selected_ids,
            "upca_selected": upca_sel0,
            "upca_tp_per_1k": round(tp_up, 3),
            "global_cmi_tp_per_1k": round(tp_gf, 3),
            "delta_tp_component_nb": round(tp_diff, 4),
            "upca_fp_per_1k": round(fp_up, 3),
            "global_cmi_fp_per_1k": round(fp_gf, 3),
            "delta_fp_per_1k": round(fp_up - fp_gf, 3),
            "delta_fp_component_nb": round(fp_drag, 4),
            "delta_query_rate": round(float(np.mean(q_up0)) - float(np.mean(q_gf0)), 4),
            "delta_query_fee_component_nb": round(q_drag, 4),
            "sum_components_delta_nb": round(tp_diff + fp_drag + q_drag, 4),
            "near_tau_band_015_060_mean_pred_upca": round(float(np.mean(pr_up0[near_m])), 5),
            "near_tau_band_015_060_obs_rate": round(float(np.mean(y_te[near_m])), 5),
        }

    if task_id in ("mcmed_bounce72_admit_2h", "ntuh_pre_arrest_sca_pi002"):
        mean_pred = float(np.mean(pr_up0)) if w_te is None else float(np.sum(w_te * pr_up0) / np.sum(w_te))
        p_aligned = adjust_probs_for_target_prevalence(pr_up0, mean_pred, eval_prev)
        cal_before = compute_calibration_metrics(y_te, pr_up0, sample_weight=w_te)
        cal_after = compute_calibration_metrics(y_te, p_aligned, sample_weight=w_te)
        nb_after = weighted_net_benefit_per_1k(y_te, p_aligned, q_up0, contract, sample_weight=w_te)
        extra_diag["label_free_marginal_intercept_alignment"] = {
            "delta_0": round(float(np.log(eval_prev / (1 - eval_prev)) - np.log(mean_pred / (1 - mean_pred))), 4),
            "citl_before": cal_before["cal_intercept"],
            "citl_after": cal_after["cal_intercept"],
            "slope_before": cal_before["cal_slope"],
            "slope_after": cal_after["cal_slope"],
            "ici_before": cal_before["ici"],
            "ici_after": cal_after["ici"],
            "nb_before": round(weighted_net_benefit_per_1k(y_te, pr_up0, q_up0, contract, sample_weight=w_te), 4),
            "nb_after": round(nb_after, 4),
        }

    bcz_arr = np.array(bcz_upca_splits, dtype=float)
    print(
        f"  [{task_id}] Done ({time.time()-t0:.1f}s) | Bagged UPCA={triad_summary['upca_fluid']['bagged_voi_nb']:+.2f} vs WCZ={triad_summary['weighted_cz_fluid']['bagged_voi_nb']:+.2f} "
        f"(d={r10_boots['bagged_upca_vs_weighted_cz']['diff_mean']:+.2f}, p={r10_boots['bagged_upca_vs_weighted_cz']['p_two_sided']:.4f}) | "
        f"Sym UPCA={triad_summary['upca_fluid']['sym_ens_nb']:+.2f} vs WCZ={triad_summary['weighted_cz_fluid']['sym_ens_nb']:+.2f} "
        f"(d={r10_boots['sym_ens_upca_vs_weighted_cz']['diff_mean']:+.2f}, p={r10_boots['sym_ens_upca_vs_weighted_cz']['p_two_sided']:.4f}) | "
        f"Covert={afa_arm['nb_routed']:+.2f} | beta_CZ={np.mean(bcz_arr):.2f}",
        flush=True,
    )
    return {
        "covert2023_greedy_cmi_afa": afa_arm,
        "canonical_new_bootstraps_10k": canonical_new_boots,
        "canonical_learned_beta_cz": {
            "upca_fluid_beta_cz": round(bcz_up0, 4),
            "upca_fluid_kappa": round(kap_up0, 3),
            "global_cmi_fluid_beta_cz": round(bcz_gf0, 4),
        },
        "r10_estimator_triad": {
            "n_splits": n_splits,
            "learned_beta_cz_mean": round(float(np.mean(bcz_arr)), 4),
            "learned_beta_cz_se": round(float(np.std(bcz_arr, ddof=1) / np.sqrt(len(bcz_arr))), 4),
            "selected_channel_frequencies_r10": sel_counts_r10,
            "arms": triad_summary,
            "bootstraps_10k_patient_cluster": r10_boots,
        },
        "exchange_rate_tables": exchange_out,
        "diagnostics": extra_diag,
    }


# ---------------------------------------------------------------------------
# Action #6: Stanford ICU Within-Stratum Pilot Independence Deep-Dive (`seed=44`)
# ---------------------------------------------------------------------------
def run_stanford_icu_independence_deep_dive(
    mc: pd.DataFrame,
    mc_base: list[str],
    mc_channels: list[ChannelDef],
) -> dict[str, Any]:
    print("\n=== [6/6] Stanford ICU Within-Stratum Pilot Independence Deep-Dive (`seed=44`) ===")
    tr = mc[mc["split"] == "train"].reset_index(drop=True)
    cal = mc[mc["split"] == "cal"].reset_index(drop=True)
    te = mc[mc["split"] == "test"].reset_index(drop=True)
    y_tr = tr["y_critical_icu_death"].astype(int).to_numpy()
    y_cal = cal["y_critical_icu_death"].astype(int).to_numpy()
    y_te = te["y_critical_icu_death"].astype(int).to_numpy()
    p1_oof = compute_oof_stage1(tr[mc_base], y_tr, n_folds=3, seed=44)

    cols = list(dict.fromkeys(c for ch in mc_channels for c in ch.columns))
    enc = FastFrameEncoder(include_missing_indicators=False).fit(tr[cols])
    z_mat = enc.transform(tr[cols]).astype(float)
    feat_names = list(cols) if z_mat.shape[1] == len(cols) else [f"col_{i}" for i in range(z_mat.shape[1])]

    pv = np.clip(p1_oof, 1e-12, 1.0 - 1e-12)
    quantiles = np.quantile(logit(pv), np.linspace(0.0, 1.0, 9))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    bins = np.clip(np.digitize(logit(pv), quantiles[1:-1], right=False), 0, 7)
    _, strata = np.unique(y_tr * 8 + bins, return_inverse=True)

    # Precompute within-stratum centered features and finite-population covariance once
    p_dim = z_mat.shape[1]
    zc_all = np.zeros_like(z_mat)
    cov_mat = np.zeros((p_dim, p_dim), dtype=float)
    pilot_44 = sample_pilot(y_tr, p1_oof, pilot_fraction=0.20, mode="collision_neyman", tau_star=0.05, n_bins=8, seed=44)
    in_44 = np.zeros(len(y_tr), dtype=float)
    in_44[pilot_44.indices] = 1.0

    for h in np.unique(strata):
        m = strata == h
        N_h = int(np.sum(m))
        n_h = int(np.sum(in_44[m]))
        if N_h <= 2 or n_h <= 0 or n_h >= N_h:
            continue
        zh = z_mat[m]
        zc = zh - np.mean(zh, axis=0, keepdims=True)
        zc_all[m] = zc
        s_cov = (zc.T @ zc) / float(N_h - 1)
        cov_mat += float(n_h) * (1.0 - float(n_h) / float(N_h)) * s_cov

    diag_v = np.diag(cov_mat)
    valid = diag_v > 1e-12
    valid_names = [feat_names[i] for i in np.flatnonzero(valid)]
    sd_vec = np.sqrt(diag_v[valid])

    score_44 = in_44 @ zc_all[:, valid]
    z_44 = score_44 / sd_vec
    order_top = np.argsort(np.abs(z_44))[::-1]
    top_features = [
        {
            "feature": valid_names[int(i)],
            "z_stat": round(float(z_44[int(i)]), 4),
            "p_unadjusted": round(float(2.0 * (1.0 - norm.cdf(abs(z_44[int(i)])))), 6),
        }
        for i in order_top[:5]
    ]

    # Scale-invariant correlation-standardized Mahalanobis statistic across all valid features (rank = 70 vs raw unscaled rank = 19)
    corr_mat = cov_mat[np.ix_(valid, valid)] / np.outer(sd_vec, sd_vec)
    eig_c, vec_c = np.linalg.eigh(corr_mat)
    keep_c = eig_c > 1e-5
    rank_corr = int(np.sum(keep_c))
    wald_corr_44 = float(np.sum(((vec_c[:, keep_c].T @ z_44) ** 2) / eig_c[keep_c]))
    wald_corr_p_asymp = float(1.0 - chi2.cdf(wald_corr_44, df=rank_corr))

    # Also raw unscaled rank-19 statistic from patch_r15_exact_indep.py for exact design-based permutation check across 500 random pilot seeds
    eig_u, vec_u = np.linalg.eigh(cov_mat[np.ix_(valid, valid)])
    keep_u = eig_u > (1e-6 * float(np.max(eig_u)))
    rank_unscaled = int(np.sum(keep_u))
    wald_u_44 = float(np.sum(((vec_u[:, keep_u].T @ score_44) ** 2) / eig_u[keep_u]))

    perm_wald_corr = []
    perm_wald_u = []
    perm_asymp_u_rejects = 0
    perm_asymp_c_rejects = 0
    for s_perm in range(500):
        pil_s = sample_pilot(y_tr, p1_oof, pilot_fraction=0.20, mode="collision_neyman", tau_star=0.05, n_bins=8, seed=9000 + s_perm)
        in_s = np.zeros(len(y_tr), dtype=float)
        in_s[pil_s.indices] = 1.0
        sc_s = in_s @ zc_all[:, valid]
        zs_s = sc_s / sd_vec
        wc_s = float(np.sum(((vec_c[:, keep_c].T @ zs_s) ** 2) / eig_c[keep_c]))
        wu_s = float(np.sum(((vec_u[:, keep_u].T @ sc_s) ** 2) / eig_u[keep_u]))
        perm_wald_corr.append(wc_s)
        perm_wald_u.append(wu_s)
        if float(1.0 - chi2.cdf(wu_s, df=rank_unscaled)) < 0.05:
            perm_asymp_u_rejects += 1
        if float(1.0 - chi2.cdf(wc_s, df=rank_corr)) < 0.05:
            perm_asymp_c_rejects += 1

    perm_p_corr = float((1 + sum(x >= wald_corr_44 for x in perm_wald_corr)) / 501.0)
    perm_p_unscaled = float((1 + sum(x >= wald_u_44 for x in perm_wald_u)) / 501.0)

    # Sensitivity: Exclude the top |z| feature on `seed=44` and re-run UPCA-Fluid on Stanford ICU!
    top_feat = top_features[0]["feature"]
    ch_dropped = []
    for ch in mc_channels:
        rem_c = tuple(c for c in ch.columns if c != top_feat)
        if rem_c:
            ch_dropped.append(ChannelDef(ch.channel_id, rem_c, ch.cost))

    ck_icu = ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.030)
    rng44 = np.random.default_rng(44)
    mc_d = mc.copy()
    mc_d["null_1a"], mc_d["null_1b"] = rng44.normal(size=len(mc_d)), rng44.normal(size=len(mc_d))
    mc_d["null_2a"], mc_d["null_2b"] = rng44.normal(size=len(mc_d)), rng44.normal(size=len(mc_d))
    ch_all_drop = ch_dropped + [
        ChannelDef("ch_null_noise_1", ("null_1a", "null_1b"), 0.50, is_null_stress=True),
        ChannelDef("ch_null_noise_2", ("null_2a", "null_2b"), 0.50, is_null_stress=True),
    ]
    ch_map_drop = {c.channel_id: c for c in ch_all_drop}
    tr_d = mc_d[mc_d["split"] == "train"].reset_index(drop=True)
    cal_d = mc_d[mc_d["split"] == "cal"].reset_index(drop=True)
    te_d = mc_d[mc_d["split"] == "test"].reset_index(drop=True)

    disc_drop, _ = discover_stage_a_ablation(
        tr_d, y_tr, mc_base, ch_all_drop, p1_oof, ck_icu, budget=3.0,
        mode="offset_cz_schervish", pilot_fraction=0.20, seed=44,
    )
    sel_drop, _ = oof_policy_screening_stage_b(disc_drop.selected_ids, tr_d, y_tr, p1_oof, mc_base, ch_map_drop, ck_icu, seed=44)
    p1_ca, p1_tes = fit_stage1_predictions(tr_d, y_tr, cal_d, y_cal, [te_d], mc_base, seed=44)
    p1_te = p1_tes[0]
    eb = FastFrameEncoder().fit(tr_d[mc_base])
    xb_tr, xb_ca, xb_te = eb.transform(tr_d[mc_base]), eb.transform(cal_d[mc_base]), eb.transform(te_d[mc_base])
    s2_c = [c for cid in sel_drop for c in ch_map_drop[cid].columns]
    ez = FastFrameEncoder().fit(tr_d[s2_c])
    z_tr, z_ca, z_te = ez.transform(tr_d[s2_c]), ez.transform(cal_d[s2_c]), ez.transform(te_d[s2_c])
    clf = FluidCollisionBackboneClassifier(tau_star=ck_icu.tau_star, kappa_max=20.0, min_cz_ess=240.0, seed=51).fit(
        np.hstack([xb_tr, z_tr]), z_tr, y_tr, logit(np.clip(p1_oof, 1e-5, 1 - 1e-5))
    )
    r_tr = clf.decision_function(np.hstack([xb_tr, z_tr]), z_tr, logit(np.clip(p1_oof, 1e-5, 1 - 1e-5)))
    r_ca = clf.decision_function(np.hstack([xb_ca, z_ca]), z_ca, logit(p1_ca))
    r_te = clf.decision_function(np.hstack([xb_te, z_te]), z_te, logit(p1_te))
    pl = fit_platt_calibrator(r_ca, y_cal)
    p2_tr, p2_ca, p2_te = np.clip(apply_platt(r_tr, pl), 1e-6, 1 - 1e-6), np.clip(apply_platt(r_ca, pl), 1e-6, 1 - 1e-6), np.clip(apply_platt(r_te, pl), 1e-6, 1 - 1e-6)
    voi_te = fit_predict_prepurchase_voi(xb_tr, p1_oof, p2_tr, xb_ca, p1_ca, p2_ca, xb_te, p1_te, ck_icu, seed=44)
    q_te = voi_te > ck_icu.per_instance_query_cost
    p_rt = np.where(q_te, p2_te, p1_te)
    nb_drop = weighted_net_benefit_per_1k(y_te, p_rt, q_te, ck_icu)

    out_ind = {
        "top_5_features_seed44": top_features,
        "raw_unscaled_rank19_wald_chi2": round(wald_u_44, 3),
        "raw_unscaled_rank19_df": rank_unscaled,
        "raw_unscaled_design_permutation_p_500": round(perm_p_unscaled, 4),
        "raw_unscaled_500seed_nominal_005_rejection_rate": round(perm_asymp_u_rejects / 500.0, 4),
        "standardized_fullrank_wald_chi2": round(wald_corr_44, 3),
        "standardized_fullrank_df": rank_corr,
        "standardized_fullrank_asymp_p": round(wald_corr_p_asymp, 4),
        "standardized_fullrank_design_permutation_p_500": round(perm_p_corr, 4),
        "standardized_fullrank_500seed_nominal_005_rejection_rate": round(perm_asymp_c_rejects / 500.0, 4),
        "exclude_top_outlier_feature_sensitivity": {
            "excluded_feature": top_feat,
            "selected_channels_after_exclusion": sel_drop,
            "nb_routed_after_exclusion": round(nb_drop, 4),
            "auprc_full_after_exclusion": round(float(average_precision_score(y_te, p2_te)), 4),
        },
    }
    print("  [Independence Deep-Dive]:", json.dumps(out_ind, indent=2), flush=True)
    return out_ind


def main() -> None:
    t_start = time.time()
    res = json.loads(RES_JSON.read_text())

    # Load Stanford MC-MED
    mc = pd.read_parquet(DATA_DIR / "mcmed_stanford_2h_v2.parquet")
    mc["gender_male"] = mc["Gender"].astype(str).str.upper().str.startswith("M").astype(int)
    top_cc = mc.loc[mc["split"] == "train", "CC"].value_counts().head(25).index.tolist()
    mc["cc_clean"] = mc["CC"].where(mc["CC"].isin(top_cc), "OTHER_CC").astype(str)
    mc["arrival_clean"] = mc["Means_of_arrival"].fillna("Missing").astype(str)
    mc_base = ["Age", "gender_male", "Triage_Temp", "Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP", "Triage_DBP", "esi_num", "cc_clean", "arrival_clean"]
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

    # 1. 500-Pilot Discovery Suite
    eff_500 = run_500_pilot_discovery_suite(mc[mc["split"] == "train"].reset_index(drop=True), mc_base, mc_channels)
    res["clinical"]["two_phase_efficiency_r15"] = eff_500
    RES_JSON.write_text(json.dumps(res, indent=2))

    # 2. Load BWH & NTUH
    bwh = pd.read_parquet(DATA_DIR / "bwh_ed_ecg_v2.parquet")
    bwh_base = [c for c in ["age_at_admit", "sex_male", "vent_rate", "pr_interval", "qrs_duration", "qt_interval", "qtc_interval", "p_axes", "r_axes", "t_axes"] if c in bwh.columns]
    bwh_channels = [
        ChannelDef("ch_ecg_ischemia_flags", tuple(c for c in ["has_st_eleva", "has_depress", "has_twave_inver", "has_twave_abnormal", "has_jpoint_eleva", "has_jpoint_repol", "has_st", "has_twave"] if c in bwh.columns), 0.40),
        ChannelDef("ch_ecg_rhythm_struct", tuple(c for c in ["has_afib", "has_bbb", "has_aberran_bbb", "has_lvh", "has_pacemaker", "has_prolonged_qt", "has_rhythm_disturbance", "has_normal_ecg", "has_normal_sinus"] if c in bwh.columns), 0.40),
        ChannelDef("ch_ecg_waveform", tuple(c for c in bwh.columns if c.startswith("wf_")), 0.85),
    ]

    ntuh_cc = pd.read_parquet(DATA_DIR / "ntuh_pre_arrest_case_control_v2.parquet")
    ntuh_base = [c for c in ["age", "sex_male", "ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration", "qt_interval", "qt_corrected", "p_axis", "r_axis", "t_axis", "qtc_frederica"] if c in ntuh_cc.columns]
    ntuh_channels = [
        ChannelDef("ch_ant_v1v4", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_ant")), 0.55),
        ChannelDef("ch_lat_i_avl_v5v6", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_lat")), 0.55),
        ChannelDef("ch_inf_ii_iii_avf", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_inf")), 0.55),
        ChannelDef("ch_avr", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_avr")), 0.45),
        ChannelDef("ch_ecg_diag_stmts", tuple(c for c in ntuh_cc.columns if c.startswith("dx_")), 0.40),
        ChannelDef("ch_waveform_12lead_median", tuple(c for c in ntuh_cc.columns if c.startswith("medwf_")), 0.85),
    ]

    mc_disch = mc[mc["is_discharged"] == 1].copy().reset_index(drop=True)

    task_specs = [
        ("mcmed_critical_icu_death", mc, "y_critical_icu_death", "MRN", mc_base, mc_channels, ClinicalContract(1.0, 19.0, 0.25, 0.030), 3.00, None, 44),
        ("mcmed_hospital_admission", mc, "y_admit", "MRN", mc_base, mc_channels, ClinicalContract(1.0, 5.67, 0.50, 0.025), 3.00, None, 45),
        ("mcmed_bounce72_admit_2h", mc_disch, "y_bounce72_admit", "MRN", mc_base, mc_channels, ClinicalContract(1.0, 32.33, 0.15, 0.025), 3.00, None, 46),
        ("bwh_death_030_ecg_only", bwh, "death_030_day", "patient_ngsci_id", bwh_base, bwh_channels, ClinicalContract(1.0, 32.33, 0.20, 0.020), 1.80, None, 42),
        ("bwh_stent_cabg_010_ecg_only", bwh, "stent_or_cabg_010_day", "patient_ngsci_id", bwh_base, bwh_channels, ClinicalContract(1.0, 32.33, 0.20, 0.020), 1.80, None, 43),
        ("ntuh_pre_arrest_sca_pi002", ntuh_cc, "y_pre_arrest", "patient_ngsci_id", ntuh_base, ntuh_channels, ClinicalContract(1.0, 19.0, 0.25, 0.020), 1.80, 0.02, 47),
    ]

    print("\n=== [2-5/6] Running R=10 Estimator Triad, Covert-AFA, CMI+GBDT Bootstraps & Diagnostics Across All 6 Tasks ===")
    v2_1_tasks: dict[str, Any] = {}
    for tid, df_t, yc, clc, bc, chs, ck, bgt, pi_ov, sd in task_specs:
        tout = run_r10_triad_and_task_upgrades(
            tid, df_t, yc, clc, bc, chs, ck, bgt, pi_target_override=pi_ov, canonical_seed=sd, n_splits=10
        )
        v2_1_tasks[tid] = tout
        res["clinical"]["primary_tasks"][tid]["arms"]["covert2023_greedy_cmi_afa"] = tout["covert2023_greedy_cmi_afa"]
        res["clinical"]["primary_tasks"][tid]["bootstraps_10k_patient_cluster"].update(tout["canonical_new_bootstraps_10k"])
        res["clinical"]["primary_tasks"][tid]["learned_beta_cz"] = tout["canonical_learned_beta_cz"]
        res["clinical"]["primary_tasks"][tid]["r10_estimator_triad"] = tout["r10_estimator_triad"]
        if tout["exchange_rate_tables"]:
            res["clinical"]["primary_tasks"][tid]["exchange_rate_tables"] = tout["exchange_rate_tables"]
        if tout["diagnostics"]:
            res["clinical"]["primary_tasks"][tid]["v2_1_diagnostics"] = tout["diagnostics"]
        RES_JSON.write_text(json.dumps(res, indent=2))

    # Also compute CMI+GBDT bootstrap CIs on `[1]ROSC` supplementary tasks (`Action #3`)
    ntuh_rosc = pd.read_parquet(DATA_DIR / "ntuh_post_rosc_v2.parquet")
    rosc_base = [c for c in ["age", "sex_male", "witnessed_collapse", "initial_rhythm", "location_of_cardiac_arrest", "cpr_duration", "epinerphine_dose", "dc_shock"] if c in ntuh_rosc.columns]
    rosc_channels = [
        ChannelDef("ch_rosc_comorbidities", tuple(c for c in ["dm", "htn", "cad", "hf", "arrhythmia", "copd_asthma", "esrd", "cva"] if c in ntuh_rosc.columns), 0.50),
        ChannelDef("ch_rosc_ecg_intervals", tuple(c for c in ["ventricular_rate", "pr_interval", "qrs_duration", "qt_interval", "qt_corrected", "qtc_frederica", "p_axis", "r_axis", "t_axis"] if c in ntuh_rosc.columns), 0.50),
        ChannelDef("ch_rosc_waveform_12lead", tuple(c for c in ntuh_rosc.columns if c.startswith("medwf_")), 0.70),
    ]
    for rtid, rycol, rck, rsd in [
        ("ntuh_post_rosc_survival_supplement", "y_survive", ClinicalContract(1.0, 4.0, 0.40, 0.020), 48),
        ("ntuh_post_rosc_cpc12_supplement", "y_good_neuro_cpc12", ClinicalContract(1.0, 5.67, 0.35, 0.020), 49),
    ]:
        rout = run_r10_triad_and_task_upgrades(
            rtid, ntuh_rosc, rycol, "patient_ngsci_id", rosc_base, rosc_channels, rck, 1.20, canonical_seed=rsd, n_splits=5
        )
        res["clinical"]["secondary_and_sensitivity_tasks"][rtid]["bootstraps_10k_patient_cluster"].update(rout["canonical_new_bootstraps_10k"])
        res["clinical"]["secondary_and_sensitivity_tasks"][rtid]["learned_beta_cz"] = rout["canonical_learned_beta_cz"]

    # 6. Stanford ICU Independence Deep-Dive
    ind_deep = run_stanford_icu_independence_deep_dive(mc, mc_base, mc_channels)
    res["clinical"]["stanford_icu_independence_deep_dive_r15"] = ind_deep
    res["schema_version"] = "paper3_v2_1_triad_500pilots_covertafa"
    res["companion_provenance_commit"] = "8c8c4c8 (2026-11-22T18:16:58-05:00)"
    res["v2_1_upgrade_elapsed_sec"] = round(time.time() - t_start, 1)

    RES_JSON.write_text(json.dumps(res, indent=2))
    print(f"\n=== ALL PAPER 3 v2.1 UPGRADES COMPLETED IN {time.time()-t_start:.1f}s -> {RES_JSON} ===")


if __name__ == "__main__":
    main()

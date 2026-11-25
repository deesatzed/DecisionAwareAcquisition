"""Core leak-free statistical, calibration, decision-curve, and patient-cluster inference primitives for Paper 3 v2.

Implements all mathematical remediations (`R1` through `R24`):
  - Vectorized `10,000`-replicate patient-cluster ratio-estimator bootstrap with exact Davison-Hinkley `(k+1)/(B+1)` p-values.
  - Directional Patient-Cluster Skipped-Case Action-Disagreement Upper Bound (`Neg->Pos`, `Pos->Neg`, beneficial vs harmful
    flips, skipped events, skipped sub-tau* events, and Risk-Difference Bound).
  - Flexible spline calibration curve, logistic calibration intercept & slope, and Integrated Calibration Index (`ICI`).
  - Standardized Net Benefit (`sNB`), Vickers Decision Curves `NB(tau)` with Treat-All / Treat-None, and case-control
    prior-odds prevalence adjustment (Prentice-Pyke / Bayes shift + inverse-sampling-fraction target population weights).
  - Unconfounded Stage-A discovery comparators (`Global-CMI`, `Weighted-Pilot CZ [0.05 floor]`, `IPW-Pilot CZ [w_HT]`,
    `Offset-Pilot LogLoss`, `Offset-Pilot CZ [Breslow-Cain]`) + Stage-B OOF Policy Screening (`UPCA-Fluid`), plus
    empirical within-stratum independence check `I _|_ Z | (Y, b)` and Tao-Zeng-Lin / Breslow-Cain two-phase efficiency test.
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "collision_zone_ed" / "src"))

from collision_zone.three_predicates import (
    ChannelDef,
    ClinicalContract,
    DiscoveryOutcome,
    DiscoveryStepLog,
    FastFrameEncoder,
    FluidCollisionBackboneClassifier,
    PilotSample,
    apply_platt,
    bernoulli_kl_from_tau,
    collision_zone_relevance_weights,
    empirical_martingale_voi,
    fit_platt_calibrator,
    holm_adjust,
    log_likelihood_score,
    sample_pilot,
    schervish_collision_proper_score,
    solve_ess_kappa,
    two_phase_ht_mean_and_se,
)


# ---------------------------------------------------------------------------
# 1. Case-Control Target-Prevalence Adjustment (NTUH R17)
# ---------------------------------------------------------------------------


def case_control_prevalence_weights(y: np.ndarray, pi_target: float) -> np.ndarray:
    """Compute normalized inverse-sampling-fraction weights mapping a case-control sample to target prevalence pi_target."""
    yv = np.asarray(y, dtype=int)
    n = len(yv)
    n1 = max(1, int(np.sum(yv == 1)))
    n0 = max(1, int(np.sum(yv == 0)))
    y_bar = n1 / float(n)
    w = np.where(yv == 1, pi_target / y_bar, (1.0 - pi_target) / (1.0 - y_bar))
    return w * (n / float(np.sum(w)))


def adjust_probs_for_target_prevalence(
    p_cc: np.ndarray,
    pi_sample: float,
    pi_target: float,
) -> np.ndarray:
    """Exact Bayes prior-odds log-odds shift from sample case fraction pi_sample to clinical prevalence pi_target."""
    ps = float(np.clip(pi_sample, 1e-6, 1.0 - 1e-6))
    pt = float(np.clip(pi_target, 1e-6, 1.0 - 1e-6))
    shift = np.log(pt / (1.0 - pt)) - np.log(ps / (1.0 - ps))
    ell = logit(np.clip(np.asarray(p_cc, dtype=float), 1e-6, 1.0 - 1e-6))
    return np.clip(expit(np.clip(ell + shift, -25.0, 25.0)), 1e-6, 1.0 - 1e-6)


# ---------------------------------------------------------------------------
# 2. Net Benefit, Standardized NB, and 10,000-Rep Patient-Cluster Bootstrap
# ---------------------------------------------------------------------------


def unit_net_benefit_vec(
    y: np.ndarray,
    p_route: np.ndarray,
    queried: np.ndarray,
    contract: ClinicalContract,
) -> np.ndarray:
    """Per-encounter Net Benefit contribution scaled per 1,000 arrivals."""
    yv = np.asarray(y, dtype=float)
    pv = np.asarray(p_route, dtype=float)
    qv = np.asarray(queried, dtype=float)
    act = (pv >= contract.tau_star).astype(float)
    tp = act * yv
    fp = act * (1.0 - yv)
    return 1000.0 * (
        tp
        - (contract.cost_fp / contract.cost_fn) * fp
        - (contract.per_instance_query_cost / contract.cost_fn) * qv
    )


def weighted_net_benefit_per_1k(
    y: np.ndarray,
    p_route: np.ndarray,
    queried: np.ndarray,
    contract: ClinicalContract,
    sample_weight: np.ndarray | None = None,
) -> float:
    u = unit_net_benefit_vec(y, p_route, queried, contract)
    if sample_weight is None:
        return float(np.mean(u))
    w = np.asarray(sample_weight, dtype=float)
    return float(np.sum(w * u) / max(1e-12, float(np.sum(w))))


def standardized_net_benefit(nb_per_1k: float, prevalence: float) -> float:
    """Prevalence-normalized Standardized Net Benefit sNB = NB / (1000 * pi)."""
    if prevalence <= 1e-9:
        return 0.0
    return float(nb_per_1k / (1000.0 * prevalence))


def patient_cluster_bootstrap_diff(
    y: np.ndarray,
    p_arm: np.ndarray,
    q_arm: np.ndarray,
    p_ref: np.ndarray,
    q_ref: np.ndarray,
    cluster_ids: np.ndarray,
    contract: ClinicalContract,
    *,
    sample_weight: np.ndarray | None = None,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Vectorized 10,000-replicate patient-cluster ratio-estimator bootstrap for NB(arm) - NB(ref)
    with exact Davison-Hinkley (1997) finite-sample two-sided p-value `(2 * (min(k+, k-) + 1)) / (B + 1)`.
    """
    u_arm = unit_net_benefit_vec(y, p_arm, q_arm, contract)
    u_ref = unit_net_benefit_vec(y, p_ref, q_ref, contract)
    diff_i = u_arm - u_ref
    w_i = np.ones_like(diff_i, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)

    obs_diff = float(np.sum(w_i * diff_i) / max(1e-12, float(np.sum(w_i))))
    if float(np.max(np.abs(diff_i))) <= 1e-15:
        return {
            "diff_mean": 0.0,
            "boot_se": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "p_two_sided": 1.0,
            "n_clusters": int(len(np.unique(cluster_ids))),
            "n_boot": int(n_boot),
            "exact_tie": 1.0,
        }

    _, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    n_clusters = int(inv.max()) + 1
    cluster_num = np.bincount(inv, weights=w_i * diff_i, minlength=n_clusters)
    cluster_den = np.bincount(inv, weights=w_i, minlength=n_clusters)

    rng = np.random.default_rng(seed)
    probs = np.full(n_clusters, 1.0 / float(n_clusters), dtype=float)
    boot_vals = np.empty(n_boot, dtype=float)
    chunk = 2500
    for start in range(0, n_boot, chunk):
        end = min(n_boot, start + chunk)
        counts = rng.multinomial(n_clusters, probs, size=end - start).astype(float)
        nums = counts @ cluster_num
        dens = np.maximum(1e-12, counts @ cluster_den)
        boot_vals[start:end] = nums / dens

    se = float(np.std(boot_vals, ddof=1))
    ci_low, ci_high = np.quantile(boot_vals, [0.025, 0.975])
    k_le = int(np.sum(boot_vals <= 0.0))
    k_ge = int(np.sum(boot_vals >= 0.0))
    p_two = min(1.0, 2.0 * float(min(k_le, k_ge) + 1) / float(n_boot + 1))

    return {
        "diff_mean": round(obs_diff, 4),
        "boot_se": round(se, 4),
        "ci95_low": round(float(ci_low), 4),
        "ci95_high": round(float(ci_high), 4),
        "p_two_sided": round(float(p_two), 6),
        "n_clusters": n_clusters,
        "n_boot": int(n_boot),
        "exact_tie": 0.0,
    }


def patient_cluster_bootstrap_scalar(
    values_i: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    n_boot: int = 2_000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Return (point_est, se, ci95_low, ci95_high) for a weighted encounter-level mean via patient-cluster bootstrap."""
    v = np.asarray(values_i, dtype=float)
    w = np.ones_like(v, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    pt = float(np.sum(w * v) / max(1e-12, float(np.sum(w))))
    _, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    n_c = int(inv.max()) + 1
    if n_c < 2:
        return pt, 0.0, pt, pt
    c_num = np.bincount(inv, weights=w * v, minlength=n_c)
    c_den = np.bincount(inv, weights=w, minlength=n_c)
    rng = np.random.default_rng(seed)
    probs = np.full(n_c, 1.0 / float(n_c))
    counts = rng.multinomial(n_c, probs, size=n_boot).astype(float)
    b = (counts @ c_num) / np.maximum(1e-12, counts @ c_den)
    lo, hi = np.quantile(b, [0.025, 0.975])
    return pt, float(np.std(b, ddof=1)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# 3. Directional Skipped-Case Action-Disagreement & Risk-Difference Bound (R5, R22)
# ---------------------------------------------------------------------------


def clopper_pearson_upper_conservative(k: int, n: int, alpha: float = 0.05) -> float:
    """Conservative (>= 1 - alpha) one-sided Clopper-Pearson Binomial upper bound."""
    if n <= 0 or k >= n:
        return 1.0
    if k <= 0:
        return float(1.0 - (alpha ** (1.0 / float(n))))
    return float(beta_dist.ppf(1.0 - alpha, k + 1, n - k))


def skipped_case_disagreement_audit(
    y_true: np.ndarray,
    p_stage1: np.ndarray,
    p_stage2_full: np.ndarray,
    queried_mask: np.ndarray,
    cluster_ids: np.ndarray,
    contract: ClinicalContract,
    *,
    sample_weight: np.ndarray | None = None,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Comprehensive directional Skipped-Case Action-Disagreement Upper Bound & Risk-Difference Bound (`R5, R22`).

    Decomposes skipped encounters (Q=0) into:
      - Neg -> Pos disagreements (p1 < tau*, p2 >= tau*): missed escalations (further split by Y=1 vs Y=0)
      - Pos -> Neg disagreements (p1 >= tau*, p2 < tau*): missed de-escalations (further split by Y=0 vs Y=1)
      - Events among skipped encounters & events among skipped encounters with p1 < tau*
      - Beneficial skips (Stage-1 action matched Bayes truth while Stage-2 flipped away) vs Harmful skips
      - Both conservative encounter Clopper-Pearson 95% UB and patient-cluster-robust 95% UB
      - Theoretical Risk-Difference Bound |R_route - R_full| <= max(C_FN, C_FP) * P(Q=0) * p_flip (excluding query cost).
    """
    yv = np.asarray(y_true, dtype=int)
    p1 = np.asarray(p_stage1, dtype=float)
    p2 = np.asarray(p_stage2_full, dtype=float)
    qv = np.asarray(queried_mask, dtype=bool)
    w_all = np.ones(len(yv), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    tau = contract.tau_star

    skip = ~qv
    n_tot = len(yv)
    n_skip = int(np.sum(skip))
    p_skip_weighted = float(np.sum(w_all[skip]) / max(1e-12, float(np.sum(w_all))))

    if n_skip == 0:
        return {
            "n_total": n_tot,
            "n_skipped": 0,
            "n_skipped_patients": 0,
            "skip_rate": 0.0,
            "flip_count": 0,
            "flip_rate": 0.0,
            "flip_ub95_clopper_pearson": 0.0,
            "flip_ub95_patient_cluster": 0.0,
            "neg_to_pos_count": 0,
            "neg_to_pos_rate": 0.0,
            "neg_to_pos_ub95_cluster": 0.0,
            "pos_to_neg_count": 0,
            "pos_to_neg_rate": 0.0,
            "pos_to_neg_ub95_cluster": 0.0,
            "skipped_events_total": 0,
            "skipped_events_below_tau": 0,
            "harmful_flip_count": 0,
            "beneficial_flip_count": 0,
            "missed_event_escalation_count": 0,
            "missed_event_escalation_ub95_cluster": 0.0,
            "realized_abs_decision_nb_diff_per_1k": 0.0,
            "theoretical_nb_diff_bound_ub95_per_1k": 0.0,
        }

    ys = yv[skip]
    ws = w_all[skip]
    cs = np.asarray(cluster_ids)[skip]
    a1 = p1[skip] >= tau
    a2 = p2[skip] >= tau

    flips = (a1 != a2).astype(float)
    neg_to_pos = ((~a1) & a2).astype(float)
    pos_to_neg = (a1 & (~a2)).astype(float)

    # Harmful flip (skipping caused worse action than p2): (Y=1, a1=0, a2=1) OR (Y=0, a1=1, a2=0)
    harmful = (((ys == 1) & (~a1) & a2) | ((ys == 0) & a1 & (~a2))).astype(float)
    # Beneficial flip (skipping kept correct action while p2 would have erred): (Y=0, a1=0, a2=1) OR (Y=1, a1=1, a2=0)
    beneficial = (((ys == 0) & (~a1) & a2) | ((ys == 1) & a1 & (~a2))).astype(float)
    missed_tp = ((ys == 1) & (~a1) & a2).astype(float)

    _, inv_s = np.unique(cs, return_inverse=True)
    n_skip_pats = int(inv_s.max()) + 1
    c_den = np.bincount(inv_s, weights=ws, minlength=n_skip_pats)
    rng = np.random.default_rng(seed)
    probs = np.full(n_skip_pats, 1.0 / float(n_skip_pats))
    counts = rng.multinomial(n_skip_pats, probs, size=n_boot).astype(float)
    boot_den = np.maximum(1e-12, counts @ c_den)

    def _cluster_ub95(ind_vec: np.ndarray) -> tuple[float, float]:
        rate_w = float(np.sum(ws * ind_vec) / max(1e-12, float(np.sum(ws))))
        k_raw = int(np.sum(ind_vec > 0.5))
        # Conservative cluster-level floor when k_raw == 0: 1 - alpha^(1 / G_skip)
        cp_cluster_zero = clopper_pearson_upper_conservative(0, n_skip_pats, alpha=alpha)
        if k_raw == 0:
            return 0.0, cp_cluster_zero
        c_num = np.bincount(inv_s, weights=ws * ind_vec, minlength=n_skip_pats)
        b_rates = (counts @ c_num) / boot_den
        ub_boot = float(np.quantile(b_rates, 1.0 - alpha))
        # Take max with conservative encounter Clopper-Pearson so small-k discreteness is never underestimated
        ub_cp = clopper_pearson_upper_conservative(k_raw, min(n_skip, n_skip_pats), alpha=alpha)
        return rate_w, max(ub_boot, ub_cp if k_raw <= 3 else ub_boot)

    flip_rate_w, flip_ub_cluster = _cluster_ub95(flips)
    n2p_rate_w, n2p_ub_cluster = _cluster_ub95(neg_to_pos)
    p2n_rate_w, p2n_ub_cluster = _cluster_ub95(pos_to_neg)
    mtp_rate_w, mtp_ub_cluster = _cluster_ub95(missed_tp)

    # Decision-only risk difference (c_query = 0): |NB_dec(route) - NB_dec(full)| per 1,000 arrivals
    c_zero = ClinicalContract(
        cost_fp=contract.cost_fp,
        cost_fn=contract.cost_fn,
        target_precision=contract.target_precision,
        per_instance_query_cost=0.0,
    )
    p_route = np.where(qv, p2, p1)
    nb_dec_route = weighted_net_benefit_per_1k(yv, p_route, np.zeros_like(qv), c_zero, sample_weight=w_all)
    nb_dec_full = weighted_net_benefit_per_1k(yv, p2, np.zeros_like(qv), c_zero, sample_weight=w_all)
    realized_abs_nb_diff = abs(nb_dec_route - nb_dec_full)

    # Exact bound in per-1,000 NB scale: max(1, C_FP / C_FN) * 1000 * P(Q=0) * p_flip
    max_nb_swing = 1000.0 * max(1.0, contract.cost_fp / contract.cost_fn)
    theo_bound_emp = max_nb_swing * p_skip_weighted * flip_rate_w
    theo_bound_ub95 = max_nb_swing * p_skip_weighted * flip_ub_cluster

    # Also raw Bayes risk scale: max(C_FN, C_FP) * P(Q=0) * p_flip
    max_cost = max(contract.cost_fn, contract.cost_fp)
    risk_diff_ub95 = max_cost * p_skip_weighted * flip_ub_cluster

    return {
        "n_total": n_tot,
        "n_skipped": n_skip,
        "n_skipped_patients": n_skip_pats,
        "skip_rate": round(p_skip_weighted, 4),
        "flip_count": int(np.sum(flips)),
        "flip_rate": round(flip_rate_w, 5),
        "flip_ub95_clopper_pearson": round(clopper_pearson_upper_conservative(int(np.sum(flips)), n_skip, alpha=alpha), 5),
        "flip_ub95_patient_cluster": round(flip_ub_cluster, 5),
        "neg_to_pos_count": int(np.sum(neg_to_pos)),
        "neg_to_pos_rate": round(n2p_rate_w, 5),
        "neg_to_pos_ub95_cluster": round(n2p_ub_cluster, 5),
        "pos_to_neg_count": int(np.sum(pos_to_neg)),
        "pos_to_neg_rate": round(p2n_rate_w, 5),
        "pos_to_neg_ub95_cluster": round(p2n_ub_cluster, 5),
        "skipped_events_total": int(np.sum(ys == 1)),
        "skipped_events_below_tau": int(np.sum((ys == 1) & (~a1))),
        "skipped_events_above_tau": int(np.sum((ys == 1) & a1)),
        "harmful_flip_count": int(np.sum(harmful)),
        "beneficial_flip_count": int(np.sum(beneficial)),
        "missed_event_escalation_count": int(np.sum(missed_tp)),
        "missed_event_escalation_rate": round(mtp_rate_w, 5),
        "missed_event_escalation_ub95_cluster": round(mtp_ub_cluster, 5),
        "realized_abs_decision_nb_diff_per_1k": round(realized_abs_nb_diff, 4),
        "theoretical_nb_diff_bound_emp_per_1k": round(theo_bound_emp, 4),
        "theoretical_nb_diff_bound_ub95_per_1k": round(theo_bound_ub95, 4),
        "risk_diff_bound_ub95_raw_cost": round(risk_diff_ub95, 5),
    }


# ---------------------------------------------------------------------------
# 4. Flexible Calibration Curve, Intercept, Slope & ICI (R3) + Decision Curves (R4)
# ---------------------------------------------------------------------------


def compute_calibration_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute logistic calibration intercept (calibration-in-the-large), calibration slope,
    spline-smoothed Integrated Calibration Index (`ICI`, `E50`, `E90`), and quantile calibration curve points.
    """
    yv = np.asarray(y_true, dtype=int)
    pv = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1.0 - 1e-6)
    wv = np.ones(len(yv), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    ell = logit(pv)

    # 1. Calibration slope & intercept: logit P(Y=1) = a + b * logit(p)
    lr_slope = LogisticRegression(C=1e5, solver="lbfgs", max_iter=500)
    lr_slope.fit(ell.reshape(-1, 1), yv, sample_weight=wv)
    cal_slope = float(lr_slope.coef_[0, 0])

    # Calibration-in-the-large (intercept with fixed slope = 1.0 via Newton-Raphson on weighted Bernoulli log-likelihood)
    a_citl = 0.0
    for _ in range(25):
        q = expit(np.clip(ell + a_citl, -25.0, 25.0))
        grad = float(np.sum(wv * (yv - q)))
        hess = float(np.sum(wv * q * (1.0 - q))) + 1e-9
        step = np.clip(grad / hess, -2.0, 2.0)
        a_citl += step
        if abs(step) < 1e-8:
            break

    # 2. Flexible B-spline logistic calibration curve for ICI (Austin & Steyerberg 2019)
    n_events = int(np.sum(yv == 1))
    n_knots = 5 if n_events >= 120 else (4 if n_events >= 50 else 3)
    try:
        sp = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
        x_sp = sp.fit_transform(ell.reshape(-1, 1))
        lr_sp = LogisticRegression(C=2.0, solver="lbfgs", max_iter=500)
        lr_sp.fit(x_sp, yv, sample_weight=wv)
        p_cal_smooth = lr_sp.predict_proba(x_sp)[:, 1]
    except Exception:
        p_cal_smooth = expit(np.clip(lr_slope.intercept_[0] + cal_slope * ell, -25.0, 25.0))

    abs_err = np.abs(p_cal_smooth - pv)
    ici = float(np.sum(wv * abs_err) / max(1e-12, float(np.sum(wv))))
    order = np.argsort(abs_err)
    cw = np.cumsum(wv[order]) / max(1e-12, float(np.sum(wv)))
    e50 = float(abs_err[order[min(len(order) - 1, int(np.searchsorted(cw, 0.50)))]])
    e90 = float(abs_err[order[min(len(order) - 1, int(np.searchsorted(cw, 0.90)))]])

    # Smooth curve on grid + quantile bin points for plotting
    grid_p = np.unique(np.quantile(pv, np.linspace(0.01, 0.99, 60)))
    try:
        grid_obs = lr_sp.predict_proba(sp.transform(logit(grid_p).reshape(-1, 1)))[:, 1]
    except Exception:
        grid_obs = expit(np.clip(lr_slope.intercept_[0] + cal_slope * logit(grid_p), -25.0, 25.0))

    q_edges = np.quantile(pv, np.linspace(0.0, 1.0, n_bins + 1))
    q_edges[0], q_edges[-1] = -1.0, 2.0
    bins = np.clip(np.digitize(pv, q_edges[1:-1]), 0, n_bins - 1)
    bin_pred, bin_obs, bin_n = [], [], []
    for bi in range(n_bins):
        m = bins == bi
        if int(np.sum(m)) >= 5:
            wm = wv[m]
            bin_pred.append(round(float(np.sum(wm * pv[m]) / np.sum(wm)), 5))
            bin_obs.append(round(float(np.sum(wm * yv[m]) / np.sum(wm)), 5))
            bin_n.append(int(np.sum(m)))

    return {
        "cal_intercept": round(float(a_citl), 4),
        "cal_slope": round(cal_slope, 4),
        "ici": round(ici, 5),
        "e50": round(e50, 5),
        "e90": round(e90, 5),
        "smooth_grid_pred": [round(float(x), 5) for x in grid_p],
        "smooth_grid_obs": [round(float(x), 5) for x in grid_obs],
        "bin_pred": bin_pred,
        "bin_obs": bin_obs,
        "bin_n": bin_n,
    }


def compute_weighted_decision_curve(
    y_true: np.ndarray,
    p_routed: np.ndarray,
    queried_mask: np.ndarray,
    taus: np.ndarray | list[float],
    gamma_query_over_cfn: float,
    *,
    sample_weight: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Vickers & Elkin (2006) Decision Curve NB(tau) across threshold range tau, incorporating query cost
    gamma = c_query / C_FN (scaled per 1,000 arrivals), with Treat-All and Treat-None references.
    """
    yv = np.asarray(y_true, dtype=float)
    pv = np.asarray(p_routed, dtype=float)
    qv = np.asarray(queried_mask, dtype=float)
    wv = np.ones(len(yv), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    w_tot = max(1e-12, float(np.sum(wv)))
    q_rate = float(np.sum(wv * qv) / w_tot)
    pos_rate = float(np.sum(wv * yv) / w_tot)
    neg_rate = 1.0 - pos_rate

    t_list = [round(float(t), 5) for t in taus]
    nb_m, nb_all, nb_none = [], [], []
    for tau in t_list:
        odds = tau / max(1e-9, 1.0 - tau)
        act = (pv >= tau).astype(float)
        tp_r = float(np.sum(wv * act * yv) / w_tot)
        fp_r = float(np.sum(wv * act * (1.0 - yv)) / w_tot)
        nb_m.append(round(1000.0 * (tp_r - odds * fp_r - gamma_query_over_cfn * q_rate), 4))
        nb_all.append(round(1000.0 * (pos_rate - odds * neg_rate), 4))
        nb_none.append(0.0)
    return {"taus": t_list, "nb_model": nb_m, "nb_treat_all": nb_all, "nb_treat_none": nb_none}


# ---------------------------------------------------------------------------
# 5. Pre-Purchase VOI Router (12x2 Dispersion vs 12-Bin Ablation)
# ---------------------------------------------------------------------------


def fit_predict_prepurchase_voi(
    xb_fit: np.ndarray,
    p1_fit: np.ndarray,
    p2_fit: np.ndarray,
    xb_cal: np.ndarray,
    p1_cal: np.ndarray,
    p2_cal: np.ndarray,
    xb_tgt: np.ndarray,
    p1_tgt: np.ndarray,
    contract: ClinicalContract,
    *,
    use_dispersion_2d: bool = True,
    dispersion_quantile: float = 0.75,
    min_bin_for_split: int = 60,
    seed: int = 42,
) -> np.ndarray:
    """Out-of-sample Pre-Purchase Empirical Martingale VOI (`12x2` dispersion-stratified vs `12`-bin baseline)."""
    delta_cal = logit(np.clip(p2_cal, 1e-6, 1 - 1e-6)) - logit(np.clip(p1_cal, 1e-6, 1 - 1e-6))
    edges = np.quantile(p1_cal, np.linspace(0.0, 1.0, 13))
    edges[0], edges[-1] = 0.0, 1.0
    b_p1_cal = np.clip(np.digitize(p1_cal, edges[1:-1], right=False), 0, 11)
    b_p1_tgt = np.clip(np.digitize(p1_tgt, edges[1:-1], right=False), 0, 11)
    q_levels = np.linspace(0.02, 0.98, 33)
    global_q = np.quantile(delta_cal, q_levels) if delta_cal.size >= 5 else np.zeros(33, dtype=float)

    if not use_dispersion_2d:
        bin_q12 = np.zeros((12, 33), dtype=float)
        for b_i in range(12):
            m = b_p1_cal == b_i
            bin_q12[b_i, :] = np.quantile(delta_cal[m], q_levels) if int(np.sum(m)) >= 15 else global_q
        return empirical_martingale_voi(p1_tgt, b_p1_tgt, bin_q12, contract)

    d_fit = np.abs(logit(np.clip(p2_fit, 1e-6, 1 - 1e-6)) - logit(np.clip(p1_fit, 1e-6, 1 - 1e-6)))
    if xb_fit.shape[0] >= 120 and float(np.std(d_fit)) > 1e-6:
        reg_u = lgb.LGBMRegressor(
            n_estimators=25,
            learning_rate=0.08,
            num_leaves=7,
            min_child_samples=max(20, min(50, xb_fit.shape[0] // 25)),
            random_state=seed,
            n_jobs=int(os.environ.get("UPCA_LGBM_JOBS", "4")),
            verbose=-1,
        )
        reg_u.fit(xb_fit, d_fit)
        u_cal = reg_u.predict(xb_cal)
        u_tgt = reg_u.predict(xb_tgt)
    else:
        u_cal = np.zeros(len(p1_cal), dtype=float)
        u_tgt = np.zeros(len(p1_tgt), dtype=float)

    bin_quantiles = np.zeros((24, 33), dtype=float)
    u_thresh = np.full(12, np.inf, dtype=float)
    for b_i in range(12):
        m = b_p1_cal == b_i
        n_m = int(np.sum(m))
        base_q = np.quantile(delta_cal[m], q_levels) if n_m >= 15 else global_q
        bin_quantiles[2 * b_i, :] = base_q
        bin_quantiles[2 * b_i + 1, :] = base_q
        if n_m >= min_bin_for_split and float(np.std(u_cal[m])) > 1e-6:
            thr = float(np.quantile(u_cal[m], dispersion_quantile))
            m_lo = m & (u_cal <= thr)
            m_hi = m & (u_cal > thr)
            if int(np.sum(m_lo)) >= 12 and int(np.sum(m_hi)) >= 12:
                u_thresh[b_i] = thr
                bin_quantiles[2 * b_i, :] = np.quantile(delta_cal[m_lo], q_levels)
                bin_quantiles[2 * b_i + 1, :] = np.quantile(delta_cal[m_hi], q_levels)

    hi_tgt = (u_tgt > u_thresh[b_p1_tgt]).astype(int)
    cell_tgt = 2 * b_p1_tgt + hi_tgt
    return empirical_martingale_voi(p1_tgt, cell_tgt, bin_quantiles, contract)


# ---------------------------------------------------------------------------
# 6. Unified Stage-A Pilot Discovery Engine Supporting All 5 Matched Modes
# ---------------------------------------------------------------------------


def discover_stage_a_ablation(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    base_cols: list[str],
    channels: list[ChannelDef],
    oof_p_base_eval: np.ndarray,
    contract: ClinicalContract,
    budget: float,
    *,
    mode: str = "offset_cz_schervish",
    pilot_fraction: float = 0.20,
    n_score_bins: int = 8,
    kappa: float = 24.0,
    alpha: float = 0.05,
    target_pop_weights: np.ndarray | None = None,
    raw_oof_p_base_for_fit: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[DiscoveryOutcome, PilotSample]:
    """Matched Stage-A two-phase pilot discovery supporting clean single-factor ablations:
      - `global_cmi`: global_neyman pilot + w_HT pilot GBDT weights + Bernoulli log-likelihood score
      - `weighted_cz_005`: collision_neyman pilot + w_HT * omega_CZ(0.05 floor) weights + Schervish score
      - `ipw_cz_wht`: collision_neyman pilot + raw w_HT weights + Schervish score
      - `offset_cz_logloss`: collision_neyman pilot + Breslow-Cain unweighted offset + Bernoulli log-likelihood
      - `offset_cz_schervish`: collision_neyman pilot + Breslow-Cain unweighted offset + Schervish score (UPCA Stage A)

    Note on case-control cohorts (`target_pop_weights` provided):
      `raw_oof_p_base_for_fit` is the sample-prevalence OOF probability matching `y_train`'s empirical prevalence
      for GBDT log-likelihood fitting, shifted to `oof_p_base_eval` (target clinical prevalence) before scoring,
      and weighted by `target_pop_weights` in the two-phase HT mean/SE!
    """
    yv = np.asarray(y_train, dtype=int)
    tau = contract.tau_star
    p_eval_all = np.clip(np.asarray(oof_p_base_eval, dtype=float), 1e-5, 1.0 - 1e-5)
    p_fit_all = (
        p_eval_all
        if raw_oof_p_base_for_fit is None
        else np.clip(np.asarray(raw_oof_p_base_for_fit, dtype=float), 1e-5, 1.0 - 1e-5)
    )
    prior_eval_minus_fit = logit(p_eval_all) - logit(p_fit_all)

    pilot_alloc_mode = "global_neyman" if mode == "global_cmi" else "collision_neyman"
    pilot = sample_pilot(
        yv,
        p_eval_all,
        pilot_fraction=pilot_fraction,
        mode=pilot_alloc_mode,
        tau_star=tau,
        kappa=kappa,
        n_bins=n_score_bins,
        seed=seed,
    )
    p_idx = pilot.indices
    yp = yv[p_idx]
    p_eval_p = p_eval_all[p_idx]
    p_fit_p = p_fit_all[p_idx]
    init_logit_fit = logit(p_fit_p)
    w_ht = pilot.ht_weights

    kappa_eff = solve_ess_kappa(
        bernoulli_kl_from_tau(p_eval_p, tau),
        kappa_max=kappa,
        min_effective_cz_samples=max(160.0, 0.24 * float(yp.size)),
    )

    # Compute exact stratum log-odds offset delta_P(b) from the exact `sample_pilot` score bins!
    quantiles = np.quantile(logit(p_eval_all), np.linspace(0.0, 1.0, n_score_bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    bin_all = np.clip(np.digitize(logit(p_eval_all), quantiles[1:-1], right=False), 0, n_score_bins - 1)
    bin_p = bin_all[p_idx]

    log_odds_shift_p = np.zeros(yp.size, dtype=float)
    for b_id in range(n_score_bins):
        N1 = max(1, int(np.sum((yv == 1) & (bin_all == b_id))))
        N0 = max(1, int(np.sum((yv == 0) & (bin_all == b_id))))
        n1 = max(1, int(np.sum((yp == 1) & (bin_p == b_id))))
        n0 = max(1, int(np.sum((yp == 0) & (bin_p == b_id))))
        log_odds_shift_p[bin_p == b_id] = float(np.log((n1 / float(N1)) / (n0 / float(N0))))

    use_offset = mode in ("offset_cz_schervish", "offset_cz_logloss")
    if use_offset:
        fit_init = init_logit_fit + log_odds_shift_p
        fit_weights = np.ones(yp.size, dtype=float)
    elif mode == "weighted_cz_005":
        fit_init = init_logit_fit.copy()
        fit_weights = w_ht * collision_zone_relevance_weights(p_eval_p, tau, kappa=kappa_eff, floor=0.05)
    elif mode in ("ipw_cz_wht", "global_cmi"):
        fit_init = init_logit_fit.copy()
        fit_weights = w_ht.copy()
    else:
        raise ValueError(f"Unknown Stage-A discovery mode: {mode}")

    use_schervish = mode in ("weighted_cz_005", "ipw_cz_wht", "offset_cz_schervish")

    rng = np.random.default_rng(seed + 17)
    pos = rng.permutation(np.flatnonzero(yp == 1))
    neg = rng.permutation(np.flatnonzero(yp == 0))
    n_folds = 4
    folds = [np.concatenate([pos[k::n_folds], neg[k::n_folds]]) for k in range(n_folds)]

    df_p = df_train.iloc[p_idx].reset_index(drop=True)
    fold_base: list[tuple[np.ndarray, np.ndarray]] = []
    fold_ch: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {ch.channel_id: [] for ch in channels}
    for k in range(n_folds):
        va = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        enc_b = FastFrameEncoder().fit(df_p[base_cols].iloc[tr])
        fold_base.append((enc_b.transform(df_p[base_cols].iloc[tr]), enc_b.transform(df_p[base_cols].iloc[va])))
        for ch in channels:
            enc_c = FastFrameEncoder().fit(df_p[list(ch.columns)].iloc[tr])
            fold_ch[ch.channel_id].append(
                (enc_c.transform(df_p[list(ch.columns)].iloc[tr]), enc_c.transform(df_p[list(ch.columns)].iloc[va]))
            )

    shift_to_eval_p = prior_eval_minus_fit[p_idx]

    def _oof_probs(sel_ids: list[str], step_seed: int) -> np.ndarray:
        oof = np.empty(yp.size, dtype=float)
        for k in range(n_folds):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            xb_tr, xb_va = fold_base[k]
            x_tr_k = np.hstack([xb_tr] + [fold_ch[cid][k][0] for cid in sel_ids]) if sel_ids else xb_tr
            x_va_k = np.hstack([xb_va] + [fold_ch[cid][k][1] for cid in sel_ids]) if sel_ids else xb_va
            clf = lgb.LGBMClassifier(
                n_estimators=85,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=20,
                max_delta_step=1.0,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=step_seed + k,
                n_jobs=int(os.environ.get("UPCA_LGBM_JOBS", "4")),
                verbose=-1,
            )
            clf.fit(x_tr_k, yp[tr], sample_weight=fit_weights[tr], init_score=fit_init[tr])
            # Subtract pilot offset (when use_offset) by anchoring evaluation on init_logit_fit[va] + shift_to_eval_p[va]
            raw_va = init_logit_fit[va] + shift_to_eval_p[va] + clf.predict(x_va_k, raw_score=True)
            oof[va] = np.clip(expit(np.clip(raw_va, -22.0, 22.0)), 1e-6, 1.0 - 1e-6)
        return oof

    def _score(p_vec: np.ndarray) -> np.ndarray:
        if use_schervish:
            return schervish_collision_proper_score(yp, p_vec, tau, kappa=kappa_eff)
        return log_likelihood_score(yp, p_vec)

    w_pop_p = np.ones(yp.size, dtype=float) if target_pop_weights is None else np.asarray(target_pop_weights, dtype=float)[p_idx]
    pop_scale = float(pilot.n_total) / max(1e-12, float(np.sum(w_ht * w_pop_p))) if target_pop_weights is not None else 1.0

    p_curr = _oof_probs([], seed + 100)
    sc_curr = _score(p_curr)
    selected: list[str] = []
    spent = 0.0
    steps_log: list[DiscoveryStepLog] = []
    step1_scores: dict[str, float] = {}

    for step in range(1, len(channels) + 1):
        affordable = [ch for ch in channels if ch.channel_id not in selected and spent + ch.cost <= budget + 1e-9]
        if not affordable:
            break
        cand_stats = []
        raw_pvals = []
        for ch in affordable:
            p_cand = _oof_probs(selected + [ch.channel_id], seed + 100 * step)
            sc_cand = _score(p_cand)
            psi = (sc_cand - sc_curr) * (w_pop_p * pop_scale)
            mean_g, se_g = two_phase_ht_mean_and_se(psi, pilot)
            z = mean_g / max(se_g, 1e-12)
            pval = float(1.0 - norm.cdf(z))
            raw_pvals.append(pval)
            if step == 1:
                step1_scores[ch.channel_id] = float(mean_g)
            cand_stats.append((ch, mean_g, se_g, z, sc_cand))

        p_holm = holm_adjust(raw_pvals)
        passing = [
            (ch, g, se, z, float(p_holm[i]), sc_c)
            for i, (ch, g, se, z, sc_c) in enumerate(cand_stats)
            if p_holm[i] < alpha and g > 0.0
        ]
        if not passing:
            best_i = int(np.argmax([x[1] for x in cand_stats]))
            ch_b, g_b, se_b, z_b, _ = cand_stats[best_i]
            steps_log.append(DiscoveryStepLog(step, None, float(g_b), float(se_b), float(z_b), float(p_holm[best_i]), 0.0))
            break
        winner = max(passing, key=lambda item: item[1] / max(item[0].cost, 1e-6))
        ch_w, g_w, se_w, z_w, ph_w, sc_w = winner
        selected.append(ch_w.channel_id)
        spent += ch_w.cost
        sc_curr = sc_w
        steps_log.append(DiscoveryStepLog(step, ch_w.channel_id, float(g_w), float(se_w), float(z_w), ph_w, ch_w.cost))

    return DiscoveryOutcome(mode, pilot_fraction, selected, spent, pilot.high_signal_fp_share_of_controls, steps_log, step1_scores), pilot


# ---------------------------------------------------------------------------
# 7. Stage-B Nested OOF Pre-Purchase Policy Screening (`UPCA-Fluid`)
# ---------------------------------------------------------------------------


def oof_policy_screening_stage_b(
    selected_ids: list[str],
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    p1_tr_oof_sample: np.ndarray,
    base_cols: list[str],
    ch_by_id: dict[str, ChannelDef],
    contract: ClinicalContract,
    *,
    pi_target_override: float | None = None,
    dispersion_quantile: float = 0.75,
    alpha_pol: float = 0.10,
    seed: int = 42,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Stage B of UPCA-Fluid: Nested 3-fold OOF Pre-Purchase Policy Screening on D_train (`0%` peek at D_cal/D_test)."""
    if len(selected_ids) == 0:
        return [], []

    z_crit = float(norm.ppf(1.0 - alpha_pol))  # 1.2816 for alpha_pol = 0.10
    yv = np.asarray(y_tr, dtype=int)
    pi_sample = float(np.mean(yv))
    w_tr = np.ones(len(yv), dtype=float) if pi_target_override is None else case_control_prevalence_weights(yv, pi_target_override)
    p1_eval_all = (
        np.clip(p1_tr_oof_sample, 1e-5, 1.0 - 1e-5)
        if pi_target_override is None
        else adjust_probs_for_target_prevalence(p1_tr_oof_sample, pi_sample, pi_target_override)
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 31)
    folds = list(skf.split(df_tr, yv))
    cache_nb: dict[tuple[str, ...], np.ndarray] = {}

    def _oof_nb_vec(cids: list[str]) -> np.ndarray:
        key = tuple(cids)
        if key in cache_nb:
            return cache_nb[key]
        if len(cids) == 0:
            res0 = unit_net_benefit_vec(yv, p1_eval_all, np.zeros(len(yv)), contract)
            cache_nb[key] = res0
            return res0
        s2_cols = [c for cid in cids for c in ch_by_id[cid].columns]
        p2_oof = np.zeros(len(yv), dtype=float)
        q_oof = np.zeros(len(yv), dtype=bool)
        for k, (tr_i, va_i) in enumerate(folds):
            inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 71 + k)
            fit_rel, cal_rel = next(inner_skf.split(np.zeros(len(tr_i)), yv[tr_i]))
            fit_i, cal_i = tr_i[fit_rel], tr_i[cal_rel]

            enc_b = FastFrameEncoder().fit(df_tr[base_cols].iloc[fit_i])
            enc_z = FastFrameEncoder().fit(df_tr[s2_cols].iloc[fit_i])
            xb_fit = enc_b.transform(df_tr[base_cols].iloc[fit_i])
            xb_cal = enc_b.transform(df_tr[base_cols].iloc[cal_i])
            xb_va = enc_b.transform(df_tr[base_cols].iloc[va_i])
            z_fit = enc_z.transform(df_tr[s2_cols].iloc[fit_i])
            z_cal = enc_z.transform(df_tr[s2_cols].iloc[cal_i])
            z_va = enc_z.transform(df_tr[s2_cols].iloc[va_i])

            p1_fit_s = np.clip(p1_tr_oof_sample[fit_i], 1e-5, 1.0 - 1e-5)
            p1_cal_s = np.clip(p1_tr_oof_sample[cal_i], 1e-5, 1.0 - 1e-5)
            p1_va_s = np.clip(p1_tr_oof_sample[va_i], 1e-5, 1.0 - 1e-5)

            # Adjust tau_star into sample-prevalence scale when fitting FluidCollisionBackbone on case-control data
            tau_fit = (
                contract.tau_star
                if pi_target_override is None
                else float(adjust_probs_for_target_prevalence(np.array([contract.tau_star]), pi_target_override, pi_sample)[0])
            )
            clf = FluidCollisionBackboneClassifier(
                tau_star=tau_fit, kappa_max=20.0, min_cz_ess=180.0, seed=seed + k
            ).fit(np.hstack([xb_fit, z_fit]), z_fit, yv[fit_i], logit(p1_fit_s))

            raw_fit = clf.decision_function(np.hstack([xb_fit, z_fit]), z_fit, logit(p1_fit_s))
            raw_cal = clf.decision_function(np.hstack([xb_cal, z_cal]), z_cal, logit(p1_cal_s))
            raw_va = clf.decision_function(np.hstack([xb_va, z_va]), z_va, logit(p1_va_s))
            platt = fit_platt_calibrator(raw_cal, yv[cal_i])
            p2_fit_k = np.clip(apply_platt(raw_fit, platt), 1e-6, 1.0 - 1.0e-6)
            p2_cal_k = np.clip(apply_platt(raw_cal, platt), 1e-6, 1.0 - 1.0e-6)
            p2_va_k = np.clip(apply_platt(raw_va, platt), 1e-6, 1.0 - 1.0e-6)

            if pi_target_override is not None:
                pi_cal_k = float(np.mean(yv[cal_i]))
                p2_fit_k = adjust_probs_for_target_prevalence(p2_fit_k, pi_cal_k, pi_target_override)
                p2_cal_k = adjust_probs_for_target_prevalence(p2_cal_k, pi_cal_k, pi_target_override)
                p2_va_k = adjust_probs_for_target_prevalence(p2_va_k, pi_cal_k, pi_target_override)

            voi_va = fit_predict_prepurchase_voi(
                xb_fit,
                p1_eval_all[fit_i],
                p2_fit_k,
                xb_cal,
                p1_eval_all[cal_i],
                p2_cal_k,
                xb_va,
                p1_eval_all[va_i],
                contract,
                use_dispersion_2d=True,
                dispersion_quantile=dispersion_quantile,
                seed=seed + k,
            )
            q_va = voi_va > contract.per_instance_query_cost
            p2_oof[va_i] = np.where(q_va, p2_va_k, p1_eval_all[va_i])
            q_oof[va_i] = q_va

        res_vec = unit_net_benefit_vec(yv, p2_oof, q_oof, contract)
        cache_nb[key] = res_vec
        return res_vec

    def _weighted_z(diff_vec: np.ndarray) -> tuple[float, float]:
        w_sum = float(np.sum(w_tr))
        mean_d = float(np.sum(w_tr * diff_vec) / max(1e-12, w_sum))
        resid = diff_vec - mean_d
        se_d = float(np.sqrt(np.sum((w_tr ** 2) * (resid ** 2)) / max(1e-12, w_sum ** 2)))
        return mean_d, mean_d / max(se_d, 1e-9)

    nb_s1 = _oof_nb_vec([])
    active = list(selected_ids)
    prune_log: list[dict[str, Any]] = []
    curr_nb = _oof_nb_vec(active)

    while len(active) > 0:
        best_drop = None
        best_score = -np.inf
        for cid in active:
            reduced = [c for c in active if c != cid]
            red_nb = _oof_nb_vec(reduced)
            solo_nb = curr_nb if len(active) == 1 else _oof_nb_vec([cid])
            solo_gain, z_solo = _weighted_z(solo_nb - nb_s1)
            marg_mean, z_marg = _weighted_z(curr_nb - red_nb)
            z_drop = -z_marg
            should_prune = (z_drop > z_crit) or (z_solo < z_crit and z_marg < z_crit)
            prune_log.append({
                "tested_drop": cid,
                "active_size": len(active),
                "solo_routed_gain": round(solo_gain, 4),
                "z_solo": round(z_solo, 3),
                "marginal_bundle_gain": round(marg_mean, 4),
                "z_marg": round(z_marg, 3),
                "pruned_candidate": bool(should_prune),
            })
            if should_prune and (-marg_mean) > best_score:
                best_score = -marg_mean
                best_drop = (cid, red_nb)
        if best_drop is None:
            break
        cid_dropped, curr_nb = best_drop
        active.remove(cid_dropped)

    return active, prune_log


# ---------------------------------------------------------------------------
# 8. R15 Diagnostics: Within-Stratum Independence & Two-Phase MLE Efficiency
# ---------------------------------------------------------------------------


def check_within_stratum_pilot_independence(
    df_train: pd.DataFrame,
    channels: list[ChannelDef],
    pilot: PilotSample,
    y_train: np.ndarray,
    oof_p_base: np.ndarray,
    n_bins: int = 8,
) -> dict[str, Any]:
    """Empirically verify I_i _|_ Z_i | (Y_i, b_i) for Phase-2 pilot inclusion (`R15`).

    Computes the stratified Mantel-Haenszel / within-stratum standardized score statistic testing
    association between pilot indicator I_i and every Stage-2 feature column conditional on stratum h = (Y, b).
    """
    yv = np.asarray(y_train, dtype=int)
    pv = np.clip(np.asarray(oof_p_base, dtype=float), 1e-6, 1.0 - 1e-6)
    quantiles = np.quantile(logit(pv), np.linspace(0.0, 1.0, n_bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    bins = np.clip(np.digitize(logit(pv), quantiles[1:-1], right=False), 0, n_bins - 1)
    strata = yv * n_bins + bins

    in_pilot = np.zeros(len(yv), dtype=float)
    in_pilot[pilot.indices] = 1.0

    cols = list(dict.fromkeys(c for ch in channels for c in ch.columns if not ch.channel_id.startswith("ch_null")))
    enc = FastFrameEncoder(include_missing_indicators=False).fit(df_train[cols])
    z_mat = enc.transform(df_train[cols]).astype(float)

    z_stats = []
    for j in range(z_mat.shape[1]):
        zj = z_mat[:, j]
        num_sum = 0.0
        var_sum = 0.0
        for h in np.unique(strata):
            m = strata == h
            N_h = int(np.sum(m))
            n_h = int(np.sum(in_pilot[m]))
            if N_h <= 2 or n_h <= 0 or n_h >= N_h:
                continue
            zh = zj[m]
            var_z = float(np.var(zh, ddof=1))
            if var_z <= 1e-12:
                continue
            mean_z = float(np.mean(zh))
            num_sum += float(np.sum(in_pilot[m] * (zh - mean_z)))
            var_sum += float(n_h) * (1.0 - float(n_h) / float(N_h)) * var_z
        if var_sum > 1e-12:
            z_stats.append(num_sum / np.sqrt(var_sum))

    pvals = [float(2.0 * (1.0 - norm.cdf(abs(z)))) for z in z_stats]
    min_p_bonf = min(1.0, min(pvals) * max(1, len(pvals))) if pvals else 1.0
    omnibus_chi2 = float(np.sum(np.asarray(z_stats) ** 2))
    omnibus_p = float(1.0 - chi2.cdf(omnibus_chi2, df=max(1, len(z_stats)))) if z_stats else 1.0
    return {
        "n_features_tested": len(z_stats),
        "max_abs_within_stratum_z": round(float(max( (abs(z) for z in z_stats), default=0.0 )), 4),
        "median_p_value": round(float(np.median(pvals)) if pvals else 1.0, 4),
        "min_bonferroni_p_value": round(min_p_bonf, 4),
        "omnibus_chi2": round(omnibus_chi2, 3),
        "omnibus_df": len(z_stats),
        "omnibus_p_value": round(omnibus_p, 4),
    }


def cochran_q_heterogeneity(estimates: list[float], ses: list[float]) -> dict[str, float]:
    """Compute Cochran's Q statistic and heterogeneity p-value across subgroup Delta-NB estimates (`R11`)."""
    est = np.asarray(estimates, dtype=float)
    se = np.maximum(1e-6, np.asarray(ses, dtype=float))
    k = len(est)
    if k < 2:
        return {"cochran_q": 0.0, "df": 0, "p_heterogeneity": 1.0, "i2_pct": 0.0}
    w = 1.0 / (se ** 2)
    mu_fe = float(np.sum(w * est) / np.sum(w))
    q_val = float(np.sum(w * ((est - mu_fe) ** 2)))
    df = k - 1
    p_het = float(1.0 - chi2.cdf(q_val, df=df))
    i2 = max(0.0, 100.0 * (q_val - df) / max(1e-9, q_val))
    return {
        "cochran_q": round(q_val, 3),
        "df": int(df),
        "p_heterogeneity": round(p_het, 4),
        "i2_pct": round(i2, 1),
    }

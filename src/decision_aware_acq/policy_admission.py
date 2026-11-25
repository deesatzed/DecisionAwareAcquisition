"""Track 1 Policy-Aware Channel Admission, Skipped-Case Safety Certification, and Murphy Diagrams.

Reuses validated primitives from `collision_zone.three_predicates` while implementing:
  1. Policy-Aware Pre-Purchase Admission: Delta_m(S) = V(S U {m}) - V(S) under frozen VOI routing
  2. Skipped-Case Safety Certification: exact Clopper-Pearson upper bounds on skipped decision flips & missed rescues
  3. Incremental Murphy Diagrams across clinical action thresholds tau
  4. Selective-Label & Missed-Event Rescue Audit (Tested vs. Untested/Discharged encounters)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.metrics import average_precision_score, roc_auc_score


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided (1 - alpha) Clopper-Pearson Binomial upper bound for k successes in n trials."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - alpha, k + 1, n - k))


def per_record_net_utility(
    y: np.ndarray,
    p_route: np.ndarray,
    queried: np.ndarray,
    tau_star: float,
    cost_fp: float,
    cost_fn: float,
    query_cost: float,
) -> np.ndarray:
    """Per-encounter Net Benefit contribution (scaled per 1,000 arrivals) under decision contract K."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p_route, dtype=float)
    q_arr = np.asarray(queried, dtype=float)
    pred_pos = (p_arr >= tau_star).astype(float)
    tp = pred_pos * y_arr
    fp = pred_pos * (1.0 - y_arr)
    nb_i = 1000.0 * (tp - (cost_fp / cost_fn) * fp - (query_cost / cost_fn) * q_arr)
    return nb_i


def skipped_case_safety_certificate(
    y_true: np.ndarray,
    p_stage1: np.ndarray,
    p_stage2_oracle: np.ndarray,
    queried_mask: np.ndarray,
    tau_star: float,
    cost_fp: float,
    cost_fn: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Certify decision-flip rate, missed TP rescue rate, and forgone NB among skipped (Q=0) arrivals."""
    y = np.asarray(y_true, dtype=int)
    p1 = np.asarray(p_stage1, dtype=float)
    p2 = np.asarray(p_stage2_oracle, dtype=float)
    q = np.asarray(queried_mask, dtype=bool)
    skipped = ~q
    n_total = len(y)
    n_skip = int(skipped.sum())
    if n_skip == 0:
        return {
            "n_total": n_total,
            "n_skipped": 0,
            "skip_rate": 0.0,
            "flip_count": 0,
            "flip_rate": 0.0,
            "flip_rate_ub95": 0.0,
            "missed_tp_rescue_count": 0,
            "missed_tp_rescue_rate": 0.0,
            "missed_tp_rescue_ub95": 0.0,
            "forgone_nb_per_1k_all": 0.0,
        }

    a1 = p1[skipped] >= tau_star
    a2 = p2[skipped] >= tau_star
    y_s = y[skipped]
    flips = a1 != a2
    missed_tp_rescue = (y_s == 1) & (~a1) & a2
    k_flip = int(flips.sum())
    k_missed_tp = int(missed_tp_rescue.sum())

    # Forgone Net Benefit per 1,000 total arrivals (excluding query cost saved)
    nb_s1_skip = per_record_net_utility(y_s, p1[skipped], np.zeros(n_skip), tau_star, cost_fp, cost_fn, 0.0)
    nb_s2_skip = per_record_net_utility(y_s, p2[skipped], np.zeros(n_skip), tau_star, cost_fp, cost_fn, 0.0)
    forgone_nb = float(np.sum(nb_s2_skip - nb_s1_skip) / max(1, n_total))

    return {
        "n_total": n_total,
        "n_skipped": n_skip,
        "skip_rate": float(n_skip / max(1, n_total)),
        "flip_count": k_flip,
        "flip_rate": float(k_flip / n_skip),
        "flip_rate_ub95": clopper_pearson_upper(k_flip, n_skip, alpha=alpha),
        "missed_tp_rescue_count": k_missed_tp,
        "missed_tp_rescue_rate": float(k_missed_tp / n_skip),
        "missed_tp_rescue_ub95": clopper_pearson_upper(k_missed_tp, n_skip, alpha=alpha),
        "forgone_nb_per_1k_all": forgone_nb,
    }


def compute_murphy_diagram(
    y_true: np.ndarray,
    p_model: np.ndarray,
    p_ref: np.ndarray,
    taus: list[float] | np.ndarray,
) -> dict[str, list[float]]:
    """Compute incremental Murphy elementary-score gain S_tau(model) - S_tau(ref) across thresholds tau.

    Elementary score (Ehm et al., JRSS-B 2016):
        s_tau(y, p) = -(1 - tau) * I{p < tau, y = 1} - tau * I{p >= tau, y = 0}
    Positive delta means `p_model` strictly improves over `p_ref` at decision threshold tau.
    """
    y = np.asarray(y_true, dtype=float)
    pm = np.asarray(p_model, dtype=float)
    pr = np.asarray(p_ref, dtype=float)
    tau_list = [float(t) for t in taus]
    deltas = []
    for t in tau_list:
        loss_m = (1.0 - t) * ((pm < t) & (y == 1)) + t * ((pm >= t) & (y == 0))
        loss_r = (1.0 - t) * ((pr < t) & (y == 1)) + t * ((pr >= t) & (y == 0))
        deltas.append(float(np.mean(loss_r - loss_m) * 1000.0))
    return {"taus": tau_list, "murphy_gain_per_1k": deltas}


def selective_label_audit(
    y_true: np.ndarray,
    p_model: np.ndarray,
    tested_mask: np.ndarray,
    tau_star: float,
    cost_fp: float,
    cost_fn: float,
    queried_mask: np.ndarray | None = None,
    query_cost: float = 0.0,
) -> dict[str, Any]:
    """Evaluate model performance overall, on clinician-tested patients, and on untested/discharged patients."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_model, dtype=float)
    t_mask = np.asarray(tested_mask, dtype=bool)
    q = np.zeros(len(y), dtype=float) if queried_mask is None else np.asarray(queried_mask, dtype=float)

    def _slice_stats(mask: np.ndarray) -> dict[str, Any]:
        ys = y[mask]
        ps = p[mask]
        qs = q[mask]
        if len(ys) == 0 or len(np.unique(ys)) < 2:
            return {"n": int(len(ys)), "events": int(ys.sum()), "auroc": None, "auprc": None, "nb": None}
        nb = float(np.mean(per_record_net_utility(ys, ps, qs, tau_star, cost_fp, cost_fn, query_cost)))
        flagged = ps >= tau_star
        return {
            "n": int(len(ys)),
            "events": int(ys.sum()),
            "prevalence": float(ys.mean()),
            "auroc": float(roc_auc_score(ys, ps)),
            "auprc": float(average_precision_score(ys, ps)),
            "nb_per_1k": nb,
            "flag_rate": float(flagged.mean()),
            "sensitivity_at_tau": float(((flagged) & (ys == 1)).sum() / max(1, int(ys.sum()))),
            "rescued_events_at_tau": int(((flagged) & (ys == 1)).sum()),
        }

    return {
        "all_encounters": _slice_stats(np.ones(len(y), dtype=bool)),
        "clinician_tested": _slice_stats(t_mask),
        "clinician_untested": _slice_stats(~t_mask),
    }


@dataclass(frozen=True)
class PolicyAdmissionCandidate:
    channel: str
    delta_policy_nb: float
    se_policy_nb: float
    z_stat: float
    p_value: float
    query_rate: float
    cost: float
    gain_per_cost: float


def compute_policy_delta_stats(
    nb_with_m: np.ndarray,
    nb_without_m: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Compute paired policy-value difference Delta_m = V(S U {m}) - V(S), SE, Z-stat, one-sided p-value."""
    from scipy.stats import norm

    diff = np.asarray(nb_with_m, dtype=float) - np.asarray(nb_without_m, dtype=float)
    n = len(diff)
    if n < 2:
        return 0.0, 1e-6, 0.0, 1.0
    if weights is None:
        mean_d = float(np.mean(diff))
        se_d = float(np.std(diff, ddof=1) / np.sqrt(n))
    else:
        w = np.asarray(weights, dtype=float)
        w_sum = float(np.sum(w))
        mean_d = float(np.sum(w * diff) / max(1e-12, w_sum))
        resid = diff - mean_d
        # Kish-safe Horvitz-Thompson variance approximation
        se_d = float(np.sqrt(np.sum((w ** 2) * (resid ** 2)) / max(1e-12, w_sum ** 2)))
    se_d = max(se_d, 1e-9)
    z = mean_d / se_d
    pval = float(1.0 - norm.cdf(z))
    return mean_d, se_d, z, pval

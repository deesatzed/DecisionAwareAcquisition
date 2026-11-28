"""Paper 3 (`DecisionAwareAcquisition` v2.2.0) Methodological & Ledger Repair Primitives.

Propagates all cross-paper statistical, cost-accounting, and clinical-governance lessons
from sealed Paper 1 (`imboreceipt` v2.2.1) and sealed Paper 2 (`collision-zone-ed` v2.3.1):
  1. Explicit `ChannelCostSchedule` decomposing fixed encounter query overhead and
     per-channel proportional processing fees (`C_i(A_i)`), supporting Common-Panel Bagged VOI,
     Union-Panel Bagged VOI, Full 100% Stage-2 querying, Boundary Uncertainty, Stage-B OOF
     subset-cost policy screening, and proportionally channel-billed `Covert-CMI` / `Direct-VOI` AFA.
  2. Exact normalized-Beta / Boltzmann-KL threshold identity (`Buja, Stuetzle & Shen, 2005`),
     linear-weight Horvitz-Thompson two-phase variance SRS reduction (`s^2 / n`), and
     finite-capacity `XOR` counterexample motivating the learner-specific predictive-utility null.
  3. Disjoint `50/50` Split-Pilot Breslow-Cain Offset Discovery (`P = P_fit uplus P_conf`)
     with conditionally design-unbiased HT mean/SE on `P_conf`.
  4. Scale-invariant Stage-2 specialist diagnostics (`R_CZ` RMS gated log-odds contribution
     and threshold decision-flip rate) alongside `Escalation-Only` / `De-escalation-Only` VOI.
  5. 4-Way Prior-Shift Taxonomy contrasting Unadjusted, Purely Unlabeled `Saerens (2002)` EM,
     Disjoint Calibration-Slice Prior (`0` test labels), and Oracle Test-Prevalence Bound.
  6. Subgroup Cochran's `Q` heterogeneity testing for `VOI vs. Full 100%`, `UPCA vs. Stage-1`,
     and `UPCA vs. Global-CMI`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import chi2, norm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
    fit_platt_calibrator,
    holm_adjust,
    log_likelihood_score,
    sample_pilot,
    schervish_collision_proper_score,
    schervish_collision_weight,
    solve_ess_kappa,
    two_phase_ht_mean_and_se,
    union_collision_gate,
)
from decision_aware_acq.upca_v2_core import (
    adjust_probs_for_target_prevalence,
    case_control_prevalence_weights,
    fit_predict_prepurchase_voi,
    standardized_net_benefit,
)


@dataclass(frozen=True)
class ChannelCostSchedule:
    """Explicit encounter-level procedure/retrieval overhead + proportional channel fee ledger.

    Calibrated so a reference bundle of total weight `reference_bundle_weight` equals `base_c_query`
    (in `C_FP` units), while single-channel panels cost strictly less and multi-channel / union
    bundles scale linearly with acquired channel weights `c_m`:
        C_i(A_i) = mult * (I{A_i != empty} * c_overhead + sum_{m in A_i} c_unit * c_m).
    """

    base_c_query: float
    reference_bundle_weight: float
    channel_weights: dict[str, float]
    overhead_share: float = 0.25
    cost_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.base_c_query < 0.0:
            raise ValueError("base_c_query must be >= 0.")
        if self.reference_bundle_weight <= 0.0:
            raise ValueError("reference_bundle_weight must be > 0.")
        if not (0.0 <= self.overhead_share <= 1.0):
            raise ValueError("overhead_share must lie in [0, 1].")

    @property
    def c_overhead(self) -> float:
        return float(self.cost_multiplier * self.base_c_query * self.overhead_share)

    @property
    def c_unit(self) -> float:
        return float(
            self.cost_multiplier
            * self.base_c_query
            * (1.0 - self.overhead_share)
            / self.reference_bundle_weight
        )

    def with_multiplier(self, mult: float) -> ChannelCostSchedule:
        return ChannelCostSchedule(
            base_c_query=self.base_c_query,
            reference_bundle_weight=self.reference_bundle_weight,
            channel_weights=dict(self.channel_weights),
            overhead_share=self.overhead_share,
            cost_multiplier=float(mult),
        )

    def panel_cost_cfp(self, channel_ids: list[str] | tuple[str, ...] | set[str]) -> float:
        unique_ids = sorted(set(channel_ids))
        if not unique_ids:
            return 0.0
        w_sum = sum(float(self.channel_weights[cid]) for cid in unique_ids)
        return float(self.c_overhead + self.c_unit * w_sum)

    def mask_costs_cfp(self, mask_matrix: np.ndarray, ordered_channel_ids: list[str]) -> np.ndarray:
        """Compute per-encounter cost vector `C_i(M_i)` (in `C_FP` units) for binary mask `(N, M)`."""
        mm = np.asarray(mask_matrix, dtype=float) > 0.5
        if mm.ndim != 2 or mm.shape[1] != len(ordered_channel_ids):
            raise ValueError("mask_matrix shape must match (N, len(ordered_channel_ids)).")
        w_vec = np.array([float(self.channel_weights[cid]) for cid in ordered_channel_ids], dtype=float)
        any_q = np.any(mm, axis=1).astype(float)
        return any_q * self.c_overhead + (mm.astype(float) @ w_vec) * self.c_unit


def variable_cost_unit_nb_vec(
    y: np.ndarray,
    p_route: np.ndarray,
    cost_cfp_vec: np.ndarray | float,
    contract: ClinicalContract,
) -> np.ndarray:
    """Per-encounter Net Benefit contribution (scaled per 1,000 arrivals) with encounter-specific `C_i` (in `C_FP` units)."""
    yv = np.asarray(y, dtype=float)
    pv = np.asarray(p_route, dtype=float)
    cv = np.broadcast_to(np.asarray(cost_cfp_vec, dtype=float), yv.shape)
    act = (pv >= contract.tau_star).astype(float)
    tp = act * yv
    fp = act * (1.0 - yv)
    return 1000.0 * (tp - (contract.cost_fp / contract.cost_fn) * fp - (cv / contract.cost_fn))


def variable_cost_nb_per_1k(
    y: np.ndarray,
    p_route: np.ndarray,
    cost_cfp_vec: np.ndarray | float,
    contract: ClinicalContract,
    sample_weight: np.ndarray | None = None,
) -> float:
    u = variable_cost_unit_nb_vec(y, p_route, cost_cfp_vec, contract)
    if sample_weight is None:
        return float(np.mean(u))
    w = np.asarray(sample_weight, dtype=float)
    return float(np.sum(w * u) / max(1e-12, float(np.sum(w))))


def variable_cost_cluster_bootstrap_diff(
    y: np.ndarray,
    p_arm: np.ndarray,
    cost_arm_cfp: np.ndarray | float,
    p_ref: np.ndarray,
    cost_ref_cfp: np.ndarray | float,
    cluster_ids: np.ndarray,
    contract: ClinicalContract,
    *,
    sample_weight: np.ndarray | None = None,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Vectorized 10,000-replicate patient-cluster ratio-estimator bootstrap for variable-cost NB difference."""
    u_arm = variable_cost_unit_nb_vec(y, p_arm, cost_arm_cfp, contract)
    u_ref = variable_cost_unit_nb_vec(y, p_ref, cost_ref_cfp, contract)
    diff_i = u_arm - u_ref
    w_i = np.ones_like(diff_i, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    obs_diff = float(np.sum(w_i * diff_i) / max(1e-12, float(np.sum(w_i))))
    if float(np.max(np.abs(diff_i))) <= 1e-15:
        return {
            "diff_mean": 0.0,
            "se": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "p_two_sided": 1.0,
            "exact_tie": 1.0,
        }

    _, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    n_clusters = int(inv.max()) + 1
    c_num = np.bincount(inv, weights=w_i * diff_i, minlength=n_clusters)
    c_den = np.bincount(inv, weights=w_i, minlength=n_clusters)

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot, dtype=float)
    probs = np.full(n_clusters, 1.0 / float(n_clusters))
    chunk = 2500
    for st in range(0, n_boot, chunk):
        en = min(n_boot, st + chunk)
        counts = rng.multinomial(n_clusters, probs, size=en - st).astype(float)
        num = counts @ c_num
        den = np.maximum(1e-12, counts @ c_den)
        boot_diffs[st:en] = num / den

    ci_lo, ci_hi = np.quantile(boot_diffs, [0.025, 0.975])
    k_le = int(np.sum(boot_diffs <= 0.0))
    k_ge = int(np.sum(boot_diffs >= 0.0))
    p_dh = min(1.0, 2.0 * float(min(k_le, k_ge) + 1) / float(n_boot + 1))
    return {
        "diff_mean": round(obs_diff, 4),
        "se": round(float(np.std(boot_diffs, ddof=1)), 4),
        "ci95_low": round(float(ci_lo), 4),
        "ci95_high": round(float(ci_hi), 4),
        "p_two_sided": round(float(p_dh), 5),
        "exact_tie": 0.0,
    }


def normalized_beta_kl_identity_max_error(tau_star: float = 0.05, kappa: float = 24.0) -> float:
    """Verify `w_{tau*,kappa}(p)/w_{tau*,kappa}(tau*) == exp(-kappa * D_KL(Bern(tau*) || Bern(p)))`."""
    p_grid = np.linspace(1e-4, 1.0 - 1e-4, 2000)
    w_p = schervish_collision_weight(p_grid, tau_star, kappa)
    w_peak = float(schervish_collision_weight(np.array([tau_star]), tau_star, kappa)[0])
    lhs = w_p / w_peak
    rhs = np.exp(-float(kappa) * bernoulli_kl_from_tau(p_grid, tau_star))
    return float(np.max(np.abs(lhs - rhs)))


def two_phase_variance_srs_identity_error(n_total: int = 500, n_pilot: int = 100, seed: int = 7) -> float:
    """Verify linear-weight HT variance `V_I + V_II` reduces identically to `s^2 / n_pilot` under SRS (`H=1`)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_total, size=n_pilot, replace=False))
    psi = rng.normal(loc=0.05, scale=0.2, size=n_pilot)
    pilot_srs = PilotSample(
        indices=idx,
        strata=np.zeros(n_pilot, dtype=int),
        ht_weights=np.full(n_pilot, float(n_total) / float(n_pilot), dtype=float),
        pop_counts={0: n_total},
        sample_counts={0: n_pilot},
        n_total=n_total,
        high_signal_fp_share_of_controls=0.0,
    )
    _, se_ht = two_phase_ht_mean_and_se(psi, pilot_srs)
    s2_over_n = float(np.var(psi, ddof=1)) / float(n_pilot)
    return float(abs(se_ht**2 - s2_over_n))


def xor_finite_capacity_counterexample(n_samples: int = 2400, seed: int = 13) -> dict[str, float]:
    """Demonstrate `I(Y; Z | X1, X2) == 0` where depth-1 GBDT stumps still gain strictly positive OOF score."""
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, size=n_samples)
    x2 = rng.integers(0, 2, size=n_samples)
    y = (x1 ^ x2).astype(int)
    z = (x1 ^ x2).astype(float)  # Deterministic function of (X1, X2) -> I(Y; Z | X1, X2) = 0

    half = n_samples // 2
    xb_tr, xb_te = np.column_stack([x1[:half], x2[:half]]), np.column_stack([x1[half:], x2[half:]])
    xz_tr = np.column_stack([xb_tr, z[:half]])
    xz_te = np.column_stack([xb_te, z[half:]])
    y_tr, y_te = y[:half], y[half:]

    stump_base = lgb.LGBMClassifier(
        n_estimators=40, max_depth=1, num_leaves=2, learning_rate=0.1, random_state=seed, verbose=-1
    ).fit(xb_tr, y_tr)
    stump_aug = lgb.LGBMClassifier(
        n_estimators=40, max_depth=1, num_leaves=2, learning_rate=0.1, random_state=seed, verbose=-1
    ).fit(xz_tr, y_tr)

    p_b = np.clip(stump_base.predict_proba(xb_te)[:, 1], 1e-6, 1 - 1e-6)
    p_a = np.clip(stump_aug.predict_proba(xz_te)[:, 1], 1e-6, 1 - 1e-6)
    gain_cz = float(
        np.mean(
            schervish_collision_proper_score(y_te, p_a, 0.25, kappa=12.0)
            - schervish_collision_proper_score(y_te, p_b, 0.25, kappa=12.0)
        )
    )
    gain_ll = float(np.mean(log_likelihood_score(y_te, p_a) - log_likelihood_score(y_te, p_b)))
    return {"cz_score_gain": round(gain_cz, 5), "logloss_gain": round(gain_ll, 5)}


def discover_stage_a_split_pilot(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    base_cols: list[str],
    channels: list[ChannelDef],
    oof_p_base_eval: np.ndarray,
    contract: ClinicalContract,
    budget: float,
    *,
    pilot_fraction: float = 0.20,
    n_score_bins: int = 8,
    kappa: float = 24.0,
    alpha: float = 0.05,
    target_pop_weights: np.ndarray | None = None,
    raw_oof_p_base_for_fit: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[DiscoveryOutcome, PilotSample]:
    """Disjoint `50/50` Split-Pilot Breslow-Cain Offset Discovery (`P = P_fit uplus P_conf`).

    Fits unweighted Breslow-Cain offset GBDTs strictly on `P_fit` (50% within each stratum `h`),
    evaluating Horvitz-Thompson mean/SE on the remaining frame `N_rem = N - |P_fit|` via `P_conf`
    so `{psi_i}_{i in P_conf}` are conditionally fixed given `(D_train, P_fit)`.
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

    pilot = sample_pilot(
        yv,
        p_eval_all,
        pilot_fraction=pilot_fraction,
        mode="collision_neyman",
        tau_star=tau,
        kappa=kappa,
        n_bins=n_score_bins,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 707)
    p_idx = pilot.indices
    p_strata = pilot.strata  # length == len(p_idx)

    fit_mask_local = np.zeros(len(p_idx), dtype=bool)
    conf_pop_counts: dict[int, int] = {}
    conf_samp_counts: dict[int, int] = {}
    for h in sorted(pilot.pop_counts.keys()):
        loc_h = np.flatnonzero(p_strata == h)
        nh = len(loc_h)
        Nh = int(pilot.pop_counts[h])
        if nh < 4:
            n_conf_h = min(nh, max(2, nh // 2))
            n_fit_h = nh - n_conf_h
        else:
            n_fit_h = nh // 2
            n_conf_h = nh - n_fit_h
        perm_h = rng.permutation(loc_h)
        if n_fit_h > 0:
            fit_mask_local[perm_h[:n_fit_h]] = True
        conf_pop_counts[h] = max(n_conf_h, Nh - n_fit_h)
        conf_samp_counts[h] = n_conf_h

    idx_fit = p_idx[fit_mask_local]
    idx_conf = p_idx[~fit_mask_local]
    strata_conf = p_strata[~fit_mask_local]
    n_rem = int(sum(conf_pop_counts.values()))
    w_conf = np.array(
        [
            float(conf_pop_counts[int(h_val)]) / float(max(1, conf_samp_counts[int(h_val)]))
            for h_val in strata_conf
        ],
        dtype=float,
    )
    pilot_conf = PilotSample(
        indices=idx_conf,
        strata=strata_conf,
        ht_weights=w_conf,
        pop_counts=conf_pop_counts,
        sample_counts=conf_samp_counts,
        n_total=n_rem,
        high_signal_fp_share_of_controls=pilot.high_signal_fp_share_of_controls,
    )

    quantiles = np.quantile(logit(p_eval_all), np.linspace(0.0, 1.0, n_score_bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    bin_all = np.clip(np.digitize(logit(p_eval_all), quantiles[1:-1], right=False), 0, n_score_bins - 1)
    bin_fit = bin_all[idx_fit]
    y_fit = yv[idx_fit]
    y_conf = yv[idx_conf]

    bc_fit = np.zeros(y_fit.size, dtype=float)
    for b_id in range(n_score_bins):
        N1 = max(1, int(np.sum((yv == 1) & (bin_all == b_id))))
        N0 = max(1, int(np.sum((yv == 0) & (bin_all == b_id))))
        n1_f = max(0.5, float(np.sum((y_fit == 1) & (bin_fit == b_id))))
        n0_f = max(0.5, float(np.sum((y_fit == 0) & (bin_fit == b_id))))
        bc_fit[bin_fit == b_id] = float(np.log((n1_f / float(N1)) / (n0_f / float(N0))))

    dkl_conf = bernoulli_kl_from_tau(p_eval_all[idx_conf], tau)
    kappa_eff = solve_ess_kappa(
        dkl_conf,
        kappa_max=kappa,
        min_effective_cz_samples=max(80.0, 0.24 * float(len(idx_conf))),
    )

    enc_b = FastFrameEncoder().fit(df_train.iloc[idx_fit][base_cols])
    xb_fit = enc_b.transform(df_train.iloc[idx_fit][base_cols])
    xb_conf = enc_b.transform(df_train.iloc[idx_conf][base_cols])
    enc_z_map = {
        ch.channel_id: FastFrameEncoder().fit(df_train.iloc[idx_fit][list(ch.columns)]) for ch in channels
    }
    z_fit_map = {
        ch.channel_id: enc_z_map[ch.channel_id].transform(df_train.iloc[idx_fit][list(ch.columns)]) for ch in channels
    }
    z_conf_map = {
        ch.channel_id: enc_z_map[ch.channel_id].transform(df_train.iloc[idx_conf][list(ch.columns)]) for ch in channels
    }

    init_fit = logit(p_fit_all[idx_fit]) + bc_fit
    init_conf_eval = logit(p_fit_all[idx_conf]) + prior_eval_minus_fit[idx_conf]
    n_jobs = int(os.environ.get("UPCA_LGBM_JOBS", "4"))

    def _fit_conf_probs(sel_ids: list[str], step_seed: int) -> np.ndarray:
        xf_c = np.hstack([xb_fit] + [z_fit_map[cid] for cid in sel_ids]) if sel_ids else xb_fit
        xc_c = np.hstack([xb_conf] + [z_conf_map[cid] for cid in sel_ids]) if sel_ids else xb_conf
        clf = lgb.LGBMClassifier(
            n_estimators=85,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=max(15, min(20, len(idx_fit) // 25)),
            max_delta_step=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=step_seed,
            n_jobs=n_jobs,
            verbose=-1,
        ).fit(xf_c, y_fit, init_score=init_fit)
        raw_va = init_conf_eval + clf.predict(xc_c, raw_score=True)
        return np.clip(expit(np.clip(raw_va, -22.0, 22.0)), 1e-6, 1.0 - 1e-6)

    w_pop_conf = (
        np.ones(y_conf.size, dtype=float)
        if target_pop_weights is None
        else np.asarray(target_pop_weights, dtype=float)[idx_conf]
    )
    pop_scale = (
        float(pilot_conf.n_total) / max(1e-12, float(np.sum(w_conf * w_pop_conf)))
        if target_pop_weights is not None
        else 1.0
    )

    p_curr = _fit_conf_probs([], seed + 100)
    sc_curr = schervish_collision_proper_score(y_conf, p_curr, tau, kappa=kappa_eff)
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
        for idx_c, ch in enumerate(affordable):
            p_cand = _fit_conf_probs(selected + [ch.channel_id], seed + 100 * step + idx_c)
            sc_cand = schervish_collision_proper_score(y_conf, p_cand, tau, kappa=kappa_eff)
            psi = (sc_cand - sc_curr) * (w_pop_conf * pop_scale)
            mean_g, se_g = two_phase_ht_mean_and_se(psi, pilot_conf)
            z = mean_g / max(se_g, 1e-12)
            raw_pvals.append(float(1.0 - norm.cdf(z)))
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
            _ch_b, g_b, se_b, z_b, _ = cand_stats[best_i]
            steps_log.append(
                DiscoveryStepLog(step, None, float(g_b), float(se_b), float(z_b), float(p_holm[best_i]), 0.0)
            )
            break
        winner = max(passing, key=lambda item: item[1] / max(item[0].cost, 1e-6))
        ch_w, g_w, se_w, z_w, ph_w, sc_w = winner
        selected.append(ch_w.channel_id)
        spent += ch_w.cost
        sc_curr = sc_w
        steps_log.append(DiscoveryStepLog(step, ch_w.channel_id, float(g_w), float(se_w), float(z_w), ph_w, ch_w.cost))

    return (
        DiscoveryOutcome(
            "split_pilot_offset_cz_50_50",
            pilot_fraction,
            selected,
            spent,
            pilot_conf.high_signal_fp_share_of_controls,
            steps_log,
            step1_scores,
        ),
        pilot_conf,
    )


def oof_policy_screening_stage_b_scheduled(
    selected_ids: list[str],
    df_tr: pd.DataFrame,
    y_tr: np.ndarray,
    p1_tr_oof_sample: np.ndarray,
    base_cols: list[str],
    ch_by_id: dict[str, ChannelDef],
    contract: ClinicalContract,
    cost_schedule: ChannelCostSchedule,
    *,
    pi_target_override: float | None = None,
    dispersion_quantile: float = 0.75,
    alpha_pol: float = 0.10,
    seed: int = 42,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Stage B of UPCA-Fluid under explicit subset-dependent `ChannelCostSchedule` (`C(A)` per subset `A`)."""
    if len(selected_ids) == 0:
        return [], []

    z_crit = float(norm.ppf(1.0 - alpha_pol))
    yv = np.asarray(y_tr, dtype=int)
    pi_sample = float(np.mean(yv))
    w_tr = (
        np.ones(len(yv), dtype=float)
        if pi_target_override is None
        else case_control_prevalence_weights(yv, pi_target_override)
    )
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
            res0 = variable_cost_unit_nb_vec(yv, p1_eval_all, 0.0, contract)
            cache_nb[key] = res0
            return res0
        c_panel = cost_schedule.panel_cost_cfp(cids)
        ck_sub = ClinicalContract(
            cost_fp=contract.cost_fp,
            cost_fn=contract.cost_fn,
            target_precision=contract.target_precision,
            per_instance_query_cost=c_panel,
        )
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
            p2_fit_k = np.clip(apply_platt(raw_fit, platt), 1e-6, 1.0 - 1e-6)
            p2_cal_k = np.clip(apply_platt(raw_cal, platt), 1e-6, 1.0 - 1e-6)
            p2_va_k = np.clip(apply_platt(raw_va, platt), 1e-6, 1.0 - 1e-6)

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
                ck_sub,
                use_dispersion_2d=True,
                dispersion_quantile=dispersion_quantile,
                seed=seed + k,
            )
            q_va = voi_va > c_panel
            p2_oof[va_i] = np.where(q_va, p2_va_k, p1_eval_all[va_i])
            q_oof[va_i] = q_va

        res_vec = variable_cost_unit_nb_vec(yv, p2_oof, q_oof.astype(float) * c_panel, contract)
        cache_nb[key] = res_vec
        return res_vec

    def _weighted_z(diff_vec: np.ndarray) -> tuple[float, float]:
        w_sum = float(np.sum(w_tr))
        mean_d = float(np.sum(w_tr * diff_vec) / max(1e-12, w_sum))
        resid = diff_vec - mean_d
        se_d = float(np.sqrt(np.sum((w_tr**2) * (resid**2)) / max(1e-12, w_sum**2)))
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
            prune_log.append(
                {
                    "tested_drop": cid,
                    "active_size": len(active),
                    "solo_routed_gain": round(solo_gain, 4),
                    "z_solo": round(z_solo, 3),
                    "marginal_bundle_gain": round(marg_mean, 4),
                    "z_marg": round(z_marg, 3),
                    "pruned_candidate": bool(should_prune),
                }
            )
            if should_prune and (-marg_mean) > best_score:
                best_score = -marg_mean
                best_drop = (cid, red_nb)
        if best_drop is None:
            break
        cid_dropped, curr_nb = best_drop
        active.remove(cid_dropped)

    return active, prune_log


def evaluate_sequential_afa_scheduled(
    tr: pd.DataFrame,
    y_tr: np.ndarray,
    cal: pd.DataFrame,
    y_cal: np.ndarray,
    te: pd.DataFrame,
    y_te: np.ndarray,
    base_cols: list[str],
    ch_all: list[ChannelDef],
    p1_oof_raw: np.ndarray,
    p1_cal_raw: np.ndarray,
    p1_te_raw: np.ndarray,
    p1_cal_eval: np.ndarray,
    p1_te_eval: np.ndarray,
    contract: ClinicalContract,
    budget: float,
    cost_schedule: ChannelCostSchedule,
    *,
    policy_mode: str = "cmi",
    pi_target_override: float | None = None,
    w_cal: np.ndarray | None = None,
    w_te: np.ndarray | None = None,
    return_mask_predictor: bool = False,
    seed: int = 42,
) -> Any:
    """Matched-GBDT sequential greedy AFA policy-class reimplementation (`cmi` or `direct_voi`)
    billed strictly under `ChannelCostSchedule` (`C_i(M_i)` per patient mask `M_i`).

    Returns `(summary_dict, p_route_te, cost_cfp_te, mask_te)`.
    """
    m_ch = len(ch_all)
    ch_ids = [ch.channel_id for ch in ch_all]
    costs = np.array([float(ch.cost) for ch in ch_all], dtype=float)
    n_jobs = int(os.environ.get("UPCA_LGBM_JOBS", "4"))

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
        n_estimators=140,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        max_delta_step=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed + 41,
        n_jobs=n_jobs,
        verbose=-1,
    ).fit(np.vstack(x_reps), np.concatenate(y_reps), init_score=np.concatenate(i_reps))

    m_ca_r = _sample_masks(y_cal.size)
    raw_ca_m = logit(np.clip(p1_cal_raw, 1e-5, 1 - 1e-5)) + clf_m.predict(_masked(xb_ca, zf_ca, m_ca_r), raw_score=True)
    platt_m = fit_platt_calibrator(raw_ca_m, y_cal)
    pi_cal_s = float(np.mean(y_cal))

    def _pred_eval_p(
        xb: np.ndarray, zf: np.ndarray, p1_raw_v: np.ndarray, p1_eval_v: np.ndarray, mm: np.ndarray
    ) -> np.ndarray:
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
    p1_s_eval = (
        p1_s_raw
        if pi_target_override is None
        else adjust_probs_for_target_prevalence(p1_s_raw, float(np.mean(y_tr)), pi_target_override)
    )

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

    gain_regs: list[lgb.LGBMRegressor | None] = []
    for j in range(m_ch):
        valid = (st_m[:, j] < 0.5) & ((st_m @ costs) + costs[j] <= budget + 1e-9)
        if int(np.sum(valid)) < 40:
            gain_regs.append(None)
            continue
        m_plus = st_m[valid].copy()
        m_plus[:, j] = 1.0
        p_plus = _pred_eval_p(st_xb[valid], st_zf[valid], st_p1r[valid], st_p1e[valid], m_plus)
        pb = p_cur_st[valid]
        if policy_mode == "cmi":
            target_j = np.maximum(
                0.0, p_plus * np.log(p_plus / pb) + (1.0 - p_plus) * np.log((1.0 - p_plus) / (1.0 - pb))
            )
        elif policy_mode == "direct_voi":
            loss_cur = np.minimum(pb * contract.cost_fn, (1.0 - pb) * contract.cost_fp)
            loss_plus = np.minimum(p_plus * contract.cost_fn, (1.0 - p_plus) * contract.cost_fp)
            target_j = np.maximum(0.0, loss_cur - loss_plus)
        else:
            raise ValueError(f"Unknown AFA policy_mode: {policy_mode}")
        reg_j = lgb.LGBMRegressor(
            n_estimators=55,
            learning_rate=0.06,
            num_leaves=15,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed + 200 + j,
            n_jobs=n_jobs,
            verbose=-1,
        ).fit(x_st_feat[valid], target_j)
        gain_regs.append(reg_j)

    def _rollout(
        xb: np.ndarray, zf: np.ndarray, p1r: np.ndarray, p1e: np.ndarray, lam: float, max_steps: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            any_already = np.any(m_cur[act_i] > 0.5, axis=1)
            for j, rj in enumerate(gain_regs):
                if rj is None:
                    continue
                can = (m_cur[act_i, j] < 0.5) & (sp_cur[act_i] + costs[j] <= budget + 1e-9)
                if np.any(can):
                    marg_cfp = np.where(
                        any_already[can],
                        cost_schedule.c_unit * costs[j],
                        cost_schedule.c_overhead + cost_schedule.c_unit * costs[j],
                    )
                    ratios[can, j] = np.maximum(0.0, rj.predict(x_st[can])) / np.maximum(1e-6, marg_cfp)
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
    cand_lams = (
        np.unique(np.concatenate([np.quantile(pos_r, [0.10, 0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 0.96]), [1e9]]))
        if pos_r.size >= 10
        else np.array([1e-4, 1e-3, 1e-2, 1e9])
    )

    best_lam = float(cand_lams[-1])
    best_nb = -np.inf
    for lam in cand_lams:
        pc_t, _, mc_t, _ = _rollout(xb_ca, zf_ca, p1_cal_raw, p1_cal_eval, float(lam))
        cc_t = cost_schedule.mask_costs_cfp(mc_t, ch_ids)
        nb_t = variable_cost_nb_per_1k(y_cal, pc_t, cc_t, contract, sample_weight=w_cal)
        if nb_t > best_nb:
            best_nb = nb_t
            best_lam = float(lam)

    p_rt, q_t, m_te, _ = _rollout(xb_te, zf_te, p1_te_raw, p1_te_eval, best_lam)
    p_full, _, _, _ = _rollout(xb_te, zf_te, p1_te_raw, p1_te_eval, 0.0)
    cost_te = cost_schedule.mask_costs_cfp(m_te, ch_ids)
    eval_prev = float(pi_target_override) if pi_target_override is not None else float(np.mean(y_te))

    nb_rt = variable_cost_nb_per_1k(y_te, p_rt, cost_te, contract, sample_weight=w_te)
    q_rate = float(np.mean(q_t)) if w_te is None else float(np.sum(w_te * q_t) / np.sum(w_te))
    mean_cost_cfp = float(np.mean(cost_te)) if w_te is None else float(np.sum(w_te * cost_te) / np.sum(w_te))
    null_idxs = [j for j, ch in enumerate(ch_all) if ch.channel_id.startswith("ch_null")]
    null_q_rate = float(np.mean(np.any(m_te[:, null_idxs] > 0.5, axis=1))) if null_idxs else 0.0
    top_ch_freqs = {
        ch_all[j].channel_id: round(float(np.mean(m_te[:, j] > 0.5)), 4)
        for j in range(m_ch)
        if float(np.mean(m_te[:, j] > 0.5)) > 0.005
    }

    summary = {
        "policy_mode": f"matched_gbdt_greedy_{policy_mode}_channel_billed",
        "selected_top3": sorted(top_ch_freqs.keys(), key=lambda k: top_ch_freqs[k], reverse=True)[:3],
        "channel_query_frequencies": top_ch_freqs,
        "null_channel_query_rate": round(null_q_rate, 4),
        "nb_routed": round(nb_rt, 4),
        "snb_routed": round(standardized_net_benefit(nb_rt, eval_prev), 4),
        "query_rate": round(q_rate, 4),
        "mean_cost_cfp": round(mean_cost_cfp, 5),
        "auprc_full": round(float(average_precision_score(y_te, p_full, sample_weight=w_te)), 4),
        "auprc_routed": round(float(average_precision_score(y_te, p_rt, sample_weight=w_te)), 4),
        "auroc_full": round(float(roc_auc_score(y_te, p_full, sample_weight=w_te)), 4),
    }
    if return_mask_predictor:
        def _predict_te_for_mask(mm_te: np.ndarray) -> np.ndarray:
            return _pred_eval_p(xb_te, zf_te, p1_te_raw, p1_te_eval, mm_te.astype(np.float32))

        return summary, p_rt, cost_te, m_te, _predict_te_for_mask
    return summary, p_rt, cost_te, m_te


def evaluate_dual_afa_scheduled(
    tr: pd.DataFrame,
    y_tr: np.ndarray,
    cal: pd.DataFrame,
    y_cal: np.ndarray,
    te: pd.DataFrame,
    y_te: np.ndarray,
    base_cols: list[str],
    ch_all: list[ChannelDef],
    p1_oof_raw: np.ndarray,
    p1_cal_raw: np.ndarray,
    p1_te_raw: np.ndarray,
    p1_cal_eval: np.ndarray,
    p1_te_eval: np.ndarray,
    contract: ClinicalContract,
    budget: float,
    cost_schedule: ChannelCostSchedule,
    *,
    pi_target_override: float | None = None,
    w_cal: np.ndarray | None = None,
    w_te: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit shared masked GBDT predictor ONCE and evaluate both `cmi` and `direct_voi` policies
    with picklable state for exact `R=10` majority-mask evaluation.
    """
    m_ch = len(ch_all)
    ch_ids = [ch.channel_id for ch in ch_all]
    costs = np.array([float(ch.cost) for ch in ch_all], dtype=float)
    n_jobs = int(os.environ.get("UPCA_LGBM_JOBS", "2"))

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
        n_estimators=140,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        max_delta_step=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed + 41,
        n_jobs=n_jobs,
        verbose=-1,
    ).fit(np.vstack(x_reps), np.concatenate(y_reps), init_score=np.concatenate(i_reps))

    m_ca_r = _sample_masks(y_cal.size)
    raw_ca_m = logit(np.clip(p1_cal_raw, 1e-5, 1 - 1e-5)) + clf_m.predict(_masked(xb_ca, zf_ca, m_ca_r), raw_score=True)
    platt_m = fit_platt_calibrator(raw_ca_m, y_cal)
    pi_cal_s = float(np.mean(y_cal))

    def _pred_eval_p(
        xb: np.ndarray, zf: np.ndarray, p1_raw_v: np.ndarray, p1_eval_v: np.ndarray, mm: np.ndarray
    ) -> np.ndarray:
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
    p1_s_eval = (
        p1_s_raw
        if pi_target_override is None
        else adjust_probs_for_target_prevalence(p1_s_raw, float(np.mean(y_tr)), pi_target_override)
    )

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

    # Precompute p_plus once per channel j and share across both `cmi` and `direct_voi`!
    p_plus_by_j: list[tuple[np.ndarray, np.ndarray] | None] = []
    for j in range(m_ch):
        valid = (st_m[:, j] < 0.5) & ((st_m @ costs) + costs[j] <= budget + 1e-9)
        if int(np.sum(valid)) < 40:
            p_plus_by_j.append(None)
            continue
        m_plus = st_m[valid].copy()
        m_plus[:, j] = 1.0
        p_plus = _pred_eval_p(st_xb[valid], st_zf[valid], st_p1r[valid], st_p1e[valid], m_plus)
        p_plus_by_j.append((valid, p_plus))

    out_policies: dict[str, Any] = {}
    eval_prev = float(pi_target_override) if pi_target_override is not None else float(np.mean(y_te))
    null_idxs = [j for j, ch in enumerate(ch_all) if ch.channel_id.startswith("ch_null")]

    for policy_mode in ("cmi", "direct_voi"):
        gain_regs: list[lgb.LGBMRegressor | None] = []
        for j in range(m_ch):
            item_j = p_plus_by_j[j]
            if item_j is None:
                gain_regs.append(None)
                continue
            valid, p_plus = item_j
            pb = p_cur_st[valid]
            if policy_mode == "cmi":
                target_j = np.maximum(
                    0.0, p_plus * np.log(p_plus / pb) + (1.0 - p_plus) * np.log((1.0 - p_plus) / (1.0 - pb))
                )
            else:
                loss_cur = np.minimum(pb * contract.cost_fn, (1.0 - pb) * contract.cost_fp)
                loss_plus = np.minimum(p_plus * contract.cost_fn, (1.0 - p_plus) * contract.cost_fp)
                target_j = np.maximum(0.0, loss_cur - loss_plus)
            reg_j = lgb.LGBMRegressor(
                n_estimators=55,
                learning_rate=0.06,
                num_leaves=15,
                min_child_samples=25,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed + 200 + j,
                n_jobs=n_jobs,
                verbose=-1,
            ).fit(x_st_feat[valid], target_j)
            gain_regs.append(reg_j)

        def _rollout(
            xb: np.ndarray,
            zf: np.ndarray,
            p1r: np.ndarray,
            p1e: np.ndarray,
            lam: float,
            _regs: list[lgb.LGBMRegressor | None] = gain_regs,
            max_steps: int = 3,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                any_already = np.any(m_cur[act_i] > 0.5, axis=1)
                for j, rj in enumerate(_regs):
                    if rj is None:
                        continue
                    can = (m_cur[act_i, j] < 0.5) & (sp_cur[act_i] + costs[j] <= budget + 1e-9)
                    if np.any(can):
                        marg_cfp = np.where(
                            any_already[can],
                            cost_schedule.c_unit * costs[j],
                            cost_schedule.c_overhead + cost_schedule.c_unit * costs[j],
                        )
                        ratios[can, j] = np.maximum(0.0, rj.predict(x_st[can])) / np.maximum(1e-6, marg_cfp)
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
        cand_lams = (
            np.unique(np.concatenate([np.quantile(pos_r, [0.10, 0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 0.96]), [1e9]]))
            if pos_r.size >= 10
            else np.array([1e-4, 1e-3, 1e-2, 1e9])
        )
        best_lam = float(cand_lams[-1])
        best_nb = -np.inf
        for lam in cand_lams:
            pc_t, _, mc_t, _ = _rollout(xb_ca, zf_ca, p1_cal_raw, p1_cal_eval, float(lam))
            cc_t = cost_schedule.mask_costs_cfp(mc_t, ch_ids)
            nb_t = variable_cost_nb_per_1k(y_cal, pc_t, cc_t, contract, sample_weight=w_cal)
            if nb_t > best_nb:
                best_nb = nb_t
                best_lam = float(lam)

        p_rt, q_t, m_te, _ = _rollout(xb_te, zf_te, p1_te_raw, p1_te_eval, best_lam)
        cost_te = cost_schedule.mask_costs_cfp(m_te, ch_ids)
        nb_rt = variable_cost_nb_per_1k(y_te, p_rt, cost_te, contract, sample_weight=w_te)
        q_rate = float(np.mean(q_t)) if w_te is None else float(np.sum(w_te * q_t) / np.sum(w_te))
        null_q_rate = float(np.mean(np.any(m_te[:, null_idxs] > 0.5, axis=1))) if null_idxs else 0.0
        out_policies[policy_mode] = {
            "nb_routed": round(nb_rt, 4),
            "snb_routed": round(standardized_net_benefit(nb_rt, eval_prev), 4),
            "query_rate": round(q_rate, 4),
            "null_channel_query_rate": round(null_q_rate, 4),
            "p_rt": p_rt,
            "cost_te": cost_te,
            "mask_te": m_te,
        }

    out_policies["masked_predictor_state"] = {
        "booster_str": clf_m.booster_.model_to_string(),
        "platt_m": platt_m,
        "pi_cal_s": pi_cal_s,
        "pi_target_override": pi_target_override,
        "xb_te": xb_te,
        "zf_te": zf_te,
        "ch_slices": ch_slices,
        "p1_te_raw": p1_te_raw,
        "p1_te_eval": p1_te_eval,
    }
    return out_policies


def predict_masked_te_from_state(state: dict[str, Any], mask_te: np.ndarray) -> np.ndarray:
    """Evaluate a pickled `evaluate_dual_afa_scheduled` masked booster state on any `(N, M)` binary mask."""
    mm = np.asarray(mask_te, dtype=np.float32)
    has_any = np.any(mm > 0.5, axis=1)
    out = state["p1_te_eval"].copy()
    if not np.any(has_any):
        return out
    xb = state["xb_te"][has_any]
    zf = state["zf_te"][has_any]
    m_sub = mm[has_any]
    zm = np.full_like(zf, np.float32(-999.0))
    for j, (c0, c1) in enumerate(state["ch_slices"]):
        obs = m_sub[:, j] > 0.5
        if np.any(obs):
            zm[obs, c0:c1] = zf[obs, c0:c1]
    xm = np.hstack([xb, zm, m_sub])
    booster = lgb.Booster(model_str=state["booster_str"])
    raw = logit(np.clip(state["p1_te_raw"][has_any], 1e-5, 1 - 1e-5)) + booster.predict(xm)
    p_cal_sc = np.clip(apply_platt(raw, state["platt_m"]), 1e-6, 1.0 - 1e-6)
    if state["pi_target_override"] is not None:
        p_cal_sc = adjust_probs_for_target_prevalence(p_cal_sc, state["pi_cal_s"], state["pi_target_override"])
    out[has_any] = p_cal_sc
    return out


def compute_fluid_specialist_diagnostics(
    clf: FluidCollisionBackboneClassifier,
    x2_te: np.ndarray,
    z_te: np.ndarray,
    ell1_te: np.ndarray,
    tau_star_eval: float,
    *,
    outer_platt: tuple[float, float] = (1.0, 0.0),
    prior_shift_log_odds: float = 0.0,
) -> dict[str, float]:
    """Compute scale-invariant RMS specialist contribution `R_CZ` and threshold decision-flip fraction."""
    if clf.m_glob_ is None:
        return {
            "beta_cz": 0.0,
            "kappa_star": 0.0,
            "rms_r_cz": 0.0,
            "rms_r_cz_zone": 0.0,
            "decision_flip_rate_vs_glob": 0.0,
        }
    ell1 = np.clip(np.asarray(ell1_te, dtype=float), -20.0, 20.0)
    a_g, b_g = clf.platt_glob_
    f_glob = np.clip(a_g * clf.m_glob_.predict(x2_te, raw_score=True) + b_g, -20.0, 20.0)
    f_glob_only = float(clf.alpha_0_) + float(clf.beta_glob_) * f_glob

    if clf.m_cz_ is None or clf.beta_cz_ <= 1e-4:
        return {
            "beta_cz": round(float(clf.beta_cz_), 4),
            "kappa_star": round(float(clf.kappa_), 3),
            "rms_r_cz": 0.0,
            "rms_r_cz_zone": 0.0,
            "decision_flip_rate_vs_glob": 0.0,
        }

    pi_u = union_collision_gate(ell1, f_glob, clf.tau_star, clf.kappa_)
    cz_in = np.column_stack([ell1, f_glob, z_te]).astype(np.float32)
    d_cz = clf.m_cz_.predict(cz_in, raw_score=True)
    a_out, b_out = outer_platt
    contrib = float(a_out) * float(clf.beta_cz_) * pi_u * d_cz
    rms_all = float(np.sqrt(np.mean(contrib**2)))
    in_zone = pi_u >= 0.25
    rms_zone = float(np.sqrt(np.mean(contrib[in_zone] ** 2))) if int(np.sum(in_zone)) >= 5 else rms_all

    logit_tau = float(logit(np.clip(tau_star_eval, 1e-6, 1.0 - 1e-6)))
    cal_glob = float(a_out) * f_glob_only + float(b_out) + float(prior_shift_log_odds)
    cal_full = cal_glob + contrib
    flips = float(np.mean((cal_full >= logit_tau) != (cal_glob >= logit_tau)))
    return {
        "beta_cz": round(float(clf.beta_cz_), 4),
        "kappa_star": round(float(clf.kappa_), 3),
        "rms_r_cz": round(rms_all, 4),
        "rms_r_cz_zone": round(rms_zone, 4),
        "decision_flip_rate_vs_glob": round(flips, 4),
    }


def boundary_uncertainty_mask(p1: np.ndarray, tau_star: float, target_query_rate: float) -> np.ndarray:
    """Query the `target_query_rate` fraction of encounters closest to `|p1 - tau*| = 0`."""
    q = float(np.clip(target_query_rate, 0.0, 1.0))
    n = len(p1)
    k = round(q * n)
    if k <= 0:
        return np.zeros(n, dtype=bool)
    if k >= n:
        return np.ones(n, dtype=bool)
    dist = np.abs(np.asarray(p1, dtype=float) - float(tau_star))
    order = np.argsort(dist, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def directional_voi_masks(
    p1: np.ndarray,
    voi: np.ndarray,
    panel_cost_cfp: float,
    tau_star: float,
) -> dict[str, np.ndarray]:
    """Return Two-Sided, Escalation-Only (`p1 < tau*`), and De-escalation-Only (`p1 >= tau*`) VOI masks."""
    pv = np.asarray(p1, dtype=float)
    vv = np.asarray(voi, dtype=float)
    c = float(panel_cost_cfp)
    return {
        "two_sided": vv > c,
        "escalation_only": (pv < tau_star) & (vv > c),
        "deescalation_only": (pv >= tau_star) & (vv > c),
    }


def saerens_em_prevalence_estimate(
    p_src_on_tgt: np.ndarray,
    pi_src: float,
    *,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> float:
    """Purely unlabeled Saerens, Latinne & Decaestecker (2002) EM target prior estimator (`0` target labels)."""
    pv = np.clip(np.asarray(p_src_on_tgt, dtype=float), 1e-6, 1.0 - 1e-6)
    ps = float(np.clip(pi_src, 1e-6, 1.0 - 1e-6))
    pi_cur = ps
    for _ in range(max_iter):
        r1 = (pi_cur / ps) * pv
        r0 = ((1.0 - pi_cur) / (1.0 - ps)) * (1.0 - pv)
        post = r1 / np.maximum(1e-12, r1 + r0)
        pi_next = float(np.clip(np.mean(post), 1e-5, 1.0 - 1e-5))
        if abs(pi_next - pi_cur) < tol:
            pi_cur = pi_next
            break
        pi_cur = pi_next
    return float(pi_cur)


def cochran_q_subgroup_heterogeneity(
    estimates: list[float],
    standard_errors: list[float],
) -> dict[str, float]:
    """Inverse-variance Cochran's Q test for subgroup effect heterogeneity."""
    est = np.asarray(estimates, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    valid = np.isfinite(est) & np.isfinite(se) & (se > 1e-9)
    est, se = est[valid], se[valid]
    k = int(est.size)
    if k < 2:
        return {"q_stat": 0.0, "df": 0, "p_het": 1.0, "i2_pct": 0.0}
    w = 1.0 / (se**2)
    mu_fixed = float(np.sum(w * est) / np.sum(w))
    q_val = float(np.sum(w * ((est - mu_fixed) ** 2)))
    df = k - 1
    p_val = float(1.0 - chi2.cdf(q_val, df=df))
    i2 = max(0.0, 100.0 * (q_val - df) / max(1e-9, q_val))
    return {
        "q_stat": round(q_val, 4),
        "df": int(df),
        "p_het": round(p_val, 5),
        "i2_pct": round(i2, 2),
    }

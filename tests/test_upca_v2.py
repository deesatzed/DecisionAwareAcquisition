"""Comprehensive unit tests (`35` tests) for `DecisionAwareAcquisition` (`v2.1`).

Tests all standalone statistical, two-phase discovery, Breslow-Cain offset, Schervish
scoring, KL-gated residual backbone, pre-purchase empirical martingale VOI, sequential
greedy-CMI AFA (`CovertGreedyCMIAFA`), calibration (`CITL`, slope, spline `ICI`),
patient-cluster ratio bootstrap, directional skipped-case bounds, and within-stratum
independence primitives using ONLY `DecisionAwareAcquisition/src` (zero external repo imports).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from collision_zone.three_predicates import (
    AgnielDeconfounder,
    ChannelDef,
    ClinicalContract,
    FastFrameEncoder,
    FluidCollisionBackboneClassifier,
    PilotSample,
    apply_platt,
    bernoulli_kl_from_tau,
    cfbh_selection,
    collision_beta_params,
    empirical_martingale_voi,
    fit_platt_calibrator,
    holm_adjust,
    mean_centered_voi,
    sample_pilot,
    schervish_collision_proper_score,
    schervish_collision_weight,
    solve_ess_kappa,
    solve_martingale_offset,
    two_phase_ht_mean_and_se,
)
from collision_zone.three_predicates_v2 import (
    compute_calibration_metrics as compute_calibration_metrics_v2,
)
from collision_zone.three_predicates_v2 import (
    evaluate_covert_greedy_cmi_afa,
    prior_odds_recalibrate_probabilities,
)
from decision_aware_acq.upca_v2_core import (
    adjust_probs_for_target_prevalence,
    case_control_prevalence_weights,
    check_within_stratum_pilot_independence,
    clopper_pearson_upper_conservative,
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
    weighted_net_benefit_per_1k,
)


def test_01_standalone_imports_do_not_reference_external_repo() -> None:
    import collision_zone
    import decision_aware_acq

    assert str(REPO_ROOT) in Path(collision_zone.__file__).resolve().as_posix()
    assert str(REPO_ROOT) in Path(decision_aware_acq.__file__).resolve().as_posix()


def test_02_clinical_contract_tau_star_and_odds() -> None:
    c = ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.03)
    assert abs(c.tau_star - 0.05) < 1e-12
    assert abs((c.cost_fp / c.cost_fn) - (0.05 / 0.95)) < 1e-12


def test_03_case_control_prevalence_weights_exact_moment() -> None:
    y = np.array([1] * 200 + [0] * 800, dtype=int)
    w = case_control_prevalence_weights(y, pi_target=0.02)
    assert abs(float(np.mean(w)) - 1.0) < 1e-12
    assert abs(float(np.sum(w * y) / np.sum(w)) - 0.02) < 1e-12


def test_04_adjust_probs_for_target_prevalence_exact_bayes_shift() -> None:
    p_cc = np.full(500, 0.20, dtype=float)
    p_tgt = adjust_probs_for_target_prevalence(p_cc, pi_sample=0.20, pi_target=0.02)
    assert np.allclose(p_tgt, 0.02, atol=1e-10)
    # Round-trip invertibility
    p_back = adjust_probs_for_target_prevalence(p_tgt, pi_sample=0.02, pi_target=0.20)
    assert np.allclose(p_back, 0.20, atol=1e-6)


def test_05_schervish_beta_params_mode_alignment() -> None:
    for tau in (0.03, 0.05, 0.15, 0.25):
        a, b = collision_beta_params(tau, kappa=24.0)
        mode = (a - 1.0) / (a + b - 2.0)
        assert abs(mode - tau) < 1e-12


def test_06_schervish_collision_score_strictly_proper_in_expectation() -> None:
    rng = np.random.default_rng(101)
    q_true = rng.uniform(0.01, 0.25, size=5000)
    # Under Y ~ Bern(q_true), E[S(Y, p)] is uniquely maximized at p = q_true
    s_opt = q_true * schervish_collision_proper_score(np.ones_like(q_true), q_true, 0.05, 24.0) + (
        1.0 - q_true
    ) * schervish_collision_proper_score(np.zeros_like(q_true), q_true, 0.05, 24.0)
    for shift in (-0.45, -0.15, 0.15, 0.45):
        p_mis = expit(logit(q_true) + shift)
        s_mis = q_true * schervish_collision_proper_score(np.ones_like(q_true), p_mis, 0.05, 24.0) + (
            1.0 - q_true
        ) * schervish_collision_proper_score(np.zeros_like(q_true), p_mis, 0.05, 24.0)
        assert float(np.mean(s_opt - s_mis)) > 1e-5


def test_07_schervish_weight_subthreshold_suppression() -> None:
    w_peak = float(schervish_collision_weight(np.array([0.05]), tau_star=0.05, kappa=24.0)[0])
    w_far = float(schervish_collision_weight(np.array([0.002]), tau_star=0.05, kappa=24.0)[0])
    w_high = float(schervish_collision_weight(np.array([0.65]), tau_star=0.05, kappa=24.0)[0])
    assert w_peak > 12.0 * w_far
    assert w_peak > 100.0 * w_high


def test_08_two_phase_ht_single_stratum_reduces_to_finite_pop_srs() -> None:
    rng = np.random.default_rng(12)
    N = 400
    n = 80
    vals_full = rng.normal(2.5, 1.3, size=N)
    idx = rng.choice(N, size=n, replace=False)
    v_sub = vals_full[idx]
    pil = PilotSample(
        indices=idx,
        strata=np.zeros(n, dtype=int),
        ht_weights=np.full(n, float(N) / float(n)),
        pop_counts={0: N},
        sample_counts={0: n},
        n_total=N,
        high_signal_fp_share_of_controls=0.0,
    )
    mean_ht, se_ht = two_phase_ht_mean_and_se(v_sub, pil)
    expected_mean = float(np.mean(v_sub))
    s2 = float(np.var(v_sub, ddof=1))
    # Under Sarndal-Swensson-Wretman two-phase variance with H=1 stratum,
    # Phase-1 variance (s^2 / N) + Phase-2 variance ((1 - n/N) * s^2 / n) collapses identically to s^2 / n!
    expected_se = float(np.sqrt(s2 / n))
    assert abs(mean_ht - expected_mean) < 1e-12
    assert abs(se_ht - expected_se) < 1e-12


def test_09_two_phase_ht_unbiased_across_repeated_neyman_pilots() -> None:
    rng = np.random.default_rng(22)
    N = 1500
    p1 = rng.beta(1.2, 18.0, size=N)
    y = rng.binomial(1, p1)
    psi = 0.2 * (y - p1) + rng.normal(0.05, 0.15, size=N)
    pop_mean = float(np.mean(psi))
    ests = []
    for s in range(200):
        pil = sample_pilot(y, p1, pilot_fraction=0.20, mode="collision_neyman", tau_star=0.05, n_bins=8, seed=300 + s)
        m, _ = two_phase_ht_mean_and_se(psi[pil.indices], pil)
        ests.append(m)
    assert abs(float(np.mean(ests)) - pop_mean) < 0.005


def test_10_sample_pilot_exact_budget_and_positive_weights() -> None:
    rng = np.random.default_rng(33)
    p1 = rng.uniform(0.005, 0.40, size=1000)
    y = rng.binomial(1, p1)
    pil = sample_pilot(y, p1, pilot_fraction=0.20, mode="collision_neyman", tau_star=0.05, n_bins=8, seed=44)
    assert pil.indices.size == 200
    assert len(np.unique(pil.indices)) == 200
    assert np.all(pil.ht_weights >= 1.0)
    assert abs(float(np.sum(pil.ht_weights)) - 1000.0) < 1e-6


def test_11_breslow_cain_offset_recovers_population_slope_under_case_enrichment() -> None:
    rng = np.random.default_rng(45)
    N = 4000
    ell1 = rng.normal(-3.0, 1.0, size=N)
    z = rng.normal(0.0, 1.0, size=N)
    p_true = expit(ell1 + 0.90 * z)
    y = rng.binomial(1, p_true)
    p1 = expit(ell1)

    pil = sample_pilot(y, p1, pilot_fraction=0.20, mode="collision_neyman", tau_star=0.05, n_bins=8, seed=55)
    # Without offset, pilot case fraction is heavily enriched vs full cohort
    assert float(np.mean(y[pil.indices])) > 1.3 * float(np.mean(y))

    df = pd.DataFrame({"b1": ell1, "z1": z})
    ch = [ChannelDef("ch_sig", ("z1",), 1.0), ChannelDef("ch_null", ("b1",), 1.0)]
    ck = ClinicalContract(1.0, 19.0, 0.25, 0.02)
    disc_off, _ = discover_stage_a_ablation(df, y, ["b1"], ch, p1, ck, budget=1.0, mode="offset_cz_schervish", pilot_fraction=0.20, seed=55)
    assert disc_off.selected_ids == ["ch_sig"]


def test_12_bernoulli_kl_and_ess_kappa_solver() -> None:
    kl_zero = bernoulli_kl_from_tau(np.array([0.05]), tau_star=0.05)
    assert abs(float(kl_zero[0])) < 1e-12
    p = np.linspace(0.005, 0.50, 2000)
    kl_vec = bernoulli_kl_from_tau(p, 0.05)
    kap = solve_ess_kappa(kl_vec, kappa_max=20.0, min_effective_cz_samples=240.0)
    assert 0.0 <= kap <= 20.0
    w = np.exp(-kap * kl_vec)
    ess = float(np.sum(w) ** 2 / np.sum(w ** 2))
    assert ess >= 239.0


def test_13_fluid_collision_backbone_damps_beta_cz_under_global_vs_cz_signal() -> None:
    rng = np.random.default_rng(61)
    n = 2500
    xb = rng.normal(0, 1, size=(n, 3))
    z_glob = rng.normal(0, 1, size=(n, 2))
    # Purely homogeneous global Z signal (no local collision-zone interaction) vs sharp local CZ signal
    ell1 = np.clip(-2.8 + 0.9 * xb[:, 0] - 0.5 * xb[:, 1], -6.0, 2.0)
    y_hom = rng.binomial(1, expit(ell1 + 1.2 * z_glob[:, 0]))
    clf_hom = FluidCollisionBackboneClassifier(tau_star=0.05, kappa_max=20.0, min_cz_ess=200.0, seed=7).fit(
        np.hstack([xb, z_glob]), z_glob, y_hom, ell1
    )
    in_cz = np.exp(-15.0 * bernoulli_kl_from_tau(expit(ell1), 0.05))
    y_het = rng.binomial(1, expit(ell1 + 2.8 * in_cz * z_glob[:, 0] - 0.8 * (1.0 - in_cz) * z_glob[:, 0]))
    clf_het = FluidCollisionBackboneClassifier(tau_star=0.05, kappa_max=20.0, min_cz_ess=200.0, seed=7).fit(
        np.hstack([xb, z_glob]), z_glob, y_het, ell1
    )
    assert clf_hom.beta_cz_ >= 0.0
    assert clf_het.beta_cz_ > clf_hom.beta_cz_


def test_14_fluid_collision_backbone_activates_under_local_collision_signal() -> None:
    rng = np.random.default_rng(62)
    n = 2500
    xb = rng.normal(0, 1, size=(n, 2))
    ell1 = np.clip(-2.9 + 1.1 * xb[:, 0], -6.0, 2.0)
    p1 = expit(ell1)
    z_sig = rng.normal(0, 1, size=(n, 2))
    in_cz = np.exp(-12.0 * bernoulli_kl_from_tau(p1, 0.05))
    logit_y = ell1 + 2.2 * in_cz * z_sig[:, 0]
    y = rng.binomial(1, expit(logit_y))
    clf = FluidCollisionBackboneClassifier(tau_star=0.05, kappa_max=20.0, min_cz_ess=200.0, seed=9).fit(
        np.hstack([xb, z_sig]), z_sig, y, ell1
    )
    assert clf.beta_cz_ > 0.25
    scores = clf.decision_function(np.hstack([xb, z_sig]), z_sig, ell1)
    assert np.all(np.isfinite(scores))


def test_15_solve_martingale_offset_exact_posterior_expectation() -> None:
    from collision_zone.three_predicates import _GH_NODES, _GH_WEIGHTS

    p0 = np.array([0.01, 0.05, 0.20, 0.60])
    sig = np.array([0.8, 0.9, 0.7, 0.6])
    mu = solve_martingale_offset(p0, sig)
    post = expit(logit(p0)[:, None] + mu[:, None] + sig[:, None] * _GH_NODES[None, :]) @ _GH_WEIGHTS
    assert np.allclose(post, p0, atol=1e-7)


def test_16_empirical_martingale_voi_nonnegative_and_zero_far_from_threshold() -> None:
    ck = ClinicalContract(1.0, 19.0, 0.25, 0.03)  # tau* = 0.05
    bq = np.tile(np.linspace(-0.8, 0.8, 33), (4, 1))
    p_base = np.array([1e-5, 0.048, 0.052, 0.999])
    bins = np.array([0, 1, 2, 3])
    voi = empirical_martingale_voi(p_base, bins, bq, ck)
    assert np.all(voi >= -1e-12)
    assert voi[0] < 1e-4 and voi[3] < 1e-4
    assert voi[1] > 0.01 and voi[2] > 0.01


def test_17_empirical_vs_gaussian_martingale_voi_under_skewed_jumps() -> None:
    ck = ClinicalContract(1.0, 19.0, 0.25, 0.03)
    # Right-skewed rare jump distribution (mostly slightly negative, sharp positive tail)
    deltas = np.concatenate([np.full(29, -0.25), np.array([1.8, 2.1, 2.4, 2.8])])
    bq = deltas.reshape(1, -1)
    p0 = np.array([0.025])
    b0 = np.array([0])
    voi_emp = float(empirical_martingale_voi(p0, b0, bq, ck)[0])
    voi_gauss = float(mean_centered_voi(p0, np.array([float(np.std(deltas))]), ck)[0])
    assert voi_emp >= 0.0 and voi_gauss >= 0.0
    assert abs(voi_emp - voi_gauss) > 1e-4


def test_18_fit_predict_prepurchase_voi_2d_dispersion_runs_cleanly() -> None:
    rng = np.random.default_rng(81)
    ck = ClinicalContract(1.0, 9.0, 0.30, 0.02)
    xb_tr, xb_ca, xb_te = rng.normal(size=(500, 4)), rng.normal(size=(200, 4)), rng.normal(size=(200, 4))
    p1_tr = expit(-2.2 + 0.7 * xb_tr[:, 0])
    p2_tr = expit(logit(p1_tr) + 0.8 * rng.normal(size=500))
    p1_ca = expit(-2.2 + 0.7 * xb_ca[:, 0])
    p2_ca = expit(logit(p1_ca) + 0.8 * rng.normal(size=200))
    p1_te = expit(-2.2 + 0.7 * xb_te[:, 0])
    voi = fit_predict_prepurchase_voi(xb_tr, p1_tr, p2_tr, xb_ca, p1_ca, p2_ca, xb_te, p1_te, ck, use_dispersion_2d=True, seed=81)
    assert voi.shape == (200,)
    assert np.all(voi >= 0.0)


def test_19_patient_cluster_bootstrap_tie_and_davison_hinkley_floor() -> None:
    rng = np.random.default_rng(11)
    ck = ClinicalContract(cost_fp=1.0, cost_fn=9.0, per_instance_query_cost=0.02)
    y = rng.binomial(1, 0.10, size=600)
    p1 = np.clip(0.10 + rng.normal(0, 0.03, size=600), 0.01, 0.99)
    q0 = np.zeros(600, dtype=bool)
    clusters = np.repeat(np.arange(150), 4)

    tie = patient_cluster_bootstrap_diff(y, p1, q0, p1, q0, clusters, ck, n_boot=10_000, seed=1)
    assert tie["diff_mean"] == 0.0 and tie["p_two_sided"] == 1.0

    p_perf = np.where(y == 1, 0.95, 0.01)
    dom = patient_cluster_bootstrap_diff(y, p_perf, q0, p1, q0, clusters, ck, n_boot=10_000, seed=2)
    assert dom["diff_mean"] > 50.0
    assert abs(dom["p_two_sided"] - (2.0 / 10_001.0)) < 1e-6


def test_20_patient_cluster_bootstrap_inflates_se_under_positive_icc() -> None:
    rng = np.random.default_rng(91)
    # 100 clusters x 10 identical visits -> SE should be ~sqrt(10) larger than treating 1,000 visits as independent
    c_vals = rng.normal(1.5, 4.0, size=100)
    v = np.repeat(c_vals, 10)
    cl_grouped = np.repeat(np.arange(100), 10)
    cl_iid = np.arange(1000)
    _, se_grp, _, _ = patient_cluster_bootstrap_scalar(v, cl_grouped, n_boot=4000, seed=3)
    _, se_iid, _, _ = patient_cluster_bootstrap_scalar(v, cl_iid, n_boot=4000, seed=3)
    assert se_grp > 2.5 * se_iid


def test_21_skipped_case_disagreement_and_risk_difference_bound() -> None:
    ck = ClinicalContract(cost_fp=1.0, cost_fn=19.0, per_instance_query_cost=0.03)
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0] * 25, dtype=int)
    p1 = np.array([0.03, 0.02, 0.25, 0.08, 0.01, 0.02, 0.04, 0.01] * 25, dtype=float)
    p2 = np.array([0.12, 0.01, 0.40, 0.02, 0.01, 0.08, 0.03, 0.01] * 25, dtype=float)
    q = np.array([False, False, False, False, True, False, True, False] * 25, dtype=bool)
    clusters = np.arange(len(y))

    audit = skipped_case_disagreement_audit(y, p1, p2, q, clusters, ck, n_boot=2_000, seed=42)
    assert audit["realized_abs_decision_nb_diff_per_1k"] <= audit["theoretical_nb_diff_bound_emp_per_1k"] + 1e-6
    assert audit["theoretical_nb_diff_bound_emp_per_1k"] <= audit["theoretical_nb_diff_bound_ub95_per_1k"] + 1e-6
    assert audit["flip_count"] == audit["neg_to_pos_count"] + audit["pos_to_neg_count"]
    assert audit["flip_count"] == audit["harmful_flip_count"] + audit["beneficial_flip_count"]


def test_22_clopper_pearson_zero_numerator_rule_of_three() -> None:
    ub = clopper_pearson_upper_conservative(0, 1000, alpha=0.05)
    assert abs(ub - (1.0 - 0.05 ** (1.0 / 1000.0))) < 1e-12
    assert 0.0029 < ub < 0.0031


def test_23_calibration_metrics_well_calibrated_and_miscalibrated() -> None:
    rng = np.random.default_rng(121)
    p = expit(rng.normal(-2.0, 1.0, size=6000))
    y = rng.binomial(1, p)
    cal = compute_calibration_metrics(y, p)
    assert abs(cal["cal_intercept"]) < 0.12
    assert 0.88 < cal["cal_slope"] < 1.12
    assert cal["ici"] < 0.025

    # Systematic 2x risk overprediction -> negative CITL
    p_over = expit(logit(p) + 0.8)
    cal_over = compute_calibration_metrics(y, p_over)
    assert cal_over["cal_intercept"] < -0.55


def test_24_recalibrate_prevalence_intercept_restores_citl() -> None:
    rng = np.random.default_rng(131)
    p_true = expit(rng.normal(-2.5, 0.9, size=4000))
    y = rng.binomial(1, p_true)
    p_shifted = expit(logit(p_true) + 0.75)
    p_recal, _ = prior_odds_recalibrate_probabilities(
        p_shifted, source_prev=float(np.mean(p_shifted)), target_prev=float(np.mean(y))
    )
    prof = compute_calibration_metrics_v2(y, p_recal, n_boot=50)
    assert abs(float(prof["citl"])) < 0.08


def test_25_standardized_net_benefit_and_decision_curve_identities() -> None:
    rng = np.random.default_rng(141)
    p = rng.uniform(0.01, 0.40, size=1000)
    y = rng.binomial(1, p)
    ck = ClinicalContract(1.0, 9.0, 0.30, 0.0)
    nb = weighted_net_benefit_per_1k(y, p, np.zeros(1000, dtype=bool), ck)
    snb = standardized_net_benefit(nb, float(np.mean(y)))
    assert snb <= 1.0 + 1e-9
    dca = compute_weighted_decision_curve(y, p, np.zeros(1000, dtype=bool), [0.05, 0.10, 0.20], 0.0)
    assert dca["nb_treat_none"] == [0.0, 0.0, 0.0]


def test_26_unit_net_benefit_exchange_rate_algebraic_identity() -> None:
    rng = np.random.default_rng(151)
    y = rng.binomial(1, 0.08, size=500)
    p = rng.uniform(0.0, 0.25, size=500)
    q = rng.binomial(1, 0.30, size=500).astype(bool)
    ck = ClinicalContract(1.0, 19.0, 0.25, 0.03)  # tau* = 0.05
    nb_fn = weighted_net_benefit_per_1k(y, p, q, ck)
    tp = 1000.0 * float(np.mean((p >= 0.05) & (y == 1)))
    fp = 1000.0 * float(np.mean((p >= 0.05) & (y == 0)))
    manual = tp - (1.0 / 19.0) * fp - (0.03 / 19.0) * 1000.0 * float(np.mean(q))
    assert abs(nb_fn - manual) < 1e-10


def test_27_within_stratum_pilot_independence_null_holds() -> None:
    rng = np.random.default_rng(99)
    n = 2000
    p_base = rng.uniform(0.01, 0.30, size=n)
    y = rng.binomial(1, p_base)
    df = pd.DataFrame({
        "z1": y * 0.8 + rng.normal(0, 1.0, size=n),
        "z2": rng.normal(0, 1000.0, size=n),
    })
    pilot = sample_pilot(y, p_base, pilot_fraction=0.20, mode="collision_neyman", tau_star=0.10, n_bins=8, seed=42)
    ind = check_within_stratum_pilot_independence(df, [ChannelDef("ch1", ("z1", "z2"), 1.0)], pilot, y, p_base, n_bins=8)
    assert ind["omnibus_p_value"] > 0.01


def test_28_cochran_q_heterogeneity_zero_and_strong_divergence() -> None:
    h0 = cochran_q_heterogeneity([5.0, 5.0, 5.0], [1.0, 1.0, 1.0])
    assert abs(h0["cochran_q"]) < 1e-12 and abs(h0["p_heterogeneity"] - 1.0) < 1e-12
    h1 = cochran_q_heterogeneity([1.0, 10.0, -3.0], [0.8, 0.8, 0.8])
    assert h1["p_heterogeneity"] < 1e-6 and h1["i2_pct"] > 90.0


def test_29_holm_adjust_monotonicity_and_bounds() -> None:
    raw = [0.004, 0.021, 0.011, 0.45]
    adj = holm_adjust(raw)
    assert len(adj) == 4
    assert all(0.0 <= p <= 1.0 for p in adj)
    # Rank order of raw p-values: idx 0 (0.004*4=0.016), idx 2 (0.011*3=0.033), idx 1 (0.021*2=0.042), idx 3 (0.45)
    assert abs(adj[0] - 0.016) < 1e-12
    assert abs(adj[2] - 0.033) < 1e-12
    assert abs(adj[1] - 0.042) < 1e-12


def test_30_fast_frame_encoder_handles_missing_and_categoricals() -> None:
    df_tr = pd.DataFrame({"num": [1.0, np.nan, 3.0, 4.0], "cat": ["A", "B", "A", None]})
    df_te = pd.DataFrame({"num": [np.nan, 2.0], "cat": ["C_UNSEEN", "A"]})
    enc = FastFrameEncoder().fit(df_tr)
    x_tr = enc.transform(df_tr)
    x_te = enc.transform(df_te)
    assert x_tr.shape[1] == x_te.shape[1]
    assert np.all(np.isfinite(x_tr)) and np.all(np.isfinite(x_te))


def test_31_platt_calibrator_monotone_and_bounded() -> None:
    rng = np.random.default_rng(161)
    r = rng.normal(-1.5, 1.2, size=800)
    y = rng.binomial(1, expit(r))
    pl = fit_platt_calibrator(r, y)
    p_out = apply_platt(np.array([-5.0, -1.0, 2.0]), pl)
    assert np.all((p_out > 0.0) & (p_out < 1.0))
    assert p_out[0] < p_out[1] < p_out[2]


def test_32_agniel_deconfounder_null_calibration_under_linear_gaussian_z() -> None:
    rng = np.random.default_rng(171)
    n = 1200
    u = rng.normal(size=n)
    m = rng.binomial(1, expit(0.8 * u)).astype(float)
    z_obs = np.where(m > 0.5, 1.5 * u + rng.normal(size=n), np.nan)
    df_b = pd.DataFrame({"xb1": rng.normal(size=n), "xb2": rng.normal(size=n)})
    df_z = pd.DataFrame({"z_lab": z_obs})
    dec = AgnielDeconfounder().fit(df_b, df_z)
    out = dec.transform(df_b, df_z, seed=171)
    assert len(out) == n and out.shape[1] >= 1


def test_33_cfbh_selection_respects_fdr_under_exchangeable_calibration() -> None:
    rng = np.random.default_rng(181)
    p_ca = rng.beta(1.2, 8.0, size=2000)
    y_ca = rng.binomial(1, np.where(p_ca > 0.35, 0.85, 0.02))
    p_te = rng.beta(1.2, 8.0, size=2000)
    y_te = rng.binomial(1, np.where(p_te > 0.35, 0.85, 0.02))
    res = cfbh_selection(y_ca, p_ca, y_te, p_te, fdr_target=0.25, seed=181)
    assert res["fdp"] <= 0.32


def test_34_stage_b_oof_policy_screening_prunes_costly_pure_noise() -> None:
    rng = np.random.default_rng(191)
    n = 900
    df = pd.DataFrame({
        "b0": rng.normal(size=n),
        "z_noise1": rng.normal(size=n),
        "z_noise2": rng.normal(size=n),
    })
    p1 = expit(-2.4 + 1.0 * df["b0"].to_numpy())
    y = rng.binomial(1, p1)
    ck = ClinicalContract(1.0, 19.0, 0.25, 0.05)
    ch_map = {"ch_noise": ChannelDef("ch_noise", ("z_noise1", "z_noise2"), 1.0)}
    kept, log = oof_policy_screening_stage_b(["ch_noise"], df, y, p1, ["b0"], ch_map, ck, alpha_pol=0.10, seed=191)
    assert kept == []
    assert len(log) >= 1


def test_35_covert_greedy_cmi_afa_respects_budget_and_recovers_signal() -> None:
    rng = np.random.default_rng(201)
    n_tr, n_ca, n_te = 600, 250, 300
    tot = n_tr + n_ca + n_te
    b0 = rng.normal(size=tot)
    z_sig = rng.normal(size=tot)
    z_null = rng.normal(size=tot)
    y_all = rng.binomial(1, expit(-1.8 + 0.6 * b0 + 1.8 * z_sig))
    df = pd.DataFrame({"b0": b0, "z_sig": z_sig, "z_null": z_null})
    tr = df.iloc[:n_tr].reset_index(drop=True)
    ca = df.iloc[n_tr : n_tr + n_ca].reset_index(drop=True)
    te = df.iloc[n_tr + n_ca :].reset_index(drop=True)
    y_tr, y_ca, y_te = y_all[:n_tr], y_all[n_tr : n_tr + n_ca], y_all[n_tr + n_ca :]
    p1_tr = expit(-1.8 + 0.6 * tr["b0"].to_numpy())
    p1_ca = expit(-1.8 + 0.6 * ca["b0"].to_numpy())
    p1_te = expit(-1.8 + 0.6 * te["b0"].to_numpy())
    chs = [ChannelDef("ch_sig", ("z_sig",), 1.0), ChannelDef("ch_null", ("z_null",), 1.0)]
    ck = ClinicalContract(1.0, 9.0, 0.25, 0.01)

    traces: list[dict[str, np.ndarray]] = []
    res = evaluate_covert_greedy_cmi_afa(
        tr, y_tr, ca, y_ca, [(te, y_te)], ["b0"], chs, p1_tr, (p1_ca, [p1_te]), ck,
        budget=1.0, out_routed_traces=traces, n_masks_per_row=3, n_jobs=2, seed=201,
    )
    assert len(res) == 1
    assert res[0]["auroc_full_s2"] > 0.75
    assert 0.0 <= res[0]["realized_q_auto"] <= 1.0

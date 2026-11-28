"""Unit tests for Paper 3 v2.2.0 methodological, cost-ledger, and governance repair primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd

from collision_zone.three_predicates import (
    ChannelDef,
    ClinicalContract,
    FluidCollisionBackboneClassifier,
    solve_martingale_offset,
)
from decision_aware_acq.upca_v2_2_repairs import (
    ChannelCostSchedule,
    boundary_uncertainty_mask,
    cochran_q_subgroup_heterogeneity,
    compute_fluid_specialist_diagnostics,
    directional_voi_masks,
    discover_stage_a_split_pilot,
    evaluate_sequential_afa_scheduled,
    normalized_beta_kl_identity_max_error,
    oof_policy_screening_stage_b_scheduled,
    saerens_em_prevalence_estimate,
    two_phase_variance_srs_identity_error,
    variable_cost_cluster_bootstrap_diff,
    variable_cost_nb_per_1k,
    xor_finite_capacity_counterexample,
)


def test_channel_cost_schedule_reference_and_monotonicity() -> None:
    sched = ChannelCostSchedule(
        base_c_query=0.030,
        reference_bundle_weight=2.00,
        channel_weights={"ch_a": 0.60, "ch_b": 0.92, "ch_c": 1.40},
        overhead_share=0.25,
    )
    assert sched.panel_cost_cfp([]) == 0.0
    assert abs(sched.panel_cost_cfp(["ch_a", "ch_c"]) - 0.030) < 1e-12
    assert sched.panel_cost_cfp(["ch_a"]) < sched.panel_cost_cfp(["ch_a", "ch_b"])
    mm = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype=float)
    c_vec = sched.mask_costs_cfp(mm, ["ch_a", "ch_b", "ch_c"])
    assert c_vec[0] == 0.0
    assert abs(c_vec[2] - 0.030) < 1e-12
    assert abs(sched.with_multiplier(2.0).panel_cost_cfp(["ch_a", "ch_c"]) - 0.060) < 1e-12


def test_variable_cost_nb_and_cluster_bootstrap() -> None:
    ck = ClinicalContract(cost_fp=1.0, cost_fn=19.0, target_precision=0.25, per_instance_query_cost=0.030)
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0])
    p_arm = np.array([0.20, 0.01, 0.12, 0.02, 0.01, 0.01, 0.15, 0.02])
    p_ref = np.array([0.08, 0.08, 0.04, 0.08, 0.01, 0.01, 0.08, 0.02])
    c_arm = np.full_like(y, 0.020, dtype=float)
    c_ref = np.zeros_like(y, dtype=float)
    nb_a = variable_cost_nb_per_1k(y, p_arm, c_arm, ck)
    nb_r = variable_cost_nb_per_1k(y, p_ref, c_ref, ck)
    res = variable_cost_cluster_bootstrap_diff(
        y, p_arm, c_arm, p_ref, c_ref, np.arange(len(y)), ck, n_boot=500, seed=11
    )
    assert abs(res["diff_mean"] - (nb_a - nb_r)) < 1e-3
    assert res["diff_mean"] > 0.0


def test_beta_kl_identity_and_srs_variance_reduction() -> None:
    assert normalized_beta_kl_identity_max_error(0.05, 24.0) < 1e-14
    assert normalized_beta_kl_identity_max_error(0.15, 16.0) < 1e-14
    assert two_phase_variance_srs_identity_error(600, 120, seed=19) < 1e-12


def test_xor_finite_capacity_counterexample_positive_gain() -> None:
    out = xor_finite_capacity_counterexample(n_samples=2000, seed=23)
    assert out["cz_score_gain"] > 0.02
    assert out["logloss_gain"] > 0.30


def test_split_pilot_offset_discovery_recovers_signal_without_nulls() -> None:
    rng = np.random.default_rng(101)
    n = 1600
    x0 = rng.normal(size=n)
    sig = rng.normal(size=n)
    noise = rng.normal(size=n)
    in_cz = np.abs(x0) < 0.7
    ell = -2.6 + 1.1 * x0 + np.where(in_cz, 1.9 * sig, 0.0)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-ell)))
    df = pd.DataFrame({"x0": x0, "sig": sig, "noise": noise})
    p1 = 1.0 / (1.0 + np.exp(-(-2.6 + 1.1 * x0)))
    ck = ClinicalContract(cost_fp=1.0, cost_fn=9.0, target_precision=0.25, per_instance_query_cost=0.02)
    chs = [ChannelDef("ch_sig", ("sig",), 1.0), ChannelDef("ch_null", ("noise",), 1.0)]
    out, p_conf = discover_stage_a_split_pilot(df, y, ["x0"], chs, p1, ck, budget=2.0, pilot_fraction=0.35, seed=101)
    assert "ch_sig" in out.selected_ids
    assert "ch_null" not in out.selected_ids
    assert len(p_conf.indices) >= 100


def test_specialist_diagnostics_and_routing_comparators() -> None:
    rng = np.random.default_rng(55)
    n = 500
    xb = rng.normal(size=(n, 2))
    z = rng.normal(size=(n, 1))
    ell1 = -2.2 + 0.9 * xb[:, 0]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(ell1 + 1.4 * z[:, 0]))))
    clf = FluidCollisionBackboneClassifier(tau_star=0.10, kappa_max=20.0, min_cz_ess=60.0, seed=55).fit(
        np.hstack([xb, z]), z, y, ell1
    )
    diag = compute_fluid_specialist_diagnostics(clf, np.hstack([xb, z]), z, ell1, 0.10)
    assert diag["rms_r_cz"] >= 0.0
    assert 0.0 <= diag["decision_flip_rate_vs_glob"] <= 1.0

    p1 = 1.0 / (1.0 + np.exp(-ell1))
    bu = boundary_uncertainty_mask(p1, 0.10, 0.25)
    assert abs(float(np.mean(bu)) - 0.25) <= 0.01
    d_masks = directional_voi_masks(p1, np.full(n, 0.05), 0.02, 0.10)
    assert np.all(d_masks["escalation_only"] == (p1 < 0.10))
    assert np.all(d_masks["deescalation_only"] == (p1 >= 0.10))


def test_scheduled_stage_b_and_channel_billed_afa() -> None:
    rng = np.random.default_rng(88)
    n = 2000
    x0 = rng.normal(size=n)
    z_sig = rng.normal(size=n)
    z_null = rng.normal(size=n)
    ell1 = -2.2 + 0.9 * x0
    p1 = 1.0 / (1.0 + np.exp(-ell1))
    mu = solve_martingale_offset(p1, np.full(n, 2.0))
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(ell1 + mu + 2.0 * z_sig))))
    df = pd.DataFrame({"x0": x0, "z_sig": z_sig, "z_null": z_null})
    chs = [ChannelDef("ch_sig", ("z_sig",), 0.8), ChannelDef("ch_null_1", ("z_null",), 0.8, is_null_stress=True)]
    ch_map = {c.channel_id: c for c in chs}
    sched = ChannelCostSchedule(0.02, 1.6, {"ch_sig": 0.8, "ch_null_1": 0.8}, overhead_share=0.25)
    ck = ClinicalContract(1.0, 9.0, 0.25, 0.02)

    tr, cal, te = df.iloc[:1400].reset_index(drop=True), df.iloc[1400:1700].reset_index(drop=True), df.iloc[1700:].reset_index(drop=True)
    y_tr, y_cal, y_te = y[:1400], y[1400:1700], y[1700:]
    p1_tr, p1_cal, p1_te = p1[:1400], p1[1400:1700], p1[1700:]

    kept, _ = oof_policy_screening_stage_b_scheduled(
        ["ch_sig", "ch_null_1"], tr, y_tr, p1_tr, ["x0"], ch_map, ck, sched, seed=88
    )
    assert kept == ["ch_sig"]

    for mode in ("cmi", "direct_voi"):
        summ, pr, ct, mask = evaluate_sequential_afa_scheduled(
            tr, y_tr, cal, y_cal, te, y_te, ["x0"], chs,
            p1_tr, p1_cal, p1_te, p1_cal, p1_te, ck, 1.6, sched, policy_mode=mode, seed=88
        )
        assert pr.shape == y_te.shape
        assert ct.shape == y_te.shape
        assert mask.shape == (len(y_te), 2)
        assert np.isfinite(summ["nb_routed"])


def test_saerens_em_and_cochran_q() -> None:
    rng = np.random.default_rng(77)
    y_tgt = rng.binomial(1, 0.04, size=5000)
    # X | Y=1 ~ N(1.0, 1), X | Y=0 ~ N(-1.0, 1) -> exact Bayes LLR = 2.0 * x
    x = np.where(y_tgt == 1, rng.normal(1.0, 1.0, 5000), rng.normal(-1.0, 1.0, 5000))
    ell_src = np.log(0.12 / 0.88) + 2.0 * x
    p_src = 1.0 / (1.0 + np.exp(-ell_src))
    pi_hat = saerens_em_prevalence_estimate(p_src, pi_src=0.12)
    assert 0.025 < pi_hat < 0.055

    hom = cochran_q_subgroup_heterogeneity([1.0, 1.1, 0.95], [0.5, 0.5, 0.5])
    het = cochran_q_subgroup_heterogeneity([5.0, -2.0, 0.5], [0.6, 0.6, 0.6])
    assert hom["p_het"] > 0.80
    assert het["p_het"] < 0.001

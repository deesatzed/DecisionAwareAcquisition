"""Unit tests for Paper 3 v2 Leak-Free Statistical, Calibration, Bootstrap & Safety Primitives (`R1`-`R24`)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REBUILD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REBUILD_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "collision_zone_ed" / "src"))

from collision_zone.three_predicates import ChannelDef, ClinicalContract, sample_pilot
from upca_v2_core import (
    adjust_probs_for_target_prevalence,
    case_control_prevalence_weights,
    check_within_stratum_pilot_independence,
    cochran_q_heterogeneity,
    compute_calibration_metrics,
    compute_weighted_decision_curve,
    patient_cluster_bootstrap_diff,
    skipped_case_disagreement_audit,
    standardized_net_benefit,
    weighted_net_benefit_per_1k,
)


def test_case_control_prevalence_adjustment_exact() -> None:
    rng = np.random.default_rng(7)
    # Case-control sample with 20% cases -> target clinical prevalence 2.0%
    y = np.array([1] * 200 + [0] * 800, dtype=int)
    w = case_control_prevalence_weights(y, pi_target=0.02)
    assert abs(float(np.mean(w)) - 1.0) < 1e-12
    weighted_prev = float(np.sum(w * y) / np.sum(w))
    assert abs(weighted_prev - 0.02) < 1e-12

    # Constant sample-prevalence prediction 0.20 maps to exact target prevalence 0.02
    p_cc = np.full(1000, 0.20, dtype=float)
    p_tgt = adjust_probs_for_target_prevalence(p_cc, pi_sample=0.20, pi_target=0.02)
    assert np.allclose(p_tgt, 0.02, atol=1e-10)
    cal = compute_calibration_metrics(y, p_tgt, sample_weight=w)
    assert abs(cal["cal_intercept"]) < 1e-3


def test_patient_cluster_bootstrap_davison_hinkley_and_clustering() -> None:
    rng = np.random.default_rng(11)
    contract = ClinicalContract(cost_fp=1.0, cost_fn=9.0, per_instance_query_cost=0.02)
    y = rng.binomial(1, 0.10, size=600)
    p1 = np.clip(0.10 + rng.normal(0, 0.03, size=600), 0.01, 0.99)
    q0 = np.zeros(600, dtype=bool)
    clusters = np.repeat(np.arange(150), 4)

    # Exact tie returns 0.0 diff and p = 1.0
    tie = patient_cluster_bootstrap_diff(y, p1, q0, p1, q0, clusters, contract, n_boot=10_000, seed=1)
    assert tie["diff_mean"] == 0.0
    assert tie["p_two_sided"] == 1.0

    # Strict dominance never returns p = 0.0; floor is 2 / (B + 1) = 2 / 10001
    p_perf = np.where(y == 1, 0.95, 0.01)
    dom = patient_cluster_bootstrap_diff(y, p_perf, q0, p1, q0, clusters, contract, n_boot=10_000, seed=2)
    assert dom["diff_mean"] > 50.0
    assert abs(dom["p_two_sided"] - (2.0 / 10_001.0)) < 1e-6


def test_skipped_case_disagreement_and_risk_difference_bound() -> None:
    contract = ClinicalContract(cost_fp=1.0, cost_fn=19.0, per_instance_query_cost=0.03)  # tau* = 0.05
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0] * 25, dtype=int)
    p1 = np.array([0.03, 0.02, 0.25, 0.08, 0.01, 0.02, 0.04, 0.01] * 25, dtype=float)
    p2 = np.array([0.12, 0.01, 0.40, 0.02, 0.01, 0.08, 0.03, 0.01] * 25, dtype=float)
    q = np.array([False, False, False, False, True, False, True, False] * 25, dtype=bool)
    clusters = np.arange(len(y))

    audit = skipped_case_disagreement_audit(y, p1, p2, q, clusters, contract, n_boot=2_000, seed=42)
    # Verify algebraic Risk-Difference Bound holds both empirically and at 95% UB
    assert audit["realized_abs_decision_nb_diff_per_1k"] <= audit["theoretical_nb_diff_bound_emp_per_1k"] + 1e-6
    assert audit["theoretical_nb_diff_bound_emp_per_1k"] <= audit["theoretical_nb_diff_bound_ub95_per_1k"] + 1e-6
    assert audit["flip_count"] == audit["neg_to_pos_count"] + audit["pos_to_neg_count"]
    assert audit["flip_count"] == audit["harmful_flip_count"] + audit["beneficial_flip_count"]


def test_within_stratum_independence_and_snb_dca() -> None:
    rng = np.random.default_rng(99)
    n = 2000
    p_base = rng.uniform(0.01, 0.30, size=n)
    y = rng.binomial(1, p_base)
    df = pd.DataFrame({
        "z1": y * 0.8 + rng.normal(0, 1.0, size=n),
        "z2": rng.normal(0, 1.0, size=n),
    })
    pilot = sample_pilot(y, p_base, pilot_fraction=0.20, mode="collision_neyman", tau_star=0.10, n_bins=8, seed=42)
    ind = check_within_stratum_pilot_independence(
        df, [ChannelDef("ch1", ("z1", "z2"), 1.0)], pilot, y, p_base, n_bins=8
    )
    assert ind["omnibus_p_value"] > 0.01

    contract = ClinicalContract(cost_fp=1.0, cost_fn=9.0, per_instance_query_cost=0.0)
    nb = weighted_net_benefit_per_1k(y, p_base, np.zeros(n, dtype=bool), contract)
    snb = standardized_net_benefit(nb, float(np.mean(y)))
    assert snb <= 1.0 + 1e-9
    dca = compute_weighted_decision_curve(y, p_base, np.zeros(n, dtype=bool), [0.05, 0.10, 0.20], 0.0)
    assert len(dca["nb_model"]) == 3
    het = cochran_q_heterogeneity([5.0, 5.2, 4.9], [1.0, 1.1, 0.9])
    assert het["p_heterogeneity"] > 0.90

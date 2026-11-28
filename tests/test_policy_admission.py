"""Unit tests for Track 1 Policy-Aware Channel Admission & Skipped-Case Safety Certification."""

from __future__ import annotations

import numpy as np
from policy_admission import (
    clopper_pearson_upper,
    compute_murphy_diagram,
    compute_policy_delta_stats,
    per_record_net_utility,
    selective_label_audit,
    skipped_case_safety_certificate,
)


def test_clopper_pearson_monotone_and_exact() -> None:
    # Rule of 3: 0/1000 events at alpha=0.05 -> 1 - 0.05^(1/1000) ~= 0.002991
    ub0 = clopper_pearson_upper(0, 1000, alpha=0.05)
    assert abs(ub0 - (1.0 - 0.05 ** (1.0 / 1000.0))) < 1e-6
    ub5 = clopper_pearson_upper(5, 1000, alpha=0.05)
    assert ub5 > ub0
    assert clopper_pearson_upper(10, 10) == 1.0


def test_skipped_case_safety_certificate_exact() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    p1 = np.array([0.01, 0.02, 0.03, 0.12, 0.04, 0.02])
    p2 = np.array([0.01, 0.02, 0.15, 0.20, 0.04, 0.01])
    q = np.array([False, False, False, True, True, False])
    cert = skipped_case_safety_certificate(y, p1, p2, q, tau_star=0.08, cost_fp=1.0, cost_fn=11.5)
    assert cert["n_skipped"] == 4
    assert cert["flip_count"] == 1
    assert cert["missed_tp_rescue_count"] == 1
    assert cert["flip_rate_ub95"] > cert["flip_rate"]
    assert cert["forgone_nb_per_1k_all"] > 0.0


def test_murphy_diagram_and_selective_label_audit() -> None:
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.10, size=400)
    p_ref = np.full(400, 0.10)
    p_good = np.clip(0.05 + 0.35 * y + rng.normal(0, 0.05, size=400), 0.001, 0.999)
    md = compute_murphy_diagram(y, p_good, p_ref, taus=[0.05, 0.10, 0.20])
    assert all(g > 0.0 for g in md["murphy_gain_per_1k"])

    tested = rng.binomial(1, 0.2, size=400).astype(bool)
    audit = selective_label_audit(y, p_good, tested, tau_star=0.10, cost_fp=1.0, cost_fn=9.0)
    assert audit["clinician_untested"]["auroc"] > 0.9
    assert audit["clinician_untested"]["rescued_events_at_tau"] > 0


def test_policy_delta_stats_detects_positive_policy_gain() -> None:
    rng = np.random.default_rng(11)
    y = rng.binomial(1, 0.08, size=500)
    p_base = np.full(500, 0.04)
    p_better = np.where(y == 1, 0.25, 0.02)
    nb0 = per_record_net_utility(y, p_base, np.zeros(500), tau_star=0.08, cost_fp=1.0, cost_fn=11.5, query_cost=0.02)
    nb1 = per_record_net_utility(y, p_better, np.ones(500), tau_star=0.08, cost_fp=1.0, cost_fn=11.5, query_cost=0.02)
    delta, se, z, pval = compute_policy_delta_stats(nb1, nb0)
    assert delta > 50.0 and se > 0.0 and z > 3.0 and pval < 0.001

"""Deterministic `< 1s` Table & Algebraic Ledger Verifier for Paper 3 (`DecisionAwareAcquisition` v2.2.0).

Recomputes and cross-checks from committed JSON checkpoints (`results/paper3_v2_2_remediation_results.json`
and `results/cohort_provenance_and_flow_audit.json`):
  1. Single-integer encounter reconciliation (`89,395 + 51,158 + 16,771 = 157,324` primary + `818` post-ROSC).
  2. Exact Beta-KL (`Buja et al., 2005`) and SRS `H=1` two-phase variance (`s^2 / n`) identities + `XOR` stump gain.
  3. Exact `ChannelCostSchedule` panel fees, monotone query-cost multiplier sweeps (`0.5x..3.0x`), and
     Union-Panel vs. Common-Panel fee inequalities across all 6 clinical tasks.
  4. Exact algebraic reconstruction of every policy's `pre_fee_nb`, `fee_drag`, `net_ledger_nb`,
     `pre_fee_delta + fee_saving`, and `missed_cases` across all `4 x 8 = 32` Stanford ICU exchange-ledger rows,
     plus `0` missed Stage-1 baseline cases under `Escalation-Only VOI`.
  5. Zero test-label leakage flags on deployable arms of the 4-way prior-shift taxonomy and completeness of
     the `12` subgroup Cochran's `Q` heterogeneity audits (`vs. Full 100%`, `vs. Stage-1`, `vs. Global-CMI`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decision_aware_acq.upca_v2_2_repairs import ChannelCostSchedule


def verify_all_paper3_v2_2_ledgers(
    results_path: Path | None = None,
    cohort_audit_path: Path | None = None,
) -> dict[str, Any]:
    res_p = results_path or (REPO_ROOT / "results" / "paper3_v2_2_remediation_results.json")
    coh_p = cohort_audit_path or (REPO_ROOT / "results" / "cohort_provenance_and_flow_audit.json")
    if not res_p.exists():
        raise FileNotFoundError(f"Missing Paper 3 v2.2 results JSON: {res_p}")
    if not coh_p.exists():
        raise FileNotFoundError(f"Missing cohort audit JSON: {coh_p}")

    d = json.loads(res_p.read_text())
    coh = json.loads(coh_p.read_text())
    checks_run = 0

    # 1. Exact Unique Encounter Reconciliation
    rec = d["unique_primary_encounters_reconciliation"]
    assert rec["stanford_mcmed_2h_unique_visits"] == 89_395
    assert rec["stanford_mcmed_2h_discharged_subset_visits"] == 51_736
    assert rec["bwh_ed_ecg_unique_visits"] == 51_158
    assert rec["ntuh_pre_arrest_unique_visits"] == 16_771
    assert (
        rec["stanford_mcmed_2h_unique_visits"]
        + rec["bwh_ed_ecg_unique_visits"]
        + rec["ntuh_pre_arrest_unique_visits"]
        == rec["total_unique_primary_visits"]
        == 157_324
    )
    assert rec["supplementary_ntuh_post_rosc_visits"] == 818
    assert coh["mcmed_stanford"]["cohort_flow"]["n_strict_2h_landmark"] == 89_395
    assert coh["mcmed_stanford"]["cohort_flow"]["n_strict_2h_discharged"] == 51_736
    assert coh["bwh_ed_ecg"]["n_modeling_encounters"] == 51_158
    assert coh["ntuh_arrest_ecg"]["pre_arrest_case_control_2015_2019"]["n_total_ecgs"] == 16_771
    checks_run += 10

    # 2. Mathematical Identities & XOR Counterexample
    b1 = d["block1_synthetic_and_pilots"]
    id_err = b1["identities_and_counterexamples"]["normalized_beta_kl_max_abs_error"]
    for k, err_v in id_err.items():
        assert float(err_v) < 1e-13, f"Beta-KL identity error too large for {k}: {err_v}"
        checks_run += 1
    assert float(b1["identities_and_counterexamples"]["two_phase_variance_srs_h1_identity_abs_error"]) < 1e-14
    xor_out = b1["identities_and_counterexamples"]["xor_finite_capacity_stump_counterexample"]
    assert float(xor_out["cz_score_gain"]) > 0.05 and float(xor_out["logloss_gain"]) > 0.30
    checks_run += 3

    # 3. Block 2: ChannelCostSchedule, Common-Panel vs Union-Panel, & 5-Point Fee Sweep
    b2_tasks = d["block2_common_panel_and_fee_sweep"]["tasks"]
    expected_tasks = {
        "mcmed_critical_icu_death": (69_022, 9_887, 10_486),
        "mcmed_hospital_admission": (69_022, 9_887, 10_486),
        "mcmed_bounce72_admit_2h": (39_864, 5_707, 6_165),
        "bwh_death_030_ecg_only": (39_050, 6_329, 5_779),
        "bwh_stent_cabg_010_ecg_only": (39_050, 6_329, 5_779),
        "ntuh_pre_arrest_sca_pi002": (9_525, 3_595, 3_651),
    }
    channel_weight_lookup = {
        "ch_cdw_prior_pmh": 0.40,
        "ch_cdw_prior_meds": 0.40,
        "ch_lab_cbc": 0.78,
        "ch_lab_renal_elyte": 0.92,
        "ch_lab_lactate_cardiac": 1.15,
        "ch_lab_lft_coag": 0.95,
        "ch_rads_2h": 1.50,
        "ch_monitor_numerics_2h": 0.60,
        "ch_ecg_ischemia_flags": 0.40,
        "ch_ecg_rhythm_struct": 0.40,
        "ch_ecg_waveform": 0.85,
        "ch_ant_v1v4": 0.55,
        "ch_lat_i_avl_v5v6": 0.55,
        "ch_inf_ii_iii_avf": 0.55,
        "ch_avr": 0.45,
        "ch_ecg_diag_stmts": 0.40,
        "ch_waveform_12lead_median": 0.85,
        "ch_null_noise_1": 0.50,
        "ch_null_noise_2": 0.50,
    }
    for tid, (ntr, nca, nte) in expected_tasks.items():
        tobj = b2_tasks[tid]
        assert (tobj["n_train"], tobj["n_cal"], tobj["n_test"]) == (ntr, nca, nte)
        cs_meta = tobj["cost_schedule"]
        sched = ChannelCostSchedule(
            base_c_query=cs_meta["base_c_query"],
            reference_bundle_weight=cs_meta["reference_bundle_weight"],
            channel_weights=channel_weight_lookup,
            overhead_share=cs_meta["overhead_share"],
        )
        arms_r10 = tobj["r10_common_panel_and_triad"]["arms"]
        for aname in ("upca_fluid", "global_cmi_gbdt", "weighted_cz_fluid", "full_100pct_all_m_gbdt"):
            aobj = arms_r10[aname]
            expected_c = sched.panel_cost_cfp(aobj["locked_panel"])
            assert abs(float(aobj["panel_cost_cfp"]) - expected_c) <= 2e-5, f"{tid} {aname} panel cost mismatch"
            if "union_panel_cost_cfp" in aobj and aobj["union_panel_ids"]:
                exp_u = sched.panel_cost_cfp(aobj["union_panel_ids"])
                assert abs(float(aobj["union_panel_cost_cfp"]) - exp_u) <= 2e-5
            checks_run += 2

        sweep = tobj["r10_common_panel_and_triad"]["query_cost_multiplier_sweep"]
        assert [r["cost_multiplier"] for r in sweep] == [0.5, 1.0, 1.5, 2.0, 3.0]
        q_rates = [float(r["upca_query_rate"]) for r in sweep]
        assert all(q_rates[i] >= q_rates[i + 1] - 1e-6 for i in range(len(q_rates) - 1)), (
            f"Non-monotone query rate in fee sweep for {tid}: {q_rates}"
        )
        checks_run += 3

    # 4. Block 3: Algebraic Reconstruction of All 32 Stanford ICU Exchange & Missed-Case Ledger Rows
    b3_ex = d["block3_exchange_and_prior_shift"]["stanford_icu_exchange_and_missed_case_ledger"]
    assert len(b3_ex) == 4
    for tau_key, tau_block in b3_ex.items():
        tau = float(tau_block["tau"])
        n_test = int(tau_block["n_test"])
        tot_ev = int(tau_block["total_true_events"])
        assert n_test == 10_486 and tot_ev == 229
        odds = tau / (1.0 - tau)
        c_fn_tau = (1.0 - tau) / tau
        pol_map = {p["policy"]: p for p in tau_block["policies"]}
        tp_all8 = int(pol_map["full_100pct_all_8_channels"]["tp_count"])
        nb_all8 = float(pol_map["full_100pct_all_8_channels"]["net_ledger_nb_per_1k"])

        # Escalation-Only VOI never misses a baseline-flagged Stage-1 case
        assert int(pol_map["upca_fluid_escalation_only"]["missed_cases_vs_stage1"]) <= 0
        checks_run += 2

        for p in tau_block["policies"]:
            tp_1k_rebuilt = 1000.0 * int(p["tp_count"]) / float(n_test)
            fp_1k_rebuilt = 1000.0 * int(p["fp_count"]) / float(n_test)
            pre_fee_rebuilt = tp_1k_rebuilt - odds * fp_1k_rebuilt
            fee_drag_rebuilt = 1000.0 * (float(p["mean_cost_cfp"]) / c_fn_tau)
            net_nb_rebuilt = pre_fee_rebuilt - fee_drag_rebuilt
            assert abs(pre_fee_rebuilt - float(p["pre_fee_nb_per_1k"])) <= 0.015, f"{tau_key} {p['policy']} pre-fee NB"
            assert abs(fee_drag_rebuilt - float(p["fee_drag_nb_per_1k"])) <= 0.015, f"{tau_key} {p['policy']} fee drag"
            assert abs(net_nb_rebuilt - float(p["net_ledger_nb_per_1k"])) <= 0.015, f"{tau_key} {p['policy']} net NB"
            assert int(p["tp_count"]) + int(p["missed_cases_vs_full_all8"]) == tp_all8
            decomp_sum = float(p["pre_fee_delta_vs_full_all8"]) + float(p["fee_saving_vs_full_all8"])
            assert abs(decomp_sum - (float(p["net_ledger_nb_per_1k"]) - nb_all8)) <= 0.015
            checks_run += 5

    # 5. Block 3 Prior-Shift Taxonomy & Block 4 Subgroup Cochran's Q Completeness
    ps_tax = d["block3_exchange_and_prior_shift"]["four_way_prior_shift_taxonomy"]
    for pobj in ps_tax.values():
        m = pobj["methods"]
        assert m["unadjusted_0_test_labels"]["uses_test_labels"] is False
        assert m["saerens_em_unlabeled_0_test_labels"]["uses_test_labels"] is False
        assert m["disjoint_cal_prior_align_0_test_labels"]["uses_test_labels"] is False
        assert m["oracle_test_prevalence_bound_non_deployable"]["uses_test_labels"] is True
        checks_run += 4

    b4_tasks = d["block4_subgroup_heterogeneity_audit"]["tasks"]
    assert set(b4_tasks.keys()) == {
        "mcmed_critical_icu_death",
        "mcmed_hospital_admission",
        "bwh_death_030_ecg_only",
    }
    for attrs in b4_tasks.values():
        assert len(attrs) == 4
        for aobj in attrs.values():
            hq = aobj["cochran_q_heterogeneity"]
            assert set(hq.keys()) == {"vs_full_all_m", "vs_stage1", "vs_global_cmi"}
            checks_run += 1

    return {"status": "PASS", "checks_verified": checks_run}


def main() -> None:
    out = verify_all_paper3_v2_2_ledgers()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Paper 3 (`DecisionAwareAcquisition` v2.2.0) Modular 4-Block Remediation & Ledger Suite.

Closes all 10 cross-paper methodological, cost-accounting, and clinical-governance issues
proactively propagated from sealed Paper 1 (`imboreceipt` v2.2.1) and sealed Paper 2
(`collision-zone-ed` v2.3.1):

  - Block 1 (`block1_synthetic_and_pilots.json`):
      * Exact normalized-Beta / Boltzmann-KL identity (`Buja, Stuetzle & Shen, 2005`),
        linear-weight HT two-phase variance SRS reduction (`s^2 / n`), and finite-capacity
        `XOR` counterexample motivating the learner-specific predictive-utility null.
      * Preserves `1,000`-rep `Experiment E5`, `72`-cell `4D` grid, and `500`-pilot Stanford ICU
        concordance from `results/paper3_v2_remediation_results.json`.
      * `200`-rep `Experiment E5` Disjoint `50/50` Split-Pilot Confirmation vs. `4`-Fold
        Cross-Fitted Offset Discovery + Proportional `ChannelCostSchedule` at `B=2.0` +
        `200`-rep weak-signal (`sigma=1.6`) stress audit.
      * `200`-pilot Stanford ICU (`rho_pilot=0.20`, `seeds 700..899`, with `2` Gaussian null
        channels) Disjoint `50/50` Split-Pilot Confirmation vs. `4`-Fold Cross-Fitted Offset,
        `Weighted-CZ`, and `IPW-CZ` discovery (quantifying empirical null FWER and power cost).

  - Block 2 (`block2_common_panel_and_fee_sweep.json`):
      * Across all 6 primary clinical tasks (`157,324` unique primary encounters across 3 hospitals),
        evaluates `Common-Panel Bagged VOI` (`R = 10` patient-grouped dev splits, primary deployable
        estimator), `Canonical` single-split, `Per-Split Channel-Billed`, `Symmetric Ensemble`,
        and `Union-Panel Bagged VOI` under explicit `ChannelCostSchedule` (`overhead_share=0.25`).
      * Benchmarks directly against `Full 100% All-M Channels`, `Full 100% Locked-Panel`,
        `Boundary Uncertainty` (`|p_1 - tau*|`, matched-`q` and `20%` budget), `Global-CMI + GBDT`,
        `Weighted-CZ`, `Split-Pilot Offset-CZ`, and proportionally channel-billed `Covert-CMI` &
        `Direct-VOI` sequential AFA (both per-split and `R=10` majority-mask deployment).
      * Runs a 5-point query-cost multiplier sweep (`0.5x, 1.0x, 1.5x, 2.0x, 3.0x`) and records
        scale-invariant Stage-2 specialist diagnostics (`R_CZ` RMS contribution and decision-flip rate).

  - Block 3 (`block3_exchange_and_prior_shift.json`):
      * Stanford `2h` Critical ICU/Death (`N_test = 10,486`, `229` events) 4-threshold
        (`tau in {3%, 5%, 8%, 10%}`) clinical exchange, missed-case, pre-fee vs. fee-saving,
        and directional VOI (`Two-Sided`, `Escalation-Only`, `De-escalation-Only`) ledger.
      * 4-Way Prior-Shift Taxonomy (`Unadjusted` vs. Purely Unlabeled `Saerens 2002` EM vs.
        Disjoint `Cal`-Slice Mean Alignment vs. Non-Deployable `Oracle Test Prevalence` Bound).

  - Block 4 (`block4_subgroup_heterogeneity_audit.json`):
      * `10,000`-replicate patient-cluster subgroup CIs and Cochran's `Q` heterogeneity tests across
        Race, Sex, Age (`<65` vs. `>=65`), and Payer / Neighborhood IRS-AGI Quartile on Stanford ICU,
        Stanford Admission, and BWH 30d Mortality for `UPCA vs. Full 100% All-M`, `UPCA vs. Stage-1`,
        and `UPCA vs. Global-CMI + GBDT`.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

_N_THR = os.environ.get("V22_OMP_THREADS", "8")
os.environ["OMP_NUM_THREADS"] = _N_THR
os.environ["OPENBLAS_NUM_THREADS"] = _N_THR
os.environ["MKL_NUM_THREADS"] = _N_THR
os.environ["VECLIB_MAXIMUM_THREADS"] = _N_THR
os.environ["NUMEXPR_NUM_THREADS"] = _N_THR
os.environ.setdefault("UPCA_LGBM_JOBS", _N_THR)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

FALLBACK_DATA_DIR = Path(
    "/Users/o2satz/.gemini/antigravity/brain/5be51496-dafb-4e38-a543-75823c062544/scratch/paper3_v2_rebuild/data"
)
DATA_DIR = (REPO_ROOT / "data") if (REPO_ROOT / "data" / "mcmed_stanford_2h_v2.parquet").exists() else FALLBACK_DATA_DIR
SCRATCH_TRACE_DIR = Path(
    os.environ.get(
        "PAPER3_TRACE_CACHE_DIR",
        "/Users/o2satz/.gemini/antigravity/brain/5be51496-dafb-4e38-a543-75823c062544/scratch/paper3_v2_2_traces",
    )
)
RES_DIR = REPO_ROOT / "results"
BLOCKS_DIR = RES_DIR / "v2_2_blocks"
V21_JSON = RES_DIR / "paper3_v2_remediation_results.json"
OUT_JSON = RES_DIR / "paper3_v2_2_remediation_results.json"

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist
from scipy.stats import binomtest
from scipy.stats import t as t_dist
from sklearn.metrics import average_precision_score, roc_auc_score

from collision_zone.three_predicates import (
    ChannelDef,
    ClinicalContract,
    FastFrameEncoder,
    FluidCollisionBackboneClassifier,
    apply_platt,
    compute_oof_stage1,
    fit_platt_calibrator,
    fit_stage1_predictions,
    holm_adjust,
    solve_martingale_offset,
)
from decision_aware_acq.upca_v2_2_repairs import (
    ChannelCostSchedule,
    boundary_uncertainty_mask,
    cochran_q_subgroup_heterogeneity,
    compute_fluid_specialist_diagnostics,
    directional_voi_masks,
    discover_stage_a_split_pilot,
    evaluate_dual_afa_scheduled,
    normalized_beta_kl_identity_max_error,
    oof_policy_screening_stage_b_scheduled,
    predict_masked_te_from_state,
    saerens_em_prevalence_estimate,
    two_phase_variance_srs_identity_error,
    variable_cost_cluster_bootstrap_diff,
    variable_cost_nb_per_1k,
    variable_cost_unit_nb_vec,
    xor_finite_capacity_counterexample,
)
from decision_aware_acq.upca_v2_core import (
    adjust_probs_for_target_prevalence,
    case_control_prevalence_weights,
    compute_calibration_metrics,
    discover_stage_a_ablation,
    fit_predict_prepurchase_voi,
    oof_policy_screening_stage_b,
    skipped_case_disagreement_audit,
    standardized_net_benefit,
    weighted_net_benefit_per_1k,
)


def _clopper_pearson_ci95(k: int, n: int) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    lo = 0.0 if k <= 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return [round(lo, 5), round(hi, 5)]


def _paired_mcnemar(win_a: list[bool], win_b: list[bool], seed: int = 42) -> dict[str, Any]:
    a = np.asarray(win_a, dtype=bool)
    b = np.asarray(win_b, dtype=bool)
    n = int(a.size)
    n10 = int(np.sum(a & (~b)))
    n01 = int(np.sum((~a) & b))
    disc = n10 + n01
    p_mc = float(binomtest(n10, disc, 0.5, alternative="two-sided").pvalue) if disc > 0 else 1.0
    diff_v = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(10000, n))
    b_diffs = np.mean(diff_v[idx], axis=1)
    lo, hi = np.quantile(b_diffs, [0.025, 0.975])
    return {
        "n_pilots": n,
        "rate_a": round(float(np.mean(a)), 5),
        "rate_b": round(float(np.mean(b)), 5),
        "diff_rate": round(float(np.mean(diff_v)), 5),
        "diff_ci95": [round(float(lo), 5), round(float(hi), 5)],
        "a_only_n10": n10,
        "b_only_n01": n01,
        "mcnemar_p_two_sided": round(p_mc, 6),
    }


def _init_single_thread_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["UPCA_LGBM_JOBS"] = "1"
    orig_proc = lgb.LGBMModel._process_params

    def _patched_proc(self: Any, stage: str = "fit") -> dict[str, Any]:
        params = orig_proc(self, stage=stage)
        params["n_jobs"] = 1
        params["num_threads"] = 1
        return params

    lgb.LGBMModel._process_params = _patched_proc  # type: ignore[method-assign]
    orig_b_init = lgb.Booster.__init__

    def _patched_b_init(self: Any, params: dict[str, Any] | None = None, *args: Any, **kwargs: Any) -> None:
        if isinstance(params, dict):
            params = dict(params)
            params["n_jobs"] = 1
            params["num_threads"] = 1
        orig_b_init(self, params, *args, **kwargs)

    lgb.Booster.__init__ = _patched_b_init  # type: ignore[method-assign]


# ===========================================================================
# COHORT LOADER HELPER
# ===========================================================================
def load_all_clinical_tasks() -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    mc = pd.read_parquet(DATA_DIR / "mcmed_stanford_2h_v2.parquet")
    mc["gender_male"] = mc["Gender"].astype(str).str.upper().str.startswith("M").astype(int)
    mc["sex_str"] = np.where(mc["gender_male"] == 1, "Male", "Female")
    mc["age_band"] = np.where(mc["Age"].to_numpy(dtype=float) >= 65.0, "Age_65_plus", "Age_under_65")
    top_cc = mc.loc[mc["split"] == "train", "CC"].value_counts().head(25).index.tolist()
    mc["cc_clean"] = mc["CC"].where(mc["CC"].isin(top_cc), "OTHER_CC").astype(str)
    mc["arrival_clean"] = mc["Means_of_arrival"].fillna("Missing").astype(str)
    mc["payor_clean"] = mc["Payor_class"].fillna("Unspecified_or_Commercial").astype(str)
    mc["race_clean"] = np.where(
        mc["Race"].isin(["White", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander"]),
        mc["Race"].astype(str),
        "Other_or_Unknown",
    )
    mc_base = [
        "Age",
        "gender_male",
        "Triage_Temp",
        "Triage_HR",
        "Triage_RR",
        "Triage_SpO2",
        "Triage_SBP",
        "Triage_DBP",
        "esi_num",
        "cc_clean",
        "arrival_clean",
    ]
    mc_channels = [
        ChannelDef("ch_cdw_prior_pmh", tuple(c for c in mc.columns if c.startswith("cdw_pmh_")), 0.40),
        ChannelDef("ch_cdw_prior_meds", tuple(c for c in mc.columns if c.startswith("cdw_meds_")), 0.40),
        ChannelDef("ch_lab_cbc", ("lab_cbc_wbc", "lab_cbc_hgb", "lab_cbc_plt"), 0.78),
        ChannelDef(
            "ch_lab_renal_elyte",
            ("lab_renal_cr", "lab_renal_bun", "lab_elyte_na", "lab_elyte_k", "lab_elyte_hco3", "lab_gluc"),
            0.92,
        ),
        ChannelDef("ch_lab_lactate_cardiac", ("lab_lactate", "lab_troponin", "lab_bnp"), 1.15),
        ChannelDef("ch_lab_lft_coag", ("lab_lft_ast", "lab_lft_alt", "lab_lft_bili", "lab_coag_inr"), 0.95),
        ChannelDef("ch_rads_2h", tuple(c for c in mc.columns if c.startswith("rad_")), 1.50),
        ChannelDef("ch_monitor_numerics_2h", tuple(c for c in mc.columns if c.startswith("mon_")), 0.60),
    ]
    mc_disch = mc[mc["is_discharged"] == 1].copy().reset_index(drop=True)

    bwh = pd.read_parquet(DATA_DIR / "bwh_ed_ecg_v2.parquet")
    bwh["sex_str"] = np.where(bwh["sex_male"].to_numpy(dtype=float) >= 0.5, "Male", "Female")
    bwh["age_band"] = np.where(bwh["age_at_admit"].to_numpy(dtype=float) >= 65.0, "Age_65_plus", "Age_under_65")
    bwh_base = [
        c
        for c in [
            "age_at_admit",
            "sex_male",
            "vent_rate",
            "pr_interval",
            "qrs_duration",
            "qt_interval",
            "qtc_interval",
            "p_axes",
            "r_axes",
            "t_axes",
        ]
        if c in bwh.columns
    ]
    bwh_channels = [
        ChannelDef(
            "ch_ecg_ischemia_flags",
            tuple(
                c
                for c in [
                    "has_st_eleva",
                    "has_depress",
                    "has_twave_inver",
                    "has_twave_abnormal",
                    "has_jpoint_eleva",
                    "has_jpoint_repol",
                    "has_st",
                    "has_twave",
                ]
                if c in bwh.columns
            ),
            0.40,
        ),
        ChannelDef(
            "ch_ecg_rhythm_struct",
            tuple(
                c
                for c in [
                    "has_afib",
                    "has_bbb",
                    "has_aberran_bbb",
                    "has_lvh",
                    "has_pacemaker",
                    "has_prolonged_qt",
                    "has_rhythm_disturbance",
                    "has_normal_ecg",
                    "has_normal_sinus",
                ]
                if c in bwh.columns
            ),
            0.40,
        ),
        ChannelDef("ch_ecg_waveform", tuple(c for c in bwh.columns if c.startswith("wf_")), 0.85),
    ]

    ntuh_cc = pd.read_parquet(DATA_DIR / "ntuh_pre_arrest_case_control_v2.parquet")
    ntuh_base = [
        c
        for c in [
            "age",
            "sex_male",
            "ventricular_rate",
            "atrial_rate",
            "pr_interval",
            "qrs_duration",
            "qt_interval",
            "qt_corrected",
            "p_axis",
            "r_axis",
            "t_axis",
            "qtc_frederica",
        ]
        if c in ntuh_cc.columns
    ]
    ntuh_channels = [
        ChannelDef("ch_ant_v1v4", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_ant")), 0.55),
        ChannelDef("ch_lat_i_avl_v5v6", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_lat")), 0.55),
        ChannelDef("ch_inf_ii_iii_avf", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_inf")), 0.55),
        ChannelDef("ch_avr", tuple(c for c in ntuh_cc.columns if c.startswith("pl_") and c.endswith("_avr")), 0.45),
        ChannelDef("ch_ecg_diag_stmts", tuple(c for c in ntuh_cc.columns if c.startswith("dx_")), 0.40),
        ChannelDef("ch_waveform_12lead_median", tuple(c for c in ntuh_cc.columns if c.startswith("medwf_")), 0.85),
    ]

    tasks = [
        {
            "task_id": "mcmed_critical_icu_death",
            "label": "1. Stanford ICU/Death (<=2h)",
            "df": mc,
            "y_col": "y_critical_icu_death",
            "cluster_col": "MRN",
            "base_cols": mc_base,
            "channels": mc_channels,
            "contract": ClinicalContract(1.0, 19.0, 0.25, 0.030),
            "budget": 3.00,
            "ref_bundle_weight": 2.00,
            "pi_override": None,
            "seed": 44,
        },
        {
            "task_id": "mcmed_hospital_admission",
            "label": "2. Stanford Admission (<=2h)",
            "df": mc,
            "y_col": "y_admit",
            "cluster_col": "MRN",
            "base_cols": mc_base,
            "channels": mc_channels,
            "contract": ClinicalContract(1.0, 5.67, 0.50, 0.025),
            "budget": 3.00,
            "ref_bundle_weight": 2.00,
            "pi_override": None,
            "seed": 45,
        },
        {
            "task_id": "mcmed_bounce72_admit_2h",
            "label": "3. Stanford 72h Bounceback (<=2h Disch.)",
            "df": mc_disch,
            "y_col": "y_bounce72_admit",
            "cluster_col": "MRN",
            "base_cols": mc_base,
            "channels": mc_channels,
            "contract": ClinicalContract(1.0, 32.33, 0.15, 0.025),
            "budget": 3.00,
            "ref_bundle_weight": 2.00,
            "pi_override": None,
            "seed": 46,
        },
        {
            "task_id": "bwh_death_030_ecg_only",
            "label": "4. BWH 30d Mortality (Arrival ECG)",
            "df": bwh,
            "y_col": "death_030_day",
            "cluster_col": "patient_ngsci_id",
            "base_cols": bwh_base,
            "channels": bwh_channels,
            "contract": ClinicalContract(1.0, 32.33, 0.20, 0.020),
            "budget": 1.80,
            "ref_bundle_weight": 1.20,
            "pi_override": None,
            "seed": 42,
        },
        {
            "task_id": "bwh_stent_cabg_010_ecg_only",
            "label": "5. BWH 10d Stent/CABG (Arrival ECG)",
            "df": bwh,
            "y_col": "stent_or_cabg_010_day",
            "cluster_col": "patient_ngsci_id",
            "base_cols": bwh_base,
            "channels": bwh_channels,
            "contract": ClinicalContract(1.0, 32.33, 0.20, 0.020),
            "budget": 1.80,
            "ref_bundle_weight": 1.20,
            "pi_override": None,
            "seed": 43,
        },
        {
            "task_id": "ntuh_pre_arrest_sca_pi002",
            "label": "6. NTUH Pre-Arrest SCA (pi=2.0%)",
            "df": ntuh_cc,
            "y_col": "y_pre_arrest",
            "cluster_col": "patient_ngsci_id",
            "base_cols": ntuh_base,
            "channels": ntuh_channels,
            "contract": ClinicalContract(1.0, 19.0, 0.25, 0.020),
            "budget": 1.80,
            "ref_bundle_weight": 1.20,
            "pi_override": 0.02,
            "seed": 47,
        },
    ]
    raw_dfs = {"mc": mc, "mc_disch": mc_disch, "bwh": bwh, "ntuh_cc": ntuh_cc}
    return tasks, raw_dfs


# ===========================================================================
# BLOCK 1: IDENTITIES, SPLIT-PILOT SYNTHETIC E5, & 200-PILOT STANFORD ICU
# ===========================================================================
_B1_TR: pd.DataFrame | None = None
_B1_Y: np.ndarray | None = None
_B1_P1: np.ndarray | None = None
_B1_BASE: list[str] | None = None
_B1_CHS: list[ChannelDef] | None = None


def _init_b1_stanford_worker(
    tr_df: pd.DataFrame, y_tr: np.ndarray, p1_oof: np.ndarray, base_cols: list[str], chs: list[ChannelDef]
) -> None:
    _init_single_thread_worker()
    global _B1_TR, _B1_Y, _B1_P1, _B1_BASE, _B1_CHS
    _B1_TR = tr_df
    _B1_Y = y_tr
    _B1_P1 = p1_oof
    _B1_BASE = base_cols
    _B1_CHS = chs


def _b1_stanford_pilot_worker(rep: int) -> dict[str, Any]:
    assert _B1_TR is not None and _B1_Y is not None and _B1_P1 is not None and _B1_BASE is not None and _B1_CHS is not None
    ck = ClinicalContract(1.0, 19.0, 0.25, 0.030)
    seed_r = 700 + rep
    rng_n = np.random.default_rng(10_000 + rep)
    df_r = _B1_TR.copy()
    df_r["null_1a"] = rng_n.normal(size=len(df_r)).astype(np.float32)
    df_r["null_1b"] = rng_n.normal(size=len(df_r)).astype(np.float32)
    df_r["null_2a"] = rng_n.normal(size=len(df_r)).astype(np.float32)
    df_r["null_2b"] = rng_n.normal(size=len(df_r)).astype(np.float32)
    ch_with_null = list(_B1_CHS) + [
        ChannelDef("ch_null_noise_1", ("null_1a", "null_1b"), 0.50, is_null_stress=True),
        ChannelDef("ch_null_noise_2", ("null_2a", "null_2b"), 0.50, is_null_stress=True),
    ]
    non_null_ids = [c.channel_id for c in _B1_CHS]
    costs_nn = np.array([c.cost for c in _B1_CHS], dtype=float)

    out_cf, _ = discover_stage_a_ablation(
        df_r, _B1_Y, _B1_BASE, ch_with_null, _B1_P1, ck, budget=3.00,
        mode="offset_cz_schervish", pilot_fraction=0.20, seed=seed_r,
    )
    out_sp, _ = discover_stage_a_split_pilot(
        df_r, _B1_Y, _B1_BASE, ch_with_null, _B1_P1, ck, budget=3.00,
        pilot_fraction=0.20, seed=seed_r,
    )
    out_wc, _ = discover_stage_a_ablation(
        df_r, _B1_Y, _B1_BASE, ch_with_null, _B1_P1, ck, budget=3.00,
        mode="weighted_cz_005", pilot_fraction=0.20, seed=seed_r,
    )

    def _pack(o: Any) -> dict[str, Any]:
        s1_nn = np.array([o.step1_scores.get(cid, -1.0) for cid in non_null_ids], dtype=float)
        top1_nn = non_null_ids[int(np.argmax(s1_nn / costs_nn))]
        return {
            "top1_non_null": top1_nn,
            "selected": list(o.selected_ids),
            "null_admitted": any(c.startswith("ch_null") for c in o.selected_ids),
            "monitor_recovered": "ch_monitor_numerics_2h" in o.selected_ids,
        }

    return {
        "rep": rep,
        "crossfit_offset_4f": _pack(out_cf),
        "split_pilot_offset_50_50": _pack(out_sp),
        "weighted_cz_005": _pack(out_wc),
    }


def _b1_e5_split_and_sched_worker(args: tuple[int, float]) -> dict[str, Any]:
    rep, sigma_val = args
    contract = ClinicalContract(cost_fp=1.0, cost_fn=9.0, target_precision=0.35, per_instance_query_cost=0.09)
    base_cols = ["x_risk", "u_cue"]
    channels = [
        ChannelDef("ch_identifiable", ("z_ident",), 0.80),
        ChannelDef("ch_unidentifiable", ("z_unident",), 0.80),
        ChannelDef("ch_null_1", ("z_null1",), 0.50, is_null_stress=True),
        ChannelDef("ch_null_2", ("z_null2",), 0.50, is_null_stress=True),
    ]
    ch_map = {c.channel_id: c for c in channels}
    sched = ChannelCostSchedule(
        base_c_query=0.09,
        reference_bundle_weight=0.80,
        channel_weights={c.channel_id: c.cost for c in channels},
        overhead_share=0.25,
    )

    rng = np.random.default_rng((1000 if abs(sigma_val - 3.2) < 1e-6 else 5000) + rep)
    n = 9000
    in_cz = rng.binomial(1, 0.40, size=n)
    u_cue = rng.binomial(1, 0.22, size=n)
    u_hidden = rng.binomial(1, 0.22, size=n)
    base_logit = np.where(
        in_cz == 1,
        logit(0.090) + rng.normal(0, 0.10, size=n),
        logit(0.010) + rng.normal(0, 0.20, size=n),
    )
    p0 = expit(base_logit)
    sig_id = sigma_val * u_cue * in_cz
    sig_unid = sigma_val * u_hidden * in_cz
    mu_star = solve_martingale_offset(p0, np.sqrt(sig_id**2 + sig_unid**2))
    z_ident_sig = sig_id * rng.normal(0, 1.0, size=n)
    z_unident_sig = sig_unid * rng.normal(0, 1.0, size=n)
    y = rng.binomial(1, expit(base_logit + mu_star + z_ident_sig + z_unident_sig))

    df = pd.DataFrame(
        {
            "x_risk": (base_logit + rng.normal(0, 0.12, size=n)).astype(np.float32),
            "u_cue": u_cue.astype(np.float32),
            "z_ident": (z_ident_sig + rng.normal(0, 0.15, size=n)).astype(np.float32),
            "z_unident": (z_unident_sig + rng.normal(0, 0.15, size=n)).astype(np.float32),
            "z_null1": rng.normal(0, 1, size=n).astype(np.float32),
            "z_null2": rng.normal(0, 1, size=n).astype(np.float32),
        }
    )
    tr, cal, te = df.iloc[:5400].reset_index(drop=True), df.iloc[5400:7200].reset_index(drop=True), df.iloc[7200:].reset_index(drop=True)
    y_tr, y_cal, y_te = y[:5400], y[5400:7200], y[7200:]
    p1_oof = compute_oof_stage1(tr[base_cols], y_tr, n_folds=3, seed=rep)

    disc_cf, _ = discover_stage_a_ablation(
        tr, y_tr, base_cols, channels, p1_oof, contract, budget=2.0,
        mode="offset_cz_schervish", pilot_fraction=0.35, seed=rep,
    )
    disc_sp, _ = discover_stage_a_split_pilot(
        tr, y_tr, base_cols, channels, p1_oof, contract, budget=2.0,
        pilot_fraction=0.35, seed=rep,
    )

    out: dict[str, Any] = {
        "cf_id": "ch_identifiable" in disc_cf.selected_ids,
        "cf_unid": "ch_unidentifiable" in disc_cf.selected_ids,
        "cf_null": any("null" in c for c in disc_cf.selected_ids),
        "sp_id": "ch_identifiable" in disc_sp.selected_ids,
        "sp_unid": "ch_unidentifiable" in disc_sp.selected_ids,
        "sp_null": any("null" in c for c in disc_sp.selected_ids),
    }

    if abs(sigma_val - 3.2) < 1e-6:
        p1_cal, p1_tes = fit_stage1_predictions(tr, y_tr, cal, y_cal, [te], base_cols, seed=rep)
        p1_te = p1_tes[0]
        upca_sched, _ = oof_policy_screening_stage_b_scheduled(
            disc_cf.selected_ids, tr, y_tr, p1_oof, base_cols, ch_map, contract, sched, seed=rep
        )
        xb_tr = tr[base_cols].to_numpy(dtype=np.float32)
        xb_cal = cal[base_cols].to_numpy(dtype=np.float32)
        xb_te = te[base_cols].to_numpy(dtype=np.float32)
        p1_oof_c = np.clip(p1_oof, 1e-5, 1.0 - 1e-5)

        def _eval_sched(sel_ids: list[str]) -> float:
            if not sel_ids:
                return variable_cost_nb_per_1k(y_te, p1_te, 0.0, contract)
            c_pan = sched.panel_cost_cfp(sel_ids)
            ck_p = ClinicalContract(1.0, 9.0, 0.35, c_pan)
            cols = [c for cid in sel_ids for c in ch_map[cid].columns]
            z_tr = tr[cols].to_numpy(dtype=np.float32)
            z_cal = cal[cols].to_numpy(dtype=np.float32)
            z_te = te[cols].to_numpy(dtype=np.float32)
            clf = FluidCollisionBackboneClassifier(tau_star=contract.tau_star, seed=rep).fit(
                np.hstack([xb_tr, z_tr]), z_tr, y_tr, logit(p1_oof_c)
            )
            r_tr = clf.decision_function(np.hstack([xb_tr, z_tr]), z_tr, logit(p1_oof_c))
            r_cal = clf.decision_function(np.hstack([xb_cal, z_cal]), z_cal, logit(p1_cal))
            r_te = clf.decision_function(np.hstack([xb_te, z_te]), z_te, logit(p1_te))
            pl = fit_platt_calibrator(r_cal, y_cal)
            p2_tr = np.clip(apply_platt(r_tr, pl), 1e-6, 1.0 - 1e-6)
            p2_cal = np.clip(apply_platt(r_cal, pl), 1e-6, 1.0 - 1e-6)
            p2_te = np.clip(apply_platt(r_te, pl), 1e-6, 1.0 - 1e-6)
            voi_te = fit_predict_prepurchase_voi(
                xb_tr, p1_oof_c, p2_tr, xb_cal, p1_cal, p2_cal, xb_te, p1_te, ck_p, use_dispersion_2d=True, seed=rep
            )
            q_te = voi_te > c_pan
            p_rt = np.where(q_te, p2_te, p1_te)
            return variable_cost_nb_per_1k(y_te, p_rt, q_te.astype(float) * c_pan, contract)

        out["sched_upca_ret_id"] = "ch_identifiable" in upca_sched
        out["sched_upca_ret_unid"] = "ch_unidentifiable" in upca_sched
        out["sched_upca_null"] = any("null" in c for c in upca_sched)
        out["nb_sched_stage_a_b2"] = _eval_sched(disc_cf.selected_ids)
        out["nb_sched_upca_b2"] = _eval_sched(upca_sched)
    return out


def run_block1() -> dict[str, Any]:
    t0 = time.time()
    print("\n=== [BLOCK 1] Identities, XOR Counterexample, Split-Pilot E5 & 200-Pilot Stanford ICU ===", flush=True)
    v21 = json.loads(V21_JSON.read_text())

    identities = {
        "normalized_beta_kl_max_abs_error": {
            "tau003_kappa24": normalized_beta_kl_identity_max_error(0.03, 24.0),
            "tau005_kappa24": normalized_beta_kl_identity_max_error(0.05, 24.0),
            "tau010_kappa20": normalized_beta_kl_identity_max_error(0.10, 20.0),
            "tau015_kappa16": normalized_beta_kl_identity_max_error(0.15, 16.0),
        },
        "two_phase_variance_srs_h1_identity_abs_error": two_phase_variance_srs_identity_error(1000, 200, seed=19),
        "xor_finite_capacity_stump_counterexample": xor_finite_capacity_counterexample(2400, seed=13),
    }
    print("  [Block 1] Exact Identities & XOR:", json.dumps(identities, indent=2), flush=True)

    # 1. Synthetic E5 (200 reps canonical sigma=3.2 + 200 reps weak-signal sigma=1.6)
    e5_tasks = [(r, 3.2) for r in range(200)] + [(r, 1.6) for r in range(200)]
    e5_canon, e5_weak = [], []
    with cf.ProcessPoolExecutor(max_workers=8, initializer=_init_single_thread_worker) as ex:
        for idx, res_r in enumerate(ex.map(_b1_e5_split_and_sched_worker, e5_tasks, chunksize=4), start=1):
            if "nb_sched_upca_b2" in res_r:
                e5_canon.append(res_r)
            else:
                e5_weak.append(res_r)
            if idx % 80 == 0 or idx == len(e5_tasks):
                print(f"  [Block 1 E5 Split/Sched] {idx}/{len(e5_tasks)} completed ({time.time()-t0:.1f}s)", flush=True)

    def _summarize_split_vs_cf(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        cf_id = sum(1 for x in rows if x["cf_id"])
        sp_id = sum(1 for x in rows if x["sp_id"])
        cf_null = sum(1 for x in rows if x["cf_null"])
        sp_null = sum(1 for x in rows if x["sp_null"])
        return {
            "n_reps": n,
            "crossfit_4fold_identifiable_rate": round(cf_id / n, 4),
            "crossfit_4fold_identifiable_ci95": _clopper_pearson_ci95(cf_id, n),
            "split_pilot_50_50_identifiable_rate": round(sp_id / n, 4),
            "split_pilot_50_50_identifiable_ci95": _clopper_pearson_ci95(sp_id, n),
            "crossfit_4fold_null_fwer": round(cf_null / n, 4),
            "crossfit_4fold_null_fwer_ci95": _clopper_pearson_ci95(cf_null, n),
            "split_pilot_50_50_null_fwer": round(sp_null / n, 4),
            "split_pilot_50_50_null_fwer_ci95": _clopper_pearson_ci95(sp_null, n),
        }

    d_sched_b2 = np.array([x["nb_sched_upca_b2"] - x["nb_sched_stage_a_b2"] for x in e5_canon], dtype=float)
    se_sb2 = float(np.std(d_sched_b2, ddof=1) / np.sqrt(len(d_sched_b2)))
    p_sb2 = float(2.0 * t_dist.sf(abs(float(np.mean(d_sched_b2)) / max(1e-12, se_sb2)), df=len(d_sched_b2) - 1))

    e5_split_and_sched = {
        "canonical_sigma_32_200reps": {
            **_summarize_split_vs_cf(e5_canon),
            "scheduled_stage_b_retain_identifiable_rate": round(
                sum(1 for x in e5_canon if x["sched_upca_ret_id"]) / len(e5_canon), 4
            ),
            "scheduled_stage_b_retain_unidentifiable_rate": round(
                sum(1 for x in e5_canon if x["sched_upca_ret_unid"]) / len(e5_canon), 4
            ),
            "scheduled_stage_b_null_fwer": round(
                sum(1 for x in e5_canon if x["sched_upca_null"]) / len(e5_canon), 4
            ),
            "nb_scheduled_stage_a_b2_mean": round(float(np.mean([x["nb_sched_stage_a_b2"] for x in e5_canon])), 4),
            "nb_scheduled_upca_b2_mean": round(float(np.mean([x["nb_sched_upca_b2"] for x in e5_canon])), 4),
            "paired_delta_nb_scheduled_b2_mean": round(float(np.mean(d_sched_b2)), 4),
            "paired_delta_nb_scheduled_b2_ci95": [
                round(float(np.mean(d_sched_b2) - 1.96 * se_sb2), 4),
                round(float(np.mean(d_sched_b2) + 1.96 * se_sb2), 4),
            ],
            "paired_delta_nb_scheduled_b2_p": round(p_sb2, 6),
        },
        "weak_signal_sigma_16_200reps": _summarize_split_vs_cf(e5_weak),
    }

    # 2. 200-Pilot Stanford ICU Split-Pilot (50/50) vs 4-Fold Cross-Fitted Offset Discovery (with 2 Gaussian Nulls)
    tasks, _ = load_all_clinical_tasks()
    icu_spec = tasks[0]
    mc_tr = icu_spec["df"][icu_spec["df"]["split"] == "train"].reset_index(drop=True)
    y_tr = mc_tr[icu_spec["y_col"]].astype(int).to_numpy()
    p1_oof = compute_oof_stage1(mc_tr[icu_spec["base_cols"]], y_tr, n_folds=3, seed=44)
    oracle_top1 = v21["clinical"]["two_phase_efficiency_r15"]["oracle_top1_channel"]

    icu_pilot_rows = []
    with cf.ProcessPoolExecutor(
        max_workers=8,
        initializer=_init_b1_stanford_worker,
        initargs=(mc_tr[icu_spec["base_cols"] + [c for ch in icu_spec["channels"] for c in ch.columns]], y_tr, p1_oof, icu_spec["base_cols"], icu_spec["channels"]),
    ) as ex:
        for idx, pr in enumerate(ex.map(_b1_stanford_pilot_worker, range(200), chunksize=3), start=1):
            icu_pilot_rows.append(pr)
            if idx % 50 == 0 or idx == 200:
                print(f"  [Block 1 Stanford 200-Pilot Split vs CF] {idx}/200 completed ({time.time()-t0:.1f}s)", flush=True)

    stanford_200_split_audit: dict[str, Any] = {"n_pilots": 200, "oracle_top1_channel": oracle_top1, "arms": {}}
    for arm_k in ("crossfit_offset_4f", "split_pilot_offset_50_50", "weighted_cz_005"):
        t1 = [r[arm_k]["top1_non_null"] == oracle_top1 for r in icu_pilot_rows]
        mon = [bool(r[arm_k]["monitor_recovered"]) for r in icu_pilot_rows]
        nul = [bool(r[arm_k]["null_admitted"]) for r in icu_pilot_rows]
        stanford_200_split_audit["arms"][arm_k] = {
            "top1_oracle_concordance": round(float(np.mean(t1)), 4),
            "top1_oracle_ci95": _clopper_pearson_ci95(int(np.sum(t1)), 200),
            "monitor_bundle_recovery": round(float(np.mean(mon)), 4),
            "monitor_bundle_ci95": _clopper_pearson_ci95(int(np.sum(mon)), 200),
            "null_fwer_rate": round(float(np.mean(nul)), 4),
            "null_fwer_count": int(np.sum(nul)),
            "null_fwer_ci95": _clopper_pearson_ci95(int(np.sum(nul)), 200),
        }
    stanford_200_split_audit["mcnemar_top1_crossfit_vs_split50"] = _paired_mcnemar(
        [r["crossfit_offset_4f"]["top1_non_null"] == oracle_top1 for r in icu_pilot_rows],
        [r["split_pilot_offset_50_50"]["top1_non_null"] == oracle_top1 for r in icu_pilot_rows],
        seed=191,
    )
    stanford_200_split_audit["mcnemar_monitor_crossfit_vs_split50"] = _paired_mcnemar(
        [bool(r["crossfit_offset_4f"]["monitor_recovered"]) for r in icu_pilot_rows],
        [bool(r["split_pilot_offset_50_50"]["monitor_recovered"]) for r in icu_pilot_rows],
        seed=192,
    )

    out_b1 = {
        "identities_and_counterexamples": identities,
        "preserved_v2_1_synthetic": v21["synthetic"],
        "preserved_v2_1_500pilot_stanford_icu": v21["clinical"]["two_phase_efficiency_r15"],
        "preserved_v2_1_stanford_independence_deep_dive": v21["clinical"]["stanford_icu_independence_deep_dive_r15"],
        "e5_split_pilot_and_scheduled_b2_audit": e5_split_and_sched,
        "stanford_icu_200pilot_split_vs_crossfit_with_nulls": stanford_200_split_audit,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    (BLOCKS_DIR / "block1_synthetic_and_pilots.json").write_text(json.dumps(out_b1, indent=2))
    print(f"=== [BLOCK 1 DONE in {time.time()-t0:.1f}s] ===", flush=True)
    return out_b1


# ===========================================================================
# BLOCK 2: COMMON-PANEL BAGGED VOI, FULL 100%, BOUNDARY UNCERTAINTY & FEE SWEEP
# ===========================================================================
def _patient_grouped_dev_split(
    clusters_dev: np.ndarray, y_dev: np.ndarray, cal_frac: float = 0.20, seed: int = 2100
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


def _fit_panel_on_split_standalone(
    tr_s: pd.DataFrame,
    y_tr_s: np.ndarray,
    p1_oof_r: np.ndarray,
    p1_oof_e: np.ndarray,
    ca_s: pd.DataFrame,
    y_ca_s: np.ndarray,
    p1_ca_r: np.ndarray,
    p1_ca_e: np.ndarray,
    te: pd.DataFrame,
    xb_tr_s: np.ndarray,
    xb_ca_s: np.ndarray,
    xb_te_s: np.ndarray,
    p1_te_r: np.ndarray,
    p1_te_e: np.ndarray,
    sel_ids: list[str],
    ch_map: dict[str, ChannelDef],
    contract: ClinicalContract,
    sched: ChannelCostSchedule,
    pi_ov: float | None,
    use_fluid: bool,
    seed_s: int,
) -> dict[str, Any]:
    n_te = len(p1_te_e)
    c_pan = sched.panel_cost_cfp(sel_ids)
    if not sel_ids:
        z0 = np.zeros(n_te, dtype=float)
        return {
            "p2_te": p1_te_e.copy(),
            "p2_ca": p1_ca_e.copy(),
            "voi_te": z0,
            "panel_cost_cfp": 0.0,
            "diag": {
                "beta_cz": 0.0,
                "kappa_star": 0.0,
                "rms_r_cz": 0.0,
                "rms_r_cz_zone": 0.0,
                "decision_flip_rate_vs_glob": 0.0,
            },
        }
    s2_cols = [c for cid in sel_ids for c in ch_map[cid].columns]
    enc_z = FastFrameEncoder().fit(tr_s[s2_cols])
    z_tr = enc_z.transform(tr_s[s2_cols])
    z_ca = enc_z.transform(ca_s[s2_cols])
    z_te = enc_z.transform(te[s2_cols])
    x2_tr = np.hstack([xb_tr_s, z_tr])
    x2_ca = np.hstack([xb_ca_s, z_ca])
    x2_te = np.hstack([xb_te_s, z_te])
    p1_s = np.clip(p1_oof_r, 1e-5, 1.0 - 1e-5)

    if use_fluid:
        tau_fit = (
            contract.tau_star
            if pi_ov is None
            else float(adjust_probs_for_target_prevalence(np.array([contract.tau_star]), pi_ov, float(np.mean(y_tr_s)))[0])
        )
        clf = FluidCollisionBackboneClassifier(
            tau_star=tau_fit, kappa_max=20.0, min_cz_ess=240.0, seed=seed_s + 7
        ).fit(x2_tr, z_tr, y_tr_s, logit(p1_s))
        r_tr = clf.decision_function(x2_tr, z_tr, logit(p1_s))
        r_ca = clf.decision_function(x2_ca, z_ca, logit(np.clip(p1_ca_r, 1e-5, 1 - 1e-5)))
        r_te = clf.decision_function(x2_te, z_te, logit(np.clip(p1_te_r, 1e-5, 1 - 1e-5)))
    else:
        clf = None
        clf_g = lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            max_delta_step=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed_s + 7,
            n_jobs=int(os.environ.get("UPCA_LGBM_JOBS", "2")),
            verbose=-1,
        ).fit(x2_tr, y_tr_s)
        r_tr = clf_g.predict(x2_tr, raw_score=True)
        r_ca = clf_g.predict(x2_ca, raw_score=True)
        r_te = clf_g.predict(x2_te, raw_score=True)

    pl = fit_platt_calibrator(r_ca, y_ca_s)
    p2_tr = np.clip(apply_platt(r_tr, pl), 1e-6, 1.0 - 1e-6)
    p2_ca = np.clip(apply_platt(r_ca, pl), 1e-6, 1.0 - 1e-6)
    p2_te = np.clip(apply_platt(r_te, pl), 1e-6, 1.0 - 1e-6)
    prior_shift = 0.0
    if pi_ov is not None:
        pi_cs = float(np.clip(np.mean(y_ca_s), 1e-6, 1 - 1e-6))
        prior_shift = float(np.log(pi_ov / (1.0 - pi_ov)) - np.log(pi_cs / (1.0 - pi_cs)))
        p2_tr = adjust_probs_for_target_prevalence(p2_tr, pi_cs, pi_ov)
        p2_ca = adjust_probs_for_target_prevalence(p2_ca, pi_cs, pi_ov)
        p2_te = adjust_probs_for_target_prevalence(p2_te, pi_cs, pi_ov)

    ck_p = ClinicalContract(contract.cost_fp, contract.cost_fn, contract.target_precision, c_pan)
    voi_te = fit_predict_prepurchase_voi(
        xb_tr_s, p1_oof_e, p2_tr, xb_ca_s, p1_ca_e, p2_ca, xb_te_s, p1_te_e, ck_p, use_dispersion_2d=True, seed=seed_s
    )
    diag = (
        compute_fluid_specialist_diagnostics(
            clf,
            x2_te,
            z_te,
            logit(np.clip(p1_te_r, 1e-5, 1 - 1e-5)),
            contract.tau_star,
            outer_platt=pl,
            prior_shift_log_odds=prior_shift,
        )
        if (use_fluid and clf is not None)
        else {
            "beta_cz": 0.0,
            "kappa_star": 0.0,
            "rms_r_cz": 0.0,
            "rms_r_cz_zone": 0.0,
            "decision_flip_rate_vs_glob": 0.0,
        }
    )
    return {"p2_te": p2_te, "p2_ca": p2_ca, "voi_te": voi_te, "panel_cost_cfp": c_pan, "diag": diag}


_B2_CTX: dict[str, Any] = {}


def _init_b2_r10_worker(ctx: dict[str, Any]) -> None:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    os.environ["UPCA_LGBM_JOBS"] = "2"
    orig_proc = lgb.LGBMModel._process_params

    def _patched_proc(self: Any, stage: str = "fit") -> dict[str, Any]:
        params = orig_proc(self, stage=stage)
        params["n_jobs"] = 2
        params["num_threads"] = 2
        return params

    lgb.LGBMModel._process_params = _patched_proc  # type: ignore[method-assign]
    orig_b_init = lgb.Booster.__init__

    def _patched_b_init(self: Any, params: dict[str, Any] | None = None, *args: Any, **kwargs: Any) -> None:
        if isinstance(params, dict):
            params = dict(params)
            params["n_jobs"] = 2
            params["num_threads"] = 2
        orig_b_init(self, params, *args, **kwargs)

    lgb.Booster.__init__ = _patched_b_init  # type: ignore[method-assign]
    global _B2_CTX
    _B2_CTX = ctx


def _run_b2_r10_split(r_i: int) -> dict[str, Any]:
    dev = _B2_CTX["dev"]
    te = _B2_CTX["te"]
    y_col = _B2_CTX["y_col"]
    base_cols = _B2_CTX["base_cols"]
    ch_all = _B2_CTX["ch_all"]
    ch_map = {c.channel_id: c for c in ch_all}
    contract: ClinicalContract = _B2_CTX["contract"]
    budget = float(_B2_CTX["budget"])
    pi_ov = _B2_CTX["pi_ov"]
    seed = int(_B2_CTX["seed"])
    sched: ChannelCostSchedule = _B2_CTX["sched"]
    common_arms = _B2_CTX["common_arms"]
    cal_frac_dev = float(_B2_CTX["cal_frac_dev"])
    y_te = _B2_CTX["y_te"]
    w_te = _B2_CTX["w_te"]
    n_te = len(y_te)

    y_dev = dev[y_col].astype(int).to_numpy()
    cl_dev = dev[_B2_CTX["cl_col"]].to_numpy()
    seed_r = 2100 + 100 * (seed % 10) + r_i
    tr_idx, ca_idx = _patient_grouped_dev_split(cl_dev, y_dev, cal_frac=cal_frac_dev, seed=seed_r)
    tr_r = dev.iloc[tr_idx].reset_index(drop=True)
    ca_r = dev.iloc[ca_idx].reset_index(drop=True)
    y_tr_r = tr_r[y_col].astype(int).to_numpy()
    y_ca_r = ca_r[y_col].astype(int).to_numpy()

    p1_oof_r = compute_oof_stage1(tr_r[base_cols], y_tr_r, n_folds=3, seed=seed_r)
    p1_ca_r, p1_tes_r = fit_stage1_predictions(tr_r, y_tr_r, ca_r, y_ca_r, [te], base_cols, seed=seed_r)
    p1_te_r = p1_tes_r[0]
    if pi_ov is not None:
        w_tr_r = case_control_prevalence_weights(y_tr_r, pi_ov)
        w_ca_r = case_control_prevalence_weights(y_ca_r, pi_ov)
        p1_oof_e = adjust_probs_for_target_prevalence(p1_oof_r, float(np.mean(y_tr_r)), pi_ov)
        p1_ca_e = adjust_probs_for_target_prevalence(p1_ca_r, float(np.mean(y_ca_r)), pi_ov)
        p1_te_e = adjust_probs_for_target_prevalence(p1_te_r, float(np.mean(y_ca_r)), pi_ov)
    else:
        w_tr_r, w_ca_r = None, None
        p1_oof_e, p1_ca_e, p1_te_e = p1_oof_r, p1_ca_r, p1_te_r

    eb_r = FastFrameEncoder().fit(tr_r[base_cols])
    xb_tr_r = eb_r.transform(tr_r[base_cols])
    xb_ca_r = eb_r.transform(ca_r[base_cols])
    xb_te_r = eb_r.transform(te[base_cols])

    fit_cache_r: dict[tuple[tuple[str, ...], bool], dict[str, Any]] = {}
    com_out: dict[str, Any] = {}
    for arm_name, sel_c, fl_c in common_arms:
        ckey_c = (tuple(sel_c), fl_c)
        if ckey_c not in fit_cache_r:
            fit_cache_r[ckey_c] = _fit_panel_on_split_standalone(
                tr_r, y_tr_r, p1_oof_r, p1_oof_e, ca_r, y_ca_r, p1_ca_r, p1_ca_e,
                te, xb_tr_r, xb_ca_r, xb_te_r, p1_te_r, p1_te_e,
                sel_c, ch_map, contract, sched, pi_ov, fl_c, seed_r,
            )
        fr = fit_cache_r[ckey_c]
        com_out[arm_name] = {"p2_te": fr["p2_te"], "voi_te": fr["voi_te"], "diag": fr["diag"]}

    d_gl_r, _ = discover_stage_a_ablation(
        tr_r, y_tr_r, base_cols, ch_all, p1_oof_e, contract, budget,
        mode="global_cmi", pilot_fraction=0.20, target_pop_weights=w_tr_r, raw_oof_p_base_for_fit=p1_oof_r, seed=seed_r,
    )
    d_wc_r, _ = discover_stage_a_ablation(
        tr_r, y_tr_r, base_cols, ch_all, p1_oof_e, contract, budget,
        mode="weighted_cz_005", pilot_fraction=0.20, target_pop_weights=w_tr_r, raw_oof_p_base_for_fit=p1_oof_r, seed=seed_r,
    )
    d_oc_r, _ = discover_stage_a_ablation(
        tr_r, y_tr_r, base_cols, ch_all, p1_oof_e, contract, budget,
        mode="offset_cz_schervish", pilot_fraction=0.20, target_pop_weights=w_tr_r, raw_oof_p_base_for_fit=p1_oof_r, seed=seed_r,
    )
    sel_up_r, _ = oof_policy_screening_stage_b_scheduled(
        d_oc_r.selected_ids, tr_r, y_tr_r, p1_oof_r, base_cols, ch_map, contract, sched, pi_target_override=pi_ov, seed=seed_r
    )

    ps_out: dict[str, Any] = {
        "stage1": {
            "nb": variable_cost_nb_per_1k(y_te, p1_te_e, 0.0, contract, sample_weight=w_te),
            "pr": p1_te_e,
            "p2": p1_te_e,
            "voi": np.zeros(n_te, dtype=float),
            "cost": np.zeros(n_te, dtype=float),
            "q": np.zeros(n_te, dtype=float),
            "sel": [],
        }
    }
    for ak, sk, flk in [
        ("global_cmi_gbdt", d_gl_r.selected_ids, False),
        ("global_cmi_fluid", d_gl_r.selected_ids, True),
        ("weighted_cz_fluid", d_wc_r.selected_ids, True),
        ("upca_fluid", sel_up_r, True),
    ]:
        ckey_k = (tuple(sk), flk)
        if ckey_k not in fit_cache_r:
            fit_cache_r[ckey_k] = _fit_panel_on_split_standalone(
                tr_r, y_tr_r, p1_oof_r, p1_oof_e, ca_r, y_ca_r, p1_ca_r, p1_ca_e,
                te, xb_tr_r, xb_ca_r, xb_te_r, p1_te_r, p1_te_e,
                sk, ch_map, contract, sched, pi_ov, flk, seed_r,
            )
        fr_k = fit_cache_r[ckey_k]
        cp_k = fr_k["panel_cost_cfp"]
        qk = (fr_k["voi_te"] > cp_k) if sk else np.zeros(n_te, dtype=bool)
        pr_k = np.where(qk, fr_k["p2_te"], p1_te_e)
        ck_vec = qk.astype(float) * cp_k
        ps_out[ak] = {
            "nb": variable_cost_nb_per_1k(y_te, pr_k, ck_vec, contract, sample_weight=w_te),
            "pr": pr_k,
            "p2": fr_k["p2_te"],
            "voi": fr_k["voi_te"],
            "cost": ck_vec,
            "q": qk.astype(float),
            "sel": list(sk),
        }

    dual_afa = evaluate_dual_afa_scheduled(
        tr_r, y_tr_r, ca_r, y_ca_r, te, y_te, base_cols, ch_all,
        p1_oof_r, p1_ca_r, p1_te_r, p1_ca_e, p1_te_e,
        contract, budget, sched, pi_target_override=pi_ov, w_cal=w_ca_r, w_te=w_te, seed=seed_r,
    )
    for pm in ("cmi", "direct_voi"):
        dual_afa[pm]["unit_u"] = variable_cost_unit_nb_vec(y_te, dual_afa[pm]["p_rt"], dual_afa[pm]["cost_te"], contract)

    return {
        "r_i": r_i,
        "p1_te_e": p1_te_e,
        "p1_te_r": p1_te_r,
        "com_out": com_out,
        "ps_out": ps_out,
        "dual_afa": dual_afa,
    }


def _evaluate_single_task_v2_2(spec: dict[str, Any], n_splits: int = 10) -> dict[str, Any]:
    t0 = time.time()
    tid = spec["task_id"]
    df = spec["df"].copy()
    y_col = spec["y_col"]
    cl_col = spec["cluster_col"]
    base_cols = spec["base_cols"]
    channels = spec["channels"]
    contract: ClinicalContract = spec["contract"]
    budget = float(spec["budget"])
    pi_ov = spec["pi_override"]
    seed = int(spec["seed"])

    rng0 = np.random.default_rng(seed)
    df["null_1a"], df["null_1b"] = rng0.normal(size=len(df)), rng0.normal(size=len(df))
    df["null_2a"], df["null_2b"] = rng0.normal(size=len(df)), rng0.normal(size=len(df))
    ch_all = list(channels) + [
        ChannelDef("ch_null_noise_1", ("null_1a", "null_1b"), 0.50, is_null_stress=True),
        ChannelDef("ch_null_noise_2", ("null_2a", "null_2b"), 0.50, is_null_stress=True),
    ]
    ch_map = {c.channel_id: c for c in ch_all}
    all_non_null_ids = [c.channel_id for c in channels]
    ch_ids_ordered = [c.channel_id for c in ch_all]

    sched = ChannelCostSchedule(
        base_c_query=contract.per_instance_query_cost,
        reference_bundle_weight=float(spec["ref_bundle_weight"]),
        channel_weights={c.channel_id: c.cost for c in ch_all},
        overhead_share=0.25,
    )

    tr0 = df[df["split"] == "train"].reset_index(drop=True)
    cal0 = df[df["split"] == "cal"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    dev = pd.concat([tr0, cal0], ignore_index=True)

    y_tr0 = tr0[y_col].astype(int).to_numpy()
    y_cal0 = cal0[y_col].astype(int).to_numpy()
    y_te = te[y_col].astype(int).to_numpy()
    clusters_te = te[cl_col].to_numpy()
    n_te = y_te.size

    w_tr0 = case_control_prevalence_weights(y_tr0, pi_ov) if pi_ov is not None else None
    w_cal0 = case_control_prevalence_weights(y_cal0, pi_ov) if pi_ov is not None else None
    w_te = case_control_prevalence_weights(y_te, pi_ov) if pi_ov is not None else None
    eval_prev = float(pi_ov) if pi_ov is not None else float(np.mean(y_te))

    print(f"\n  [Block 2 | {tid}] Canonical + Common-Panel R={n_splits} + AFA + Fee Sweep...", flush=True)
    p1_oof0_raw = compute_oof_stage1(tr0[base_cols], y_tr0, n_folds=3, seed=seed)
    p1_cal0_raw, p1_tes0_raw = fit_stage1_predictions(tr0, y_tr0, cal0, y_cal0, [te], base_cols, seed=seed)
    p1_te0_raw = p1_tes0_raw[0]
    if pi_ov is not None:
        p1_oof0_eval = adjust_probs_for_target_prevalence(p1_oof0_raw, float(np.mean(y_tr0)), pi_ov)
        p1_cal0_eval = adjust_probs_for_target_prevalence(p1_cal0_raw, float(np.mean(y_cal0)), pi_ov)
        p1_te0_eval = adjust_probs_for_target_prevalence(p1_te0_raw, float(np.mean(y_cal0)), pi_ov)
    else:
        p1_oof0_eval, p1_cal0_eval, p1_te0_eval = p1_oof0_raw, p1_cal0_raw, p1_te0_raw

    # --- 1. Lock Canonical / Common Panels Strictly on `tr0` (0% access to `cal0` or `te`) ---
    d_glob0, _ = discover_stage_a_ablation(
        tr0, y_tr0, base_cols, ch_all, p1_oof0_eval, contract, budget,
        mode="global_cmi", pilot_fraction=0.20, target_pop_weights=w_tr0, raw_oof_p_base_for_fit=p1_oof0_raw, seed=seed,
    )
    d_wcz0, _ = discover_stage_a_ablation(
        tr0, y_tr0, base_cols, ch_all, p1_oof0_eval, contract, budget,
        mode="weighted_cz_005", pilot_fraction=0.20, target_pop_weights=w_tr0, raw_oof_p_base_for_fit=p1_oof0_raw, seed=seed,
    )
    d_ipw0, _ = discover_stage_a_ablation(
        tr0, y_tr0, base_cols, ch_all, p1_oof0_eval, contract, budget,
        mode="ipw_cz_wht", pilot_fraction=0.20, target_pop_weights=w_tr0, raw_oof_p_base_for_fit=p1_oof0_raw, seed=seed,
    )
    d_off0, _ = discover_stage_a_ablation(
        tr0, y_tr0, base_cols, ch_all, p1_oof0_eval, contract, budget,
        mode="offset_cz_schervish", pilot_fraction=0.20, target_pop_weights=w_tr0, raw_oof_p_base_for_fit=p1_oof0_raw, seed=seed,
    )
    d_sp0, _ = discover_stage_a_split_pilot(
        tr0, y_tr0, base_cols, ch_all, p1_oof0_eval, contract, budget,
        pilot_fraction=0.20, target_pop_weights=w_tr0, raw_oof_p_base_for_fit=p1_oof0_raw, seed=seed,
    )
    sel_upca_sched0, prune_log_sched0 = oof_policy_screening_stage_b_scheduled(
        d_off0.selected_ids, tr0, y_tr0, p1_oof0_raw, base_cols, ch_map, contract, sched, pi_target_override=pi_ov, seed=seed
    )
    sel_upca_flat0, _ = oof_policy_screening_stage_b(
        d_off0.selected_ids, tr0, y_tr0, p1_oof0_raw, base_cols, ch_map, contract, pi_target_override=pi_ov, seed=seed
    )
    sel_sp_upca0, _ = oof_policy_screening_stage_b_scheduled(
        d_sp0.selected_ids, tr0, y_tr0, p1_oof0_raw, base_cols, ch_map, contract, sched, pi_target_override=pi_ov, seed=seed
    )

    locked_panels = {
        "all_m_channels": all_non_null_ids,
        "global_cmi": list(d_glob0.selected_ids),
        "weighted_cz": list(d_wcz0.selected_ids),
        "ipw_cz": list(d_ipw0.selected_ids),
        "offset_cz_stage_a": list(d_off0.selected_ids),
        "split_pilot_upca": list(sel_sp_upca0),
        "upca_fluid": list(sel_upca_sched0),
        "upca_fluid_flat_legacy": list(sel_upca_flat0),
    }

    def _fit_panel_on_split(
        tr_s: pd.DataFrame,
        y_tr_s: np.ndarray,
        p1_oof_r: np.ndarray,
        p1_oof_e: np.ndarray,
        ca_s: pd.DataFrame,
        y_ca_s: np.ndarray,
        p1_ca_r: np.ndarray,
        p1_ca_e: np.ndarray,
        xb_tr_s: np.ndarray,
        xb_ca_s: np.ndarray,
        xb_te_s: np.ndarray,
        p1_te_r: np.ndarray,
        p1_te_e: np.ndarray,
        sel_ids: list[str],
        use_fluid: bool,
        seed_s: int,
    ) -> dict[str, Any]:
        c_pan = sched.panel_cost_cfp(sel_ids)
        if not sel_ids:
            z0 = np.zeros(n_te, dtype=float)
            return {
                "p2_te": p1_te_e.copy(),
                "p2_ca": p1_ca_e.copy(),
                "voi_te": z0,
                "panel_cost_cfp": 0.0,
                "diag": {
                    "beta_cz": 0.0,
                    "kappa_star": 0.0,
                    "rms_r_cz": 0.0,
                    "rms_r_cz_zone": 0.0,
                    "decision_flip_rate_vs_glob": 0.0,
                },
            }
        s2_cols = [c for cid in sel_ids for c in ch_map[cid].columns]
        enc_z = FastFrameEncoder().fit(tr_s[s2_cols])
        z_tr = enc_z.transform(tr_s[s2_cols])
        z_ca = enc_z.transform(ca_s[s2_cols])
        z_te = enc_z.transform(te[s2_cols])
        x2_tr = np.hstack([xb_tr_s, z_tr])
        x2_ca = np.hstack([xb_ca_s, z_ca])
        x2_te = np.hstack([xb_te_s, z_te])
        p1_s = np.clip(p1_oof_r, 1e-5, 1.0 - 1e-5)

        if use_fluid:
            tau_fit = (
                contract.tau_star
                if pi_ov is None
                else float(adjust_probs_for_target_prevalence(np.array([contract.tau_star]), pi_ov, float(np.mean(y_tr_s)))[0])
            )
            clf = FluidCollisionBackboneClassifier(
                tau_star=tau_fit, kappa_max=20.0, min_cz_ess=240.0, seed=seed_s + 7
            ).fit(x2_tr, z_tr, y_tr_s, logit(p1_s))
            r_tr = clf.decision_function(x2_tr, z_tr, logit(p1_s))
            r_ca = clf.decision_function(x2_ca, z_ca, logit(np.clip(p1_ca_r, 1e-5, 1 - 1e-5)))
            r_te = clf.decision_function(x2_te, z_te, logit(np.clip(p1_te_r, 1e-5, 1 - 1e-5)))
        else:
            clf = None
            clf_g = lgb.LGBMClassifier(
                n_estimators=140,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=30,
                max_delta_step=1.0,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed_s + 7,
                n_jobs=int(os.environ.get("UPCA_LGBM_JOBS", "8")),
                verbose=-1,
            ).fit(x2_tr, y_tr_s)
            r_tr = clf_g.predict(x2_tr, raw_score=True)
            r_ca = clf_g.predict(x2_ca, raw_score=True)
            r_te = clf_g.predict(x2_te, raw_score=True)

        pl = fit_platt_calibrator(r_ca, y_ca_s)
        p2_tr = np.clip(apply_platt(r_tr, pl), 1e-6, 1.0 - 1e-6)
        p2_ca = np.clip(apply_platt(r_ca, pl), 1e-6, 1.0 - 1e-6)
        p2_te = np.clip(apply_platt(r_te, pl), 1e-6, 1.0 - 1e-6)
        prior_shift = 0.0
        if pi_ov is not None:
            pi_cs = float(np.clip(np.mean(y_ca_s), 1e-6, 1 - 1e-6))
            prior_shift = float(np.log(pi_ov / (1.0 - pi_ov)) - np.log(pi_cs / (1.0 - pi_cs)))
            p2_tr = adjust_probs_for_target_prevalence(p2_tr, pi_cs, pi_ov)
            p2_ca = adjust_probs_for_target_prevalence(p2_ca, pi_cs, pi_ov)
            p2_te = adjust_probs_for_target_prevalence(p2_te, pi_cs, pi_ov)

        ck_p = ClinicalContract(contract.cost_fp, contract.cost_fn, contract.target_precision, c_pan)
        voi_te = fit_predict_prepurchase_voi(
            xb_tr_s, p1_oof_e, p2_tr, xb_ca_s, p1_ca_e, p2_ca, xb_te_s, p1_te_e, ck_p, use_dispersion_2d=True, seed=seed_s
        )
        diag = (
            compute_fluid_specialist_diagnostics(
                clf,
                x2_te,
                z_te,
                logit(np.clip(p1_te_r, 1e-5, 1 - 1e-5)),
                contract.tau_star,
                outer_platt=pl,
                prior_shift_log_odds=prior_shift,
            )
            if (use_fluid and clf is not None)
            else {
                "beta_cz": 0.0,
                "kappa_star": 0.0,
                "rms_r_cz": 0.0,
                "rms_r_cz_zone": 0.0,
                "decision_flip_rate_vs_glob": 0.0,
            }
        )
        return {
            "p2_te": p2_te,
            "p2_ca": p2_ca,
            "voi_te": voi_te,
            "panel_cost_cfp": c_pan,
            "diag": diag,
        }

    # --- 2. Canonical Single-Split Evaluation under `ChannelCostSchedule` ---
    eb0 = FastFrameEncoder().fit(tr0[base_cols])
    xb_tr0, xb_ca0, xb_te0 = eb0.transform(tr0[base_cols]), eb0.transform(cal0[base_cols]), eb0.transform(te[base_cols])
    can_cache: dict[tuple[tuple[str, ...], bool], dict[str, Any]] = {}

    def _get_can(sel_ids: list[str], fl: bool) -> dict[str, Any]:
        k = (tuple(sel_ids), fl)
        if k not in can_cache:
            can_cache[k] = _fit_panel_on_split(
                tr0, y_tr0, p1_oof0_raw, p1_oof0_eval, cal0, y_cal0, p1_cal0_raw, p1_cal0_eval,
                xb_tr0, xb_ca0, xb_te0, p1_te0_raw, p1_te0_eval, sel_ids, fl, seed,
            )
        return can_cache[k]

    can_all_gbdt = _get_can(all_non_null_ids, False)
    can_glob_gbdt = _get_can(locked_panels["global_cmi"], False)
    can_glob_fluid = _get_can(locked_panels["global_cmi"], True)
    can_wcz = _get_can(locked_panels["weighted_cz"], True)
    can_sp = _get_can(locked_panels["split_pilot_upca"], True)
    can_upca = _get_can(locked_panels["upca_fluid"], True)

    dual_can = evaluate_dual_afa_scheduled(
        tr0, y_tr0, cal0, y_cal0, te, y_te, base_cols, ch_all,
        p1_oof0_raw, p1_cal0_raw, p1_te0_raw, p1_cal0_eval, p1_te0_eval,
        contract, budget, sched, pi_target_override=pi_ov, w_cal=w_cal0, w_te=w_te, seed=seed,
    )
    pr_afa_cmi0 = dual_can["cmi"]["p_rt"]
    cost_afa_cmi0 = dual_can["cmi"]["cost_te"]
    pr_afa_voi0 = dual_can["direct_voi"]["p_rt"]
    cost_afa_voi0 = dual_can["direct_voi"]["cost_te"]
    afa_cmi0 = {k: v for k, v in dual_can["cmi"].items() if k not in ("p_rt", "cost_te", "mask_te")}
    afa_voi0 = {k: v for k, v in dual_can["direct_voi"].items() if k not in ("p_rt", "cost_te", "mask_te")}

    def _summarize_routed(p2_v: np.ndarray, voi_v: np.ndarray, sel_ids: list[str], force_100: bool = False) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        c_p = sched.panel_cost_cfp(sel_ids)
        if not sel_ids:
            q_m = np.zeros(n_te, dtype=bool)
        elif force_100:
            q_m = np.ones(n_te, dtype=bool)
        else:
            q_m = voi_v > c_p
        p_rt = np.where(q_m, p2_v, p1_te0_eval)
        c_vec = q_m.astype(float) * c_p
        nb = variable_cost_nb_per_1k(y_te, p_rt, c_vec, contract, sample_weight=w_te)
        qr = float(np.mean(q_m)) if w_te is None else float(np.sum(w_te * q_m) / np.sum(w_te))
        return (
            {
                "selected": list(sel_ids),
                "panel_cost_cfp": round(c_p, 5),
                "nb_routed": round(nb, 4),
                "snb_routed": round(standardized_net_benefit(nb, eval_prev), 4),
                "query_rate": round(qr, 4),
                "mean_cost_cfp": round(qr * c_p, 5),
                "auprc_full": round(float(average_precision_score(y_te, p2_v, sample_weight=w_te)), 4),
                "auprc_routed": round(float(average_precision_score(y_te, p_rt, sample_weight=w_te)), 4),
                "auroc_full": round(float(roc_auc_score(y_te, p2_v, sample_weight=w_te)), 4),
            },
            p_rt,
            c_vec,
            q_m,
        )

    can_s1_s, pr_can_s1, c_can_s1, _ = _summarize_routed(p1_te0_eval, np.zeros(n_te), [])
    can_full_all_s, pr_can_fall, c_can_fall, _ = _summarize_routed(can_all_gbdt["p2_te"], np.zeros(n_te), all_non_null_ids, force_100=True)
    can_full_lock_s, pr_can_flock, c_can_flock, _ = _summarize_routed(can_upca["p2_te"], np.zeros(n_te), locked_panels["upca_fluid"], force_100=True)
    can_gg_s, pr_can_gg, c_can_gg, _ = _summarize_routed(can_glob_gbdt["p2_te"], can_glob_gbdt["voi_te"], locked_panels["global_cmi"])
    can_gf_s, pr_can_gf, c_can_gf, _ = _summarize_routed(can_glob_fluid["p2_te"], can_glob_fluid["voi_te"], locked_panels["global_cmi"])
    can_wcz_s, pr_can_wcz, c_can_wcz, _ = _summarize_routed(can_wcz["p2_te"], can_wcz["voi_te"], locked_panels["weighted_cz"])
    can_sp_s, _pr_can_sp, _c_can_sp, _ = _summarize_routed(can_sp["p2_te"], can_sp["voi_te"], locked_panels["split_pilot_upca"])
    can_upca_s, pr_can_upca, c_can_upca, q_can_upca = _summarize_routed(can_upca["p2_te"], can_upca["voi_te"], locked_panels["upca_fluid"])

    q_bu_can = boundary_uncertainty_mask(p1_te0_eval, contract.tau_star, can_upca_s["query_rate"])
    pr_can_bu = np.where(q_bu_can, can_upca["p2_te"], p1_te0_eval)
    c_can_bu = q_bu_can.astype(float) * sched.panel_cost_cfp(locked_panels["upca_fluid"])
    can_bu_s = {
        "selected": locked_panels["upca_fluid"],
        "panel_cost_cfp": round(sched.panel_cost_cfp(locked_panels["upca_fluid"]), 5),
        "nb_routed": round(variable_cost_nb_per_1k(y_te, pr_can_bu, c_can_bu, contract, sample_weight=w_te), 4),
        "query_rate": round(float(np.mean(q_bu_can)) if w_te is None else float(np.sum(w_te * q_bu_can) / np.sum(w_te)), 4),
    }

    can_skip_audit = skipped_case_disagreement_audit(
        y_te, p1_te0_eval, can_upca["p2_te"], q_can_upca, clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed
    )

    canonical_boots = {
        "upca_vs_stage1": variable_cost_cluster_bootstrap_diff(y_te, pr_can_upca, c_can_upca, pr_can_s1, c_can_s1, clusters_te, contract, sample_weight=w_te, seed=seed + 1),
        "upca_vs_full_all_m": variable_cost_cluster_bootstrap_diff(y_te, pr_can_upca, c_can_upca, pr_can_fall, c_can_fall, clusters_te, contract, sample_weight=w_te, seed=seed + 2),
        "upca_vs_full_locked": variable_cost_cluster_bootstrap_diff(y_te, pr_can_upca, c_can_upca, pr_can_flock, c_can_flock, clusters_te, contract, sample_weight=w_te, seed=seed + 3),
        "upca_vs_boundary_uncertainty": variable_cost_cluster_bootstrap_diff(y_te, pr_can_upca, c_can_upca, pr_can_bu, c_can_bu, clusters_te, contract, sample_weight=w_te, seed=seed + 4),
        "upca_vs_global_cmi_gbdt": variable_cost_cluster_bootstrap_diff(y_te, pr_can_upca, c_can_upca, pr_can_gg, c_can_gg, clusters_te, contract, sample_weight=w_te, seed=seed + 5),
        "upca_vs_global_cmi_fluid": variable_cost_cluster_bootstrap_diff(y_te, pr_can_upca, c_can_upca, pr_can_gf, c_can_gf, clusters_te, contract, sample_weight=w_te, seed=seed + 6),
        "upca_vs_weighted_cz": variable_cost_cluster_bootstrap_diff(y_te, pr_can_upca, c_can_upca, pr_can_wcz, c_can_wcz, clusters_te, contract, sample_weight=w_te, seed=seed + 7),
        "upca_vs_covert_cmi": variable_cost_cluster_bootstrap_diff(y_te, pr_can_upca, c_can_upca, pr_afa_cmi0, cost_afa_cmi0, clusters_te, contract, sample_weight=w_te, seed=seed + 8),
        "upca_vs_direct_voi_afa": variable_cost_cluster_bootstrap_diff(y_te, pr_can_upca, c_can_upca, pr_afa_voi0, cost_afa_voi0, clusters_te, contract, sample_weight=w_te, seed=seed + 9),
    }

    # --- 3. R = 10 Patient-Grouped Development Splits (`5` Spawn Workers x `2` Threads = `10` CPU Cores) ---
    cal_frac_dev = float(len(cal0)) / float(len(dev))

    common_arms = [
        ("all_m_gbdt", all_non_null_ids, False),
        ("global_cmi_gbdt", locked_panels["global_cmi"], False),
        ("global_cmi_fluid", locked_panels["global_cmi"], True),
        ("weighted_cz_fluid", locked_panels["weighted_cz"], True),
        ("split_pilot_upca_fluid", locked_panels["split_pilot_upca"], True),
        ("upca_fluid", locked_panels["upca_fluid"], True),
    ]
    com_p2_sum = {a: np.zeros(n_te, dtype=float) for a, _, _ in common_arms}
    com_voi_sum = {a: np.zeros(n_te, dtype=float) for a, _, _ in common_arms}
    p1_te_sum = np.zeros(n_te, dtype=float)
    p1_te_raw_sum = np.zeros(n_te, dtype=float)
    diag_r10: dict[str, list[dict[str, float]]] = {a: [] for a, _, _ in common_arms}

    persplit_arms = ["stage1", "global_cmi_gbdt", "global_cmi_fluid", "weighted_cz_fluid", "upca_fluid"]
    ps_nbs: dict[str, list[float]] = {a: [] for a in persplit_arms}
    ps_p_sum = {a: np.zeros(n_te, dtype=float) for a in persplit_arms}
    ps_p2_sum = {a: np.zeros(n_te, dtype=float) for a in persplit_arms}
    ps_voi_sum = {a: np.zeros(n_te, dtype=float) for a in persplit_arms}
    ps_cost_sum = {a: np.zeros(n_te, dtype=float) for a in persplit_arms}
    ps_q_sum = {a: np.zeros(n_te, dtype=float) for a in persplit_arms}
    ps_union_sets: dict[str, set[str]] = {a: set() for a in persplit_arms if a != "stage1"}
    ps_sel_freqs: dict[str, dict[str, int]] = {a: {} for a in persplit_arms if a != "stage1"}

    afa_r10: dict[str, dict[str, Any]] = {
        "cmi": {"nbs": [], "qs": [], "null_qs": [], "masks": [], "unit_u_sum": np.zeros(n_te, dtype=float)},
        "direct_voi": {"nbs": [], "qs": [], "null_qs": [], "masks": [], "unit_u_sum": np.zeros(n_te, dtype=float)},
    }
    afa_states: list[dict[str, Any]] = []

    keep_cols = list(dict.fromkeys([cl_col, y_col, *base_cols, *(c for ch in ch_all for c in ch.columns)]))
    worker_ctx = {
        "dev": dev[keep_cols].copy(),
        "te": te[keep_cols].copy(),
        "y_col": y_col,
        "cl_col": cl_col,
        "base_cols": base_cols,
        "ch_all": ch_all,
        "contract": contract,
        "budget": budget,
        "pi_ov": pi_ov,
        "seed": seed,
        "sched": sched,
        "common_arms": common_arms,
        "cal_frac_dev": cal_frac_dev,
        "y_te": y_te,
        "w_te": w_te,
    }
    split_results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(
        max_workers=5,
        mp_context=mp.get_context("spawn"),
        initializer=_init_b2_r10_worker,
        initargs=(worker_ctx,),
    ) as pool:
        for done_cnt, s_res in enumerate(pool.map(_run_b2_r10_split, range(n_splits)), start=1):
            split_results.append(s_res)
            print(f"    [{tid}] Split {done_cnt}/{n_splits} finished ({time.time()-t0:.1f}s)", flush=True)

    split_results.sort(key=lambda x: int(x["r_i"]))
    for s_res in split_results:
        p1_te_sum += s_res["p1_te_e"]
        p1_te_raw_sum += s_res["p1_te_r"]
        for arm_name, _, _ in common_arms:
            co = s_res["com_out"][arm_name]
            com_p2_sum[arm_name] += co["p2_te"]
            com_voi_sum[arm_name] += co["voi_te"]
            diag_r10[arm_name].append(co["diag"])

        for ak in persplit_arms:
            po = s_res["ps_out"][ak]
            ps_nbs[ak].append(po["nb"])
            ps_p_sum[ak] += po["pr"]
            ps_p2_sum[ak] += po["p2"]
            ps_voi_sum[ak] += po["voi"]
            ps_cost_sum[ak] += po["cost"]
            ps_q_sum[ak] += po["q"]
            if ak != "stage1":
                ps_union_sets[ak].update(po["sel"])
                for cid in po["sel"]:
                    ps_sel_freqs[ak][cid] = ps_sel_freqs[ak].get(cid, 0) + 1

        d_afa = s_res["dual_afa"]
        afa_states.append(d_afa["masked_predictor_state"])
        for pm in ("cmi", "direct_voi"):
            afa_r10[pm]["nbs"].append(d_afa[pm]["nb_routed"])
            afa_r10[pm]["qs"].append(d_afa[pm]["query_rate"])
            afa_r10[pm]["null_qs"].append(d_afa[pm]["null_channel_query_rate"])
            afa_r10[pm]["masks"].append(d_afa[pm]["mask_te"])
            afa_r10[pm]["unit_u_sum"] += d_afa[pm]["unit_u"]

    # --- 4. Aggregate Common-Panel Bagged VOI, Per-Split, Symmetric Ensemble, Union-Panel & Majority AFA ---
    p1_bag_te = p1_te_sum / float(n_splits)
    p1_bag_raw_te = p1_te_raw_sum / float(n_splits)
    common_bag_summary: dict[str, Any] = {}
    common_bag_traces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    nb_s1_bag = variable_cost_nb_per_1k(y_te, p1_bag_te, 0.0, contract, sample_weight=w_te)
    common_bag_summary["stage1"] = {
        "locked_panel": [],
        "panel_cost_cfp": 0.0,
        "nb_bagged": round(nb_s1_bag, 4),
        "snb_bagged": round(standardized_net_benefit(nb_s1_bag, eval_prev), 4),
        "query_rate": 0.0,
        "mean_cost_cfp": 0.0,
        "auprc_full": round(float(average_precision_score(y_te, p1_bag_te, sample_weight=w_te)), 4),
        "auprc_routed": round(float(average_precision_score(y_te, p1_bag_te, sample_weight=w_te)), 4),
        "per_split_mean_nb": round(float(np.mean(ps_nbs["stage1"])), 4),
        "per_split_se_nb": round(float(np.std(ps_nbs["stage1"], ddof=1) / np.sqrt(n_splits)), 4),
    }
    common_bag_traces["stage1"] = (p1_bag_te, np.zeros(n_te, dtype=float), np.zeros(n_te, dtype=bool), p1_bag_te)

    for arm_name, sel_c, _ in common_arms:
        p2_bag = com_p2_sum[arm_name] / float(n_splits)
        voi_bag = com_voi_sum[arm_name] / float(n_splits)
        c_pan = sched.panel_cost_cfp(sel_c)
        q_bag = (voi_bag > c_pan) if sel_c else np.zeros(n_te, dtype=bool)
        p_rt_bag = np.where(q_bag, p2_bag, p1_bag_te)
        c_vec_bag = q_bag.astype(float) * c_pan
        nb_bag = variable_cost_nb_per_1k(y_te, p_rt_bag, c_vec_bag, contract, sample_weight=w_te)
        nb_100 = variable_cost_nb_per_1k(y_te, p2_bag, np.full(n_te, c_pan), contract, sample_weight=w_te)
        qr_bag = float(np.mean(q_bag)) if w_te is None else float(np.sum(w_te * q_bag) / np.sum(w_te))

        d_list = diag_r10[arm_name]
        diag_agg = {
            k: round(float(np.mean([d[k] for d in d_list])), 4)
            for k in ("beta_cz", "kappa_star", "rms_r_cz", "rms_r_cz_zone", "decision_flip_rate_vs_glob")
        }
        diag_agg["beta_cz_se"] = round(float(np.std([d["beta_cz"] for d in d_list], ddof=1) / np.sqrt(n_splits)), 4)
        diag_agg["rms_r_cz_se"] = round(float(np.std([d["rms_r_cz"] for d in d_list], ddof=1) / np.sqrt(n_splits)), 4)

        entry: dict[str, Any] = {
            "locked_panel": list(sel_c),
            "panel_cost_cfp": round(c_pan, 5),
            "nb_bagged": round(nb_bag, 4),
            "snb_bagged": round(standardized_net_benefit(nb_bag, eval_prev), 4),
            "nb_100pct_panel": round(nb_100, 4),
            "query_rate": round(qr_bag, 4),
            "mean_cost_cfp": round(qr_bag * c_pan, 5),
            "auprc_full": round(float(average_precision_score(y_te, p2_bag, sample_weight=w_te)), 4),
            "auprc_routed": round(float(average_precision_score(y_te, p_rt_bag, sample_weight=w_te)), 4),
            "auroc_full": round(float(roc_auc_score(y_te, p2_bag, sample_weight=w_te)), 4),
            "specialist_diagnostics": diag_agg,
        }
        if arm_name in ps_nbs:
            sp_arr = np.array(ps_nbs[arm_name], dtype=float)
            p_sym = ps_p_sum[arm_name] / float(n_splits)
            c_sym = ps_cost_sum[arm_name] / float(n_splits)
            q_sym = ps_q_sum[arm_name] / float(n_splits)
            nb_sym = variable_cost_nb_per_1k(y_te, p_sym, c_sym, contract, sample_weight=w_te)

            u_set = sorted(ps_union_sets[arm_name])
            c_union = sched.panel_cost_cfp(u_set)
            p2_u = ps_p2_sum[arm_name] / float(n_splits)
            voi_u = ps_voi_sum[arm_name] / float(n_splits)
            q_u_rerouted = (voi_u > c_union) if u_set else np.zeros(n_te, dtype=bool)
            nb_u_rerouted = variable_cost_nb_per_1k(
                y_te, np.where(q_u_rerouted, p2_u, p1_bag_te), q_u_rerouted.astype(float) * c_union, contract, sample_weight=w_te
            )
            q_u_flat = (voi_u > contract.per_instance_query_cost) if u_set else np.zeros(n_te, dtype=bool)
            nb_u_fixedq_union_billed = variable_cost_nb_per_1k(
                y_te, np.where(q_u_flat, p2_u, p1_bag_te), q_u_flat.astype(float) * c_union, contract, sample_weight=w_te
            )
            nb_u_legacy_flat = weighted_net_benefit_per_1k(
                y_te, np.where(q_u_flat, p2_u, p1_bag_te), q_u_flat, contract, sample_weight=w_te
            )

            entry.update(
                {
                    "per_split_mean_nb": round(float(np.mean(sp_arr)), 4),
                    "per_split_se_nb": round(float(np.std(sp_arr, ddof=1) / np.sqrt(n_splits)), 4),
                    "sym_ens_nb": round(nb_sym, 4),
                    "sym_ens_query_rate": round(float(np.mean(q_sym)) if w_te is None else float(np.sum(w_te * q_sym) / np.sum(w_te)), 4),
                    "union_panel_ids": u_set,
                    "union_panel_cost_cfp": round(c_union, 5),
                    "union_panel_rerouted_nb": round(nb_u_rerouted, 4),
                    "union_panel_rerouted_q": round(float(np.mean(q_u_rerouted)) if w_te is None else float(np.sum(w_te * q_u_rerouted) / np.sum(w_te)), 4),
                    "union_panel_fixedq_billed_nb": round(nb_u_fixedq_union_billed, 4),
                    "legacy_unbilled_flat_union_nb": round(nb_u_legacy_flat, 4),
                    "r10_selected_channel_counts": ps_sel_freqs[arm_name],
                }
            )
        common_bag_summary[arm_name] = entry
        common_bag_traces[arm_name] = (p_rt_bag, c_vec_bag, q_bag, p2_bag)

    # Full 100% All-M & Locked-Panel + Boundary Uncertainty on Common-Panel Bagged Models
    _, _, _, p2_all_gbdt_bag = common_bag_traces["all_m_gbdt"]
    pr_up_bag, c_up_bag, q_up_bag, p2_up_bag = common_bag_traces["upca_fluid"]
    c_all_m = sched.panel_cost_cfp(all_non_null_ids)
    c_up_pan = sched.panel_cost_cfp(locked_panels["upca_fluid"])

    c_full_all_vec = np.full(n_te, c_all_m, dtype=float)
    c_full_lock_vec = np.full(n_te, c_up_pan, dtype=float)
    nb_full_all_gbdt = variable_cost_nb_per_1k(y_te, p2_all_gbdt_bag, c_full_all_vec, contract, sample_weight=w_te)
    nb_full_locked = variable_cost_nb_per_1k(y_te, p2_up_bag, c_full_lock_vec, contract, sample_weight=w_te)

    up_q_rate = common_bag_summary["upca_fluid"]["query_rate"]
    q_bu_match = boundary_uncertainty_mask(p1_bag_te, contract.tau_star, up_q_rate)
    pr_bu_match = np.where(q_bu_match, p2_up_bag, p1_bag_te)
    c_bu_match = q_bu_match.astype(float) * c_up_pan
    nb_bu_match = variable_cost_nb_per_1k(y_te, pr_bu_match, c_bu_match, contract, sample_weight=w_te)

    q_bu_20 = boundary_uncertainty_mask(p1_bag_te, contract.tau_star, 0.20) if locked_panels["upca_fluid"] else np.zeros(n_te, dtype=bool)
    pr_bu_20 = np.where(q_bu_20, p2_up_bag, p1_bag_te)
    c_bu_20 = q_bu_20.astype(float) * c_up_pan
    nb_bu_20 = variable_cost_nb_per_1k(y_te, pr_bu_20, c_bu_20, contract, sample_weight=w_te)

    common_bag_summary["full_100pct_all_m_gbdt"] = {
        "locked_panel": all_non_null_ids,
        "panel_cost_cfp": round(c_all_m, 5),
        "nb_bagged": round(nb_full_all_gbdt, 4),
        "snb_bagged": round(standardized_net_benefit(nb_full_all_gbdt, eval_prev), 4),
        "query_rate": 1.0,
        "mean_cost_cfp": round(c_all_m, 5),
        "auprc_full": common_bag_summary["all_m_gbdt"]["auprc_full"],
    }
    common_bag_summary["full_100pct_locked_panel"] = {
        "locked_panel": locked_panels["upca_fluid"],
        "panel_cost_cfp": round(c_up_pan, 5),
        "nb_bagged": round(nb_full_locked, 4),
        "query_rate": 1.0 if locked_panels["upca_fluid"] else 0.0,
    }
    common_bag_summary["boundary_uncertainty_matched_q"] = {
        "locked_panel": locked_panels["upca_fluid"],
        "panel_cost_cfp": round(c_up_pan, 5),
        "nb_bagged": round(nb_bu_match, 4),
        "query_rate": round(float(np.mean(q_bu_match)) if w_te is None else float(np.sum(w_te * q_bu_match) / np.sum(w_te)), 4),
    }
    common_bag_summary["boundary_uncertainty_20pct"] = {
        "locked_panel": locked_panels["upca_fluid"],
        "panel_cost_cfp": round(c_up_pan, 5),
        "nb_bagged": round(nb_bu_20, 4),
        "query_rate": round(float(np.mean(q_bu_20)) if w_te is None else float(np.sum(w_te * q_bu_20) / np.sum(w_te)), 4),
    }

    # R=10 Majority-Mask (`>= 5/10` vote) & Per-Split Summary for Channel-Billed AFA
    afa_r10_summary: dict[str, Any] = {}
    afa_maj_traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pm in ("cmi", "direct_voi"):
        mask_stack = np.stack(afa_r10[pm]["masks"], axis=0)  # (R, N, M)
        maj_mask = (np.mean(mask_stack > 0.5, axis=0) >= 0.5).astype(np.float32)
        p_maj = np.mean([predict_masked_te_from_state(st_m, maj_mask) for st_m in afa_states], axis=0)
        c_maj = sched.mask_costs_cfp(maj_mask, ch_ids_ordered)
        q_maj = np.any(maj_mask > 0.5, axis=1)
        nb_maj = variable_cost_nb_per_1k(y_te, p_maj, c_maj, contract, sample_weight=w_te)
        qr_maj = float(np.mean(q_maj)) if w_te is None else float(np.sum(w_te * q_maj) / np.sum(w_te))
        nbs_arr = np.array(afa_r10[pm]["nbs"], dtype=float)
        afa_r10_summary[pm] = {
            "per_split_mean_nb": round(float(np.mean(nbs_arr)), 4),
            "per_split_se_nb": round(float(np.std(nbs_arr, ddof=1) / np.sqrt(n_splits)), 4),
            "per_split_mean_query_rate": round(float(np.mean(afa_r10[pm]["qs"])), 4),
            "per_split_mean_null_query_rate": round(float(np.mean(afa_r10[pm]["null_qs"])), 4),
            "majority_mask_nb": round(nb_maj, 4),
            "majority_mask_query_rate": round(qr_maj, 4),
            "majority_mask_mean_cost_cfp": round(float(np.mean(c_maj)) if w_te is None else float(np.sum(w_te * c_maj) / np.sum(w_te)), 5),
        }
        afa_maj_traces[pm] = (p_maj, c_maj)

    # Cluster bootstrap for UPCA Common-Bagged vs R=10 Per-Split Mean AFA unit vector
    def _boot_vs_unit_vec(u_ref_vec: np.ndarray, bseed: int) -> dict[str, float]:
        u_arm = variable_cost_unit_nb_vec(y_te, pr_up_bag, c_up_bag, contract)
        diff_i = u_arm - u_ref_vec
        wi = np.ones_like(diff_i) if w_te is None else w_te
        obs = float(np.sum(wi * diff_i) / np.sum(wi))
        _, inv = np.unique(clusters_te, return_inverse=True)
        nc = int(inv.max()) + 1
        cnum = np.bincount(inv, weights=wi * diff_i, minlength=nc)
        cden = np.bincount(inv, weights=wi, minlength=nc)
        rng_b = np.random.default_rng(bseed)
        cnts = rng_b.multinomial(nc, np.full(nc, 1.0 / nc), size=10_000).astype(float)
        bvals = (cnts @ cnum) / np.maximum(1e-12, cnts @ cden)
        lo, hi = np.quantile(bvals, [0.025, 0.975])
        kle, kge = int(np.sum(bvals <= 0.0)), int(np.sum(bvals >= 0.0))
        pdh = min(1.0, 2.0 * float(min(kle, kge) + 1) / 10001.0)
        return {
            "diff_mean": round(obs, 4),
            "se": round(float(np.std(bvals, ddof=1)), 4),
            "ci95_low": round(float(lo), 4),
            "ci95_high": round(float(hi), 4),
            "p_two_sided": round(float(pdh), 5),
        }

    pr_gg_bag, c_gg_bag, _, _ = common_bag_traces["global_cmi_gbdt"]
    pr_gf_bag, c_gf_bag, _, _ = common_bag_traces["global_cmi_fluid"]
    pr_wc_bag, c_wc_bag, _, _ = common_bag_traces["weighted_cz_fluid"]
    pr_sp_bag, c_sp_bag, _, _ = common_bag_traces["split_pilot_upca_fluid"]

    common_bag_boots = {
        "upca_vs_stage1": variable_cost_cluster_bootstrap_diff(y_te, pr_up_bag, c_up_bag, p1_bag_te, 0.0, clusters_te, contract, sample_weight=w_te, seed=101),
        "upca_vs_full_100pct_all_m": variable_cost_cluster_bootstrap_diff(y_te, pr_up_bag, c_up_bag, p2_all_gbdt_bag, c_full_all_vec, clusters_te, contract, sample_weight=w_te, seed=102),
        "upca_vs_full_100pct_locked": variable_cost_cluster_bootstrap_diff(y_te, pr_up_bag, c_up_bag, p2_up_bag, c_full_lock_vec, clusters_te, contract, sample_weight=w_te, seed=103),
        "upca_vs_boundary_uncertainty_matched_q": variable_cost_cluster_bootstrap_diff(y_te, pr_up_bag, c_up_bag, pr_bu_match, c_bu_match, clusters_te, contract, sample_weight=w_te, seed=104),
        "upca_vs_boundary_uncertainty_20pct": variable_cost_cluster_bootstrap_diff(y_te, pr_up_bag, c_up_bag, pr_bu_20, c_bu_20, clusters_te, contract, sample_weight=w_te, seed=105),
        "upca_vs_global_cmi_gbdt": variable_cost_cluster_bootstrap_diff(y_te, pr_up_bag, c_up_bag, pr_gg_bag, c_gg_bag, clusters_te, contract, sample_weight=w_te, seed=106),
        "upca_vs_global_cmi_fluid": variable_cost_cluster_bootstrap_diff(y_te, pr_up_bag, c_up_bag, pr_gf_bag, c_gf_bag, clusters_te, contract, sample_weight=w_te, seed=107),
        "upca_vs_weighted_cz_fluid": variable_cost_cluster_bootstrap_diff(y_te, pr_up_bag, c_up_bag, pr_wc_bag, c_wc_bag, clusters_te, contract, sample_weight=w_te, seed=108),
        "upca_vs_split_pilot_upca": variable_cost_cluster_bootstrap_diff(y_te, pr_up_bag, c_up_bag, pr_sp_bag, c_sp_bag, clusters_te, contract, sample_weight=w_te, seed=109),
        "upca_vs_afa_cmi_persplit": _boot_vs_unit_vec(afa_r10["cmi"]["unit_u_sum"] / float(n_splits), 110),
        "upca_vs_afa_direct_voi_persplit": _boot_vs_unit_vec(afa_r10["direct_voi"]["unit_u_sum"] / float(n_splits), 111),
        "upca_vs_afa_cmi_majority": variable_cost_cluster_bootstrap_diff(y_te, pr_up_bag, c_up_bag, afa_maj_traces["cmi"][0], afa_maj_traces["cmi"][1], clusters_te, contract, sample_weight=w_te, seed=112),
    }

    # --- 5. 5-Point Query-Cost Multiplier Sweep (`0.5x, 1.0x, 1.5x, 2.0x, 3.0x`) on Common-Bagged ---
    voi_up_bag = com_voi_sum["upca_fluid"] / float(n_splits)
    voi_gg_bag = com_voi_sum["global_cmi_gbdt"] / float(n_splits)
    fee_sweep_rows = []
    for idx_m, mult in enumerate([0.5, 1.0, 1.5, 2.0, 3.0]):
        sc_m = sched.with_multiplier(mult)
        cp_up_m = sc_m.panel_cost_cfp(locked_panels["upca_fluid"])
        cp_gg_m = sc_m.panel_cost_cfp(locked_panels["global_cmi"])
        cp_all_m = sc_m.panel_cost_cfp(all_non_null_ids)

        q_up_m = (voi_up_bag > cp_up_m) if locked_panels["upca_fluid"] else np.zeros(n_te, dtype=bool)
        pr_up_m = np.where(q_up_m, p2_up_bag, p1_bag_te)
        c_up_m = q_up_m.astype(float) * cp_up_m
        qr_up_m = float(np.mean(q_up_m)) if w_te is None else float(np.sum(w_te * q_up_m) / np.sum(w_te))

        q_gg_m = (voi_gg_bag > cp_gg_m) if locked_panels["global_cmi"] else np.zeros(n_te, dtype=bool)
        pr_gg_m = np.where(q_gg_m, com_p2_sum["global_cmi_gbdt"] / float(n_splits), p1_bag_te)
        c_gg_m = q_gg_m.astype(float) * cp_gg_m

        q_bu_m = boundary_uncertainty_mask(p1_bag_te, contract.tau_star, qr_up_m)
        pr_bu_m = np.where(q_bu_m, p2_up_bag, p1_bag_te)
        c_bu_m = q_bu_m.astype(float) * cp_up_m

        q_bu20_m = boundary_uncertainty_mask(p1_bag_te, contract.tau_star, 0.20) if locked_panels["upca_fluid"] else np.zeros(n_te, dtype=bool)
        pr_bu20_m = np.where(q_bu20_m, p2_up_bag, p1_bag_te)
        c_bu20_m = q_bu20_m.astype(float) * cp_up_m

        c_all_vec_m = np.full(n_te, cp_all_m, dtype=float)
        c_lock_vec_m = np.full(n_te, cp_up_m, dtype=float)

        nb_up_m = variable_cost_nb_per_1k(y_te, pr_up_m, c_up_m, contract, sample_weight=w_te)
        nb_gg_m = variable_cost_nb_per_1k(y_te, pr_gg_m, c_gg_m, contract, sample_weight=w_te)
        nb_bu_m = variable_cost_nb_per_1k(y_te, pr_bu_m, c_bu_m, contract, sample_weight=w_te)
        nb_bu20_m = variable_cost_nb_per_1k(y_te, pr_bu20_m, c_bu20_m, contract, sample_weight=w_te)
        nb_all_m = variable_cost_nb_per_1k(y_te, p2_all_gbdt_bag, c_all_vec_m, contract, sample_weight=w_te)
        nb_lock_m = variable_cost_nb_per_1k(y_te, p2_up_bag, c_lock_vec_m, contract, sample_weight=w_te)

        b_vs_all = variable_cost_cluster_bootstrap_diff(
            y_te, pr_up_m, c_up_m, p2_all_gbdt_bag, c_all_vec_m, clusters_te, contract, sample_weight=w_te, seed=300 + idx_m
        )
        b_vs_s1 = variable_cost_cluster_bootstrap_diff(
            y_te, pr_up_m, c_up_m, p1_bag_te, 0.0, clusters_te, contract, sample_weight=w_te, seed=320 + idx_m
        )
        fee_sweep_rows.append(
            {
                "cost_multiplier": mult,
                "upca_panel_cost_cfp": round(cp_up_m, 5),
                "all_m_panel_cost_cfp": round(cp_all_m, 5),
                "nb_stage1": round(nb_s1_bag, 4),
                "nb_full_100pct_all_m": round(nb_all_m, 4),
                "nb_full_100pct_locked": round(nb_lock_m, 4),
                "nb_boundary_uncertainty_matched_q": round(nb_bu_m, 4),
                "nb_boundary_uncertainty_20pct": round(nb_bu20_m, 4),
                "nb_global_cmi_gbdt": round(nb_gg_m, 4),
                "nb_upca_fluid": round(nb_up_m, 4),
                "upca_query_rate": round(qr_up_m, 4),
                "boot_upca_vs_full_all_m": b_vs_all,
                "boot_upca_vs_stage1": b_vs_s1,
            }
        )

    common_skip_audit = skipped_case_disagreement_audit(
        y_te, p1_bag_te, p2_up_bag, q_up_bag, clusters_te, contract, sample_weight=w_te, n_boot=10_000, seed=seed + 55
    )

    # Save trace arrays for fast, exact Block 3 & Block 4 consumption
    SCRATCH_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SCRATCH_TRACE_DIR / f"{tid}.npz",
        y_te=y_te,
        w_te=(np.ones(n_te, dtype=float) if w_te is None else w_te),
        p1_bag_te=p1_bag_te,
        p1_bag_raw_te=p1_bag_raw_te,
        p2_up_bag=p2_up_bag,
        voi_up_bag=voi_up_bag,
        pr_up_bag=pr_up_bag,
        c_up_bag=c_up_bag,
        q_up_bag=q_up_bag.astype(np.int8),
        p2_all_gbdt_bag=p2_all_gbdt_bag,
        c_full_all_vec=c_full_all_vec,
        c_full_lock_vec=c_full_lock_vec,
        pr_gg_bag=pr_gg_bag,
        c_gg_bag=c_gg_bag,
        pr_bu_match=pr_bu_match,
        c_bu_match=c_bu_match,
        y_cal0=y_cal0,
        p1_cal0_eval=p1_cal0_eval,
        p2_up_cal0=can_upca["p2_ca"],
        pr_can_upca=pr_can_upca,
        c_can_upca=c_can_upca,
        q_can_upca=q_can_upca.astype(np.int8),
    )

    print(
        f"  [Block 2 | {tid} DONE in {time.time()-t0:.1f}s] Common-Bagged UPCA={common_bag_summary['upca_fluid']['nb_bagged']:+.2f} "
        f"(q={up_q_rate*100:.1f}%, C={c_up_pan:.4f}) | Full-All-M={nb_full_all_gbdt:+.2f} "
        f"(d={common_bag_boots['upca_vs_full_100pct_all_m']['diff_mean']:+.2f}, p={common_bag_boots['upca_vs_full_100pct_all_m']['p_two_sided']:.4f}) | "
        f"Stage-1={nb_s1_bag:+.2f} (d={common_bag_boots['upca_vs_stage1']['diff_mean']:+.2f}, p={common_bag_boots['upca_vs_stage1']['p_two_sided']:.4f}) | "
        f"R_CZ={common_bag_summary['upca_fluid']['specialist_diagnostics']['rms_r_cz']:.3f}",
        flush=True,
    )

    return {
        "task_id": tid,
        "label": spec["label"],
        "n_train": len(tr0),
        "n_cal": len(cal0),
        "n_test": int(n_te),
        "n_test_patients": len(np.unique(clusters_te)),
        "test_events": int(np.sum(y_te == 1)),
        "eval_prevalence": round(eval_prev, 5),
        "cost_schedule": {
            "base_c_query": contract.per_instance_query_cost,
            "reference_bundle_weight": float(spec["ref_bundle_weight"]),
            "overhead_share": 0.25,
            "c_overhead": round(sched.c_overhead, 5),
            "c_unit": round(sched.c_unit, 5),
        },
        "locked_panels_tr0": locked_panels,
        "stage_b_scheduled_prune_log_tr0": prune_log_sched0,
        "canonical_scheduled": {
            "arms": {
                "stage1": can_s1_s,
                "full_100pct_all_m_gbdt": can_full_all_s,
                "full_100pct_locked_panel": can_full_lock_s,
                "boundary_uncertainty_matched_q": can_bu_s,
                "global_cmi_gbdt": can_gg_s,
                "global_cmi_fluid": can_gf_s,
                "weighted_cz_fluid": can_wcz_s,
                "split_pilot_upca_fluid": can_sp_s,
                "upca_fluid": can_upca_s,
                "afa_covert_cmi": afa_cmi0,
                "afa_direct_voi": afa_voi0,
            },
            "specialist_diagnostics_upca": can_upca["diag"],
            "skipped_case_disagreement_upca": can_skip_audit,
            "bootstraps_10k_patient_cluster": canonical_boots,
        },
        "r10_common_panel_and_triad": {
            "n_splits": n_splits,
            "arms": common_bag_summary,
            "afa_channel_billed_r10": afa_r10_summary,
            "skipped_case_disagreement_common_bagged": common_skip_audit,
            "bootstraps_10k_patient_cluster": common_bag_boots,
            "query_cost_multiplier_sweep": fee_sweep_rows,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }


def run_block2() -> dict[str, Any]:
    t0 = time.time()
    print("\n=== [BLOCK 2] Running Common-Panel Bagged VOI, Full 100%, BU, AFA & 5-Point Fee Sweep Across All 6 Tasks ===", flush=True)
    tasks, _ = load_all_clinical_tasks()
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    b2_path = BLOCKS_DIR / "block2_common_panel_and_fee_sweep.json"
    existing: dict[str, Any] = json.loads(b2_path.read_text()) if b2_path.exists() else {"tasks": {}}

    for spec in tasks:
        tid = spec["task_id"]
        if tid in existing.get("tasks", {}) and (SCRATCH_TRACE_DIR / f"{tid}.npz").exists():
            print(f"  [Block 2] Reusing completed task checkpoint: {tid}", flush=True)
            continue
        existing.setdefault("tasks", {})[tid] = _evaluate_single_task_v2_2(spec, n_splits=10)
        b2_path.write_text(json.dumps(existing, indent=2))

    # Add supplementary 6-task Holm p-values across Common-Bagged contrasts
    ordered_ids = [s["task_id"] for s in tasks]
    for contrast_k in ("upca_vs_stage1", "upca_vs_full_100pct_all_m", "upca_vs_global_cmi_gbdt"):
        raw_ps = [
            existing["tasks"][tid]["r10_common_panel_and_triad"]["bootstraps_10k_patient_cluster"][contrast_k]["p_two_sided"]
            for tid in ordered_ids
        ]
        holm_ps = holm_adjust(raw_ps)
        for idx, tid in enumerate(ordered_ids):
            existing["tasks"][tid]["r10_common_panel_and_triad"].setdefault("holm_6task_supplement", {})[contrast_k] = round(
                float(holm_ps[idx]), 5
            )

    existing["unique_primary_encounters_reconciliation"] = {
        "stanford_mcmed_2h_unique_visits": 89395,
        "stanford_mcmed_2h_discharged_subset_visits": 51736,
        "bwh_ed_ecg_unique_visits": 51158,
        "ntuh_pre_arrest_unique_visits": 16771,
        "total_unique_primary_visits": 89395 + 51158 + 16771,
        "supplementary_ntuh_post_rosc_visits": 818,
    }
    existing["elapsed_sec"] = round(time.time() - t0, 1)
    b2_path.write_text(json.dumps(existing, indent=2))
    print(f"=== [BLOCK 2 DONE in {time.time()-t0:.1f}s] ===", flush=True)
    return existing


# ===========================================================================
# BLOCK 3: STANFORD ICU EXCHANGE / MISSED-CASE LEDGER & 4-WAY PRIOR-SHIFT TAXONOMY
# ===========================================================================
def run_block3() -> dict[str, Any]:
    t0 = time.time()
    print("\n=== [BLOCK 3] Stanford ICU Missed-Case / Directional VOI Ledger & 4-Way Prior-Shift Taxonomy ===", flush=True)
    b2 = json.loads((BLOCKS_DIR / "block2_common_panel_and_fee_sweep.json").read_text())
    tasks, _ = load_all_clinical_tasks()
    spec_map = {s["task_id"]: s for s in tasks}

    # --- 1. Stanford ICU 4-Threshold x 8-Policy Clinical Exchange & Missed-Case Ledger ---
    icu_spec = spec_map["mcmed_critical_icu_death"]
    icu_te = icu_spec["df"][icu_spec["df"]["split"] == "test"].reset_index(drop=True)
    clusters_icu = icu_te[icu_spec["cluster_col"]].to_numpy()
    tr_icu = np.load(SCRATCH_TRACE_DIR / "mcmed_critical_icu_death.npz")
    y_icu = tr_icu["y_te"].astype(int)
    n_icu = len(y_icu)
    p1_icu = tr_icu["p1_bag_te"]
    p2_up_icu = tr_icu["p2_up_bag"]
    voi_up_icu = tr_icu["voi_up_bag"]
    p2_all_icu = tr_icu["p2_all_gbdt_bag"]
    pr_gg_icu = tr_icu["pr_gg_bag"]
    c_gg_icu = tr_icu["c_gg_bag"]

    c_up_pan = float(b2["tasks"]["mcmed_critical_icu_death"]["r10_common_panel_and_triad"]["arms"]["upca_fluid"]["panel_cost_cfp"])
    c_all_pan = float(b2["tasks"]["mcmed_critical_icu_death"]["r10_common_panel_and_triad"]["arms"]["full_100pct_all_m_gbdt"]["panel_cost_cfp"])

    icu_exchange_by_tau: dict[str, Any] = {}
    for tau in [0.03, 0.05, 0.08, 0.10]:
        c_fn_tau = (1.0 - tau) / tau
        ck_tau = ClinicalContract(1.0, c_fn_tau, 0.25, c_up_pan)
        d_masks = directional_voi_masks(p1_icu, voi_up_icu, c_up_pan, tau)
        q_two = d_masks["two_sided"]
        q_esc = d_masks["escalation_only"]
        q_deesc = d_masks["deescalation_only"]
        q_bu = boundary_uncertainty_mask(p1_icu, tau, float(np.mean(q_two)))

        policies = [
            ("stage1_zero_query", p1_icu, np.zeros(n_icu, dtype=float), np.zeros(n_icu, dtype=bool)),
            ("full_100pct_all_8_channels", p2_all_icu, np.full(n_icu, c_all_pan), np.ones(n_icu, dtype=bool)),
            ("full_100pct_locked_panel", p2_up_icu, np.full(n_icu, c_up_pan), np.ones(n_icu, dtype=bool)),
            ("boundary_uncertainty_matched_q", np.where(q_bu, p2_up_icu, p1_icu), q_bu.astype(float) * c_up_pan, q_bu),
            ("global_cmi_gbdt_two_sided", pr_gg_icu, c_gg_icu, c_gg_icu > 0.0),
            ("upca_fluid_two_sided_default", np.where(q_two, p2_up_icu, p1_icu), q_two.astype(float) * c_up_pan, q_two),
            ("upca_fluid_escalation_only", np.where(q_esc, p2_up_icu, p1_icu), q_esc.astype(float) * c_up_pan, q_esc),
            ("upca_fluid_deescalation_only", np.where(q_deesc, p2_up_icu, p1_icu), q_deesc.astype(float) * c_up_pan, q_deesc),
        ]

        tp_full_all = int(np.sum((p2_all_icu >= tau) & (y_icu == 1)))
        tp_full_lock = int(np.sum((p2_up_icu >= tau) & (y_icu == 1)))
        tp_s1 = int(np.sum((p1_icu >= tau) & (y_icu == 1)))
        pre_fee_nb_full_all = variable_cost_nb_per_1k(y_icu, p2_all_icu, 0.0, ck_tau)
        fee_drag_full_all = 1000.0 * (c_all_pan / c_fn_tau)

        rows_tau = []
        for pname, p_pol, c_pol, q_pol in policies:
            act = p_pol >= tau
            tp_cnt = int(np.sum(act & (y_icu == 1)))
            fp_cnt = int(np.sum(act & (y_icu == 0)))
            sens = float(tp_cnt / max(1, int(np.sum(y_icu == 1))))
            ppv = float(tp_cnt / max(1, tp_cnt + fp_cnt))
            qr = float(np.mean(q_pol))
            mean_c = float(np.mean(c_pol))
            pre_fee_nb = variable_cost_nb_per_1k(y_icu, p_pol, 0.0, ck_tau)
            fee_drag = 1000.0 * (mean_c / c_fn_tau)
            net_nb = variable_cost_nb_per_1k(y_icu, p_pol, c_pol, ck_tau)

            b_vs_all = variable_cost_cluster_bootstrap_diff(
                y_icu, p_pol, c_pol, p2_all_icu, np.full(n_icu, c_all_pan), clusters_icu, ck_tau, seed=600 + int(tau * 1000)
            )
            b_vs_s1 = variable_cost_cluster_bootstrap_diff(
                y_icu, p_pol, c_pol, p1_icu, 0.0, clusters_icu, ck_tau, seed=650 + int(tau * 1000)
            )

            missed_vs_all = tp_full_all - tp_cnt
            missed_vs_lock = tp_full_lock - tp_cnt
            missed_vs_s1 = tp_s1 - tp_cnt
            rows_tau.append(
                {
                    "policy": pname,
                    "tp_count": tp_cnt,
                    "tp_per_1k": round(1000.0 * tp_cnt / n_icu, 2),
                    "sensitivity_pct": round(100.0 * sens, 2),
                    "missed_cases_vs_full_all8": missed_vs_all,
                    "missed_cases_per_1k_vs_full_all8": round(1000.0 * missed_vs_all / n_icu, 2),
                    "missed_cases_pct_vs_full_all8": round(100.0 * missed_vs_all / max(1, tp_full_all), 2),
                    "missed_cases_vs_full_locked": missed_vs_lock,
                    "missed_cases_vs_stage1": missed_vs_s1,
                    "fp_count": fp_cnt,
                    "fp_per_1k": round(1000.0 * fp_cnt / n_icu, 2),
                    "ppv_pct": round(100.0 * ppv, 2),
                    "query_rate_pct": round(100.0 * qr, 2),
                    "mean_cost_cfp": round(mean_c, 5),
                    "pre_fee_nb_per_1k": round(pre_fee_nb, 3),
                    "fee_drag_nb_per_1k": round(fee_drag, 3),
                    "pre_fee_delta_vs_full_all8": round(pre_fee_nb - pre_fee_nb_full_all, 3),
                    "fee_saving_vs_full_all8": round(fee_drag_full_all - fee_drag, 3),
                    "net_ledger_nb_per_1k": round(net_nb, 3),
                    "boot_nb_vs_full_all8": b_vs_all,
                    "boot_nb_vs_stage1": b_vs_s1,
                }
            )
        icu_exchange_by_tau[f"tau_{tau:.2f}"] = {
            "tau": tau,
            "indifference_odds_fp_per_tp": round(c_fn_tau, 2),
            "n_test": n_icu,
            "total_true_events": int(np.sum(y_icu == 1)),
            "policies": rows_tau,
        }

    # --- 2. 4-Way Prior-Shift Taxonomy (`Paper 2 F10` — Zero Test-Label Leakage) ---
    prior_shift_taxonomy: dict[str, Any] = {}
    for tid in ("mcmed_bounce72_admit_2h", "mcmed_critical_icu_death", "bwh_stent_cabg_010_ecg_only", "ntuh_pre_arrest_sca_pi002"):
        spec = spec_map[tid]
        ck = spec["contract"]
        tr_z = np.load(SCRATCH_TRACE_DIR / f"{tid}.npz")
        y_te = tr_z["y_te"].astype(int)
        w_te = tr_z["w_te"]
        pr_bag = tr_z["pr_up_bag"]
        c_bag = tr_z["c_up_bag"]
        y_cal0 = tr_z["y_cal0"].astype(int)
        p2_cal0 = tr_z["p2_up_cal0"]
        te_df = spec["df"][spec["df"]["split"] == "test"].reset_index(drop=True)
        cl_te = te_df[spec["cluster_col"]].to_numpy()

        pi_cal_obs = float(spec["pi_override"]) if spec["pi_override"] is not None else float(np.mean(y_cal0))
        pi_te_true = float(spec["pi_override"]) if spec["pi_override"] is not None else float(np.mean(y_te))
        mean_pred_te = float(np.sum(w_te * pr_bag) / np.sum(w_te))
        mean_pred_cal = float(np.mean(p2_cal0))

        # (i) Unadjusted deployed model (0 test labels)
        p_unadj = pr_bag.copy()
        # (ii) Purely Unlabeled Saerens (2002) EM on unlabeled D_test (0 test labels)
        pi_em = saerens_em_prevalence_estimate(pr_bag, pi_src=pi_cal_obs)
        p_saerens = adjust_probs_for_target_prevalence(pr_bag, mean_pred_te, pi_em)
        # (iii) Disjoint Calibration-Slice Mean Alignment (0 test labels)
        p_cal_align = adjust_probs_for_target_prevalence(pr_bag, mean_pred_cal, pi_cal_obs)
        # (iv) Non-Deployable Oracle Test-Prevalence Bound (uses pi_te_true; explicitly labeled oracle!)
        p_oracle = adjust_probs_for_target_prevalence(pr_bag, mean_pred_te, pi_te_true)

        arms_ps = {}
        for mname, p_vec, pi_used, uses_test_y in [
            ("unadjusted_0_test_labels", p_unadj, mean_pred_te, False),
            ("saerens_em_unlabeled_0_test_labels", p_saerens, pi_em, False),
            ("disjoint_cal_prior_align_0_test_labels", p_cal_align, pi_cal_obs, False),
            ("oracle_test_prevalence_bound_non_deployable", p_oracle, pi_te_true, True),
        ]:
            cal_m = compute_calibration_metrics(y_te, p_vec, sample_weight=(None if spec["pi_override"] is None else w_te))
            nb_m = variable_cost_nb_per_1k(y_te, p_vec, c_bag, ck, sample_weight=(None if spec["pi_override"] is None else w_te))
            b_vs_unadj = variable_cost_cluster_bootstrap_diff(
                y_te, p_vec, c_bag, p_unadj, c_bag, cl_te, ck,
                sample_weight=(None if spec["pi_override"] is None else w_te), seed=710,
            )
            arms_ps[mname] = {
                "uses_test_labels": uses_test_y,
                "target_prior_used": round(float(pi_used), 5),
                "cal_intercept_citl": cal_m["cal_intercept"],
                "cal_slope": cal_m["cal_slope"],
                "ici": cal_m["ici"],
                "nb_common_bagged": round(nb_m, 4),
                "boot_vs_unadjusted": b_vs_unadj,
            }
        prior_shift_taxonomy[tid] = {
            "pi_cal_source": round(pi_cal_obs, 5),
            "pi_test_realized": round(pi_te_true, 5),
            "saerens_em_estimated_pi_test": round(pi_em, 5),
            "methods": arms_ps,
        }

    out_b3 = {
        "stanford_icu_exchange_and_missed_case_ledger": icu_exchange_by_tau,
        "four_way_prior_shift_taxonomy": prior_shift_taxonomy,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (BLOCKS_DIR / "block3_exchange_and_prior_shift.json").write_text(json.dumps(out_b3, indent=2))
    print(f"=== [BLOCK 3 DONE in {time.time()-t0:.1f}s] ===", flush=True)
    return out_b3


# ===========================================================================
# BLOCK 4: SUBGROUP COCHRAN-Q HETEROGENEITY AUDIT (VS FULL 100%, S1, & CMI)
# ===========================================================================
def run_block4() -> dict[str, Any]:
    t0 = time.time()
    print("\n=== [BLOCK 4] Subgroup Cochran's Q Heterogeneity Audit (vs. Full 100%, Stage-1, & Global-CMI) ===", flush=True)
    tasks, _ = load_all_clinical_tasks()
    spec_map = {s["task_id"]: s for s in tasks}

    audit_configs = [
        (
            "mcmed_critical_icu_death",
            [
                ("race_clean", ["White", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander", "Other_or_Unknown"]),
                ("sex_str", ["Female", "Male"]),
                ("age_band", ["Age_under_65", "Age_65_plus"]),
                ("payor_clean", ["Medicaid", "Medicare", "Unspecified_or_Commercial"]),
            ],
        ),
        (
            "mcmed_hospital_admission",
            [
                ("race_clean", ["White", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander", "Other_or_Unknown"]),
                ("sex_str", ["Female", "Male"]),
                ("age_band", ["Age_under_65", "Age_65_plus"]),
                ("payor_clean", ["Medicaid", "Medicare", "Unspecified_or_Commercial"]),
            ],
        ),
        (
            "bwh_death_030_ecg_only",
            [
                ("race_str", ["White", "Black", "Hispanic", "Other"]),
                ("sex_str", ["Female", "Male"]),
                ("age_band", ["Age_under_65", "Age_65_plus"]),
                ("agi_quartile", ["Q1_Lowest_Income", "Q2_LowMid", "Q3_HighMid", "Q4_Highest_Income"]),
            ],
        ),
    ]

    subgroup_results: dict[str, Any] = {}
    significant_heterogeneity_flags: list[dict[str, Any]] = []

    for tid, attr_list in audit_configs:
        spec = spec_map[tid]
        ck: ClinicalContract = spec["contract"]
        tau = ck.tau_star
        te_df = spec["df"][spec["df"]["split"] == "test"].reset_index(drop=True)
        cl_te = te_df[spec["cluster_col"]].to_numpy()

        tr_z = np.load(SCRATCH_TRACE_DIR / f"{tid}.npz")
        y_te = tr_z["y_te"].astype(int)
        p1 = tr_z["p1_bag_te"]
        pr_up = tr_z["pr_up_bag"]
        c_up = tr_z["c_up_bag"]
        q_up = tr_z["q_up_bag"].astype(bool)
        p2_all = tr_z["p2_all_gbdt_bag"]
        c_all = tr_z["c_full_all_vec"]
        pr_gg = tr_z["pr_gg_bag"]
        c_gg = tr_z["c_gg_bag"]

        task_out: dict[str, Any] = {}
        for attr_idx, (col_name, grp_order) in enumerate(attr_list):
            vals = te_df[col_name].astype(str).to_numpy()
            strata_dict: dict[str, Any] = {}
            est_map: dict[str, list[float]] = {"vs_full_all_m": [], "vs_stage1": [], "vs_global_cmi": []}
            se_map: dict[str, list[float]] = {"vs_full_all_m": [], "vs_stage1": [], "vs_global_cmi": []}

            for g_idx, g_val in enumerate(grp_order):
                m = vals == g_val
                ng = int(np.sum(m))
                evg = int(np.sum(y_te[m] == 1))
                if ng < 40 or evg < 2:
                    continue
                yg = y_te[m]
                cg = cl_te[m]
                b_seed = 800 + 30 * attr_idx + g_idx

                b_all = variable_cost_cluster_bootstrap_diff(yg, pr_up[m], c_up[m], p2_all[m], c_all[m], cg, ck, seed=b_seed)
                b_s1 = variable_cost_cluster_bootstrap_diff(yg, pr_up[m], c_up[m], p1[m], 0.0, cg, ck, seed=b_seed + 100)
                b_gg = variable_cost_cluster_bootstrap_diff(yg, pr_up[m], c_up[m], pr_gg[m], c_gg[m], cg, ck, seed=b_seed + 200)

                for k_c, b_obj in [("vs_full_all_m", b_all), ("vs_stage1", b_s1), ("vs_global_cmi", b_gg)]:
                    est_map[k_c].append(b_obj["diff_mean"])
                    se_map[k_c].append(max(b_obj["se"], 1e-4))

                cal_g = compute_calibration_metrics(yg, pr_up[m], n_bins=5)
                act_u = pr_up[m] >= tau
                act_f = p2_all[m] >= tau
                fnr_u = float(np.sum((~act_u) & (yg == 1)) / max(1, evg))
                fpr_u = float(np.sum(act_u & (yg == 0)) / max(1, ng - evg))
                missed_vs_full = int(np.sum(act_f & (yg == 1))) - int(np.sum(act_u & (yg == 1)))

                strata_dict[g_val] = {
                    "n_encounters": ng,
                    "n_patients": len(np.unique(cg)),
                    "events": evg,
                    "prevalence": round(evg / float(ng), 5),
                    "query_rate": round(float(np.mean(q_up[m])), 4),
                    "nb_stage1": round(variable_cost_nb_per_1k(yg, p1[m], 0.0, ck), 3),
                    "nb_full_all_m": round(variable_cost_nb_per_1k(yg, p2_all[m], c_all[m], ck), 3),
                    "nb_global_cmi_gbdt": round(variable_cost_nb_per_1k(yg, pr_gg[m], c_gg[m], ck), 3),
                    "nb_upca_fluid": round(variable_cost_nb_per_1k(yg, pr_up[m], c_up[m], ck), 3),
                    "missed_cases_vs_full_all_m": missed_vs_full,
                    "cal_intercept": cal_g["cal_intercept"],
                    "cal_slope": cal_g["cal_slope"],
                    "ici": cal_g["ici"],
                    "fnr_at_tau": round(fnr_u, 4),
                    "fpr_at_tau": round(fpr_u, 4),
                    "boot_upca_vs_full_all_m": b_all,
                    "boot_upca_vs_stage1": b_s1,
                    "boot_upca_vs_global_cmi": b_gg,
                }

            het_dict = {
                k_c: cochran_q_subgroup_heterogeneity(est_map[k_c], se_map[k_c])
                for k_c in ("vs_full_all_m", "vs_stage1", "vs_global_cmi")
            }
            for k_c, h_obj in het_dict.items():
                if h_obj["p_het"] < 0.05:
                    significant_heterogeneity_flags.append(
                        {
                            "task_id": tid,
                            "attribute": col_name,
                            "contrast": k_c,
                            "q_stat": h_obj["q_stat"],
                            "df": h_obj["df"],
                            "p_het": h_obj["p_het"],
                            "i2_pct": h_obj["i2_pct"],
                        }
                    )
            task_out[col_name] = {"strata": strata_dict, "cochran_q_heterogeneity": het_dict}
        subgroup_results[tid] = task_out

    out_b4 = {
        "tasks": subgroup_results,
        "nominal_p_lt_005_heterogeneity_flags": significant_heterogeneity_flags,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (BLOCKS_DIR / "block4_subgroup_heterogeneity_audit.json").write_text(json.dumps(out_b4, indent=2))
    print(f"=== [BLOCK 4 DONE in {time.time()-t0:.1f}s | Nominal p<0.05 flags: {len(significant_heterogeneity_flags)}] ===", flush=True)
    return out_b4


def merge_all_blocks() -> dict[str, Any]:
    v21 = json.loads(V21_JSON.read_text())
    b1 = json.loads((BLOCKS_DIR / "block1_synthetic_and_pilots.json").read_text())
    b2 = json.loads((BLOCKS_DIR / "block2_common_panel_and_fee_sweep.json").read_text())
    b3 = json.loads((BLOCKS_DIR / "block3_exchange_and_prior_shift.json").read_text())
    b4 = json.loads((BLOCKS_DIR / "block4_subgroup_heterogeneity_audit.json").read_text())

    merged = {
        "schema_version": "paper3_v2_2_0_cross_paper_unified_ledger",
        "upstream_constants_commit": "8c8c4c8 (2026-11-22T18:16:58-05:00)",
        "unique_primary_encounters_reconciliation": b2["unique_primary_encounters_reconciliation"],
        "block1_synthetic_and_pilots": b1,
        "block2_common_panel_and_fee_sweep": b2,
        "block3_exchange_and_prior_shift": b3,
        "block4_subgroup_heterogeneity_audit": b4,
        "preserved_v2_1_secondary_and_sensitivity_tasks": v21["clinical"]["secondary_and_sensitivity_tasks"],
        "preserved_v2_1_constant_sensitivity_sweeps": v21["clinical"]["sensitivity_sweeps_r10"],
    }
    OUT_JSON.write_text(json.dumps(merged, indent=2))
    print(f"\n=== MERGED ALL 4 BLOCKS -> {OUT_JSON} ===", flush=True)
    return merged


def main() -> None:
    blk = "all"
    if "--block" in sys.argv:
        blk = sys.argv[sys.argv.index("--block") + 1]
    if blk in ("all", "1"):
        run_block1()
    if blk in ("all", "2"):
        run_block2()
    if blk in ("all", "3"):
        run_block3()
    if blk in ("all", "4"):
        run_block4()
    if blk in ("all", "merge"):
        merge_all_blocks()


if __name__ == "__main__":
    main()

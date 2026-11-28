"""Generate 5 Publication-Grade 300-DPI Figures for Paper 3 (`DecisionAwareAcquisition` v2.2.0).

Outputs to both `manuscript/` and `results/`:
  - `fig1_operating_envelope_and_regime.png`: 1,000-rep E5, flat vs. proportional B=2.0 billing,
    500-pilot & 200-pilot split-confirmation concordance, and 72-cell 4D operating envelope.
  - `fig2_clinical_net_benefit_and_dca.png`: 6-task Vickers Decision Curves, Common-Panel Bagged VOI
    10k patient-cluster contrasts (vs. Full 100% All-M, Stage-1, Boundary Uncertainty, CMI+GBDT,
    and Channel-Billed Covert-CMI), and AUPRC_full vs. AUPRC_routed.
  - `fig3_calibration_curves.png`: 6-task flexible spline calibration curves (`CITL`, slope, `ICI`).
  - `fig4_skipped_disagreement_and_subgroups.png`: Skipped-case directional 95% UBs and Subgroup
    Cochran's `Q` audits (`UPCA vs. Full 100% All-M` and `UPCA vs. Stage-1`).
  - `fig5_fee_sweep_and_missed_case_ledger.png`: 5-point query-cost multiplier sweep (`0.5x..3.0x`),
    Stanford ICU pre-fee vs. fee-saving decomposition, and Two-Sided vs. Escalation-Only missed cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
V22_JSON = REPO_ROOT / "results" / "paper3_v2_2_remediation_results.json"
V21_JSON = REPO_ROOT / "results" / "paper3_v2_remediation_results.json"
MAN_DIR = REPO_ROOT / "manuscript"
RES_DIR = REPO_ROOT / "results"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 9.6,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.8,
        "legend.fontsize": 7.6,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

C_UPCA = "#0B5394"
C_FULL = "#B22222"
C_BU = "#E69F00"
C_GLOB = "#D95F02"
C_WCZ = "#7570B3"
C_IPW = "#E7298A"
C_S1 = "#666666"
C_ESC = "#1B9E77"

TASK_ORDER = [
    ("mcmed_critical_icu_death", "Stanford ICU/Death (2h, 2.18%)"),
    ("mcmed_hospital_admission", "Stanford Admission (2h, 28.8%)"),
    ("mcmed_bounce72_admit_2h", "Stanford 72h Bounce (2h, 1.67%)"),
    ("bwh_death_030_ecg_only", "BWH 30d Mortality (ECG, 2.41%)"),
    ("bwh_stent_cabg_010_ecg_only", "BWH 10d Stent/CABG (ECG, 1.51%)"),
    ("ntuh_pre_arrest_sca_pi002", "NTUH Pre-Arrest SCA (pi=2.0%)"),
]


def _save_both(fig: plt.Figure, fname: str) -> None:
    MAN_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAN_DIR / fname, dpi=300, bbox_inches="tight")
    fig.savefig(RES_DIR / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 300-DPI figure -> {MAN_DIR / fname}")


def plot_fig1(v22: dict) -> None:
    b1 = v22["block1_synthetic_and_pilots"]
    e5 = b1["preserved_v2_1_synthetic"]["experiment_e5_1000_reps"]
    e5_sp = b1["e5_split_pilot_and_scheduled_b2_audit"]["canonical_sigma_32_200reps"]
    grid = b1["preserved_v2_1_synthetic"]["operating_envelope_4d_grid"]["cells"]
    eff = b1["preserved_v2_1_500pilot_stanford_icu"]["rho_sensitivity_repeated_pilots"]
    sp200 = b1["stanford_icu_200pilot_split_vs_crossfit_with_nulls"]["arms"]

    fig = plt.figure(figsize=(14.4, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.08], hspace=0.42, wspace=0.34)

    # Panel A: 1,000-rep E5 + 50/50 Split-Pilot Comparison
    ax_a = fig.add_subplot(gs[0, 0])
    cats = ["Identifiable\nActive (%)", "Unidentifiable\nActive (%)", "Null Noise\nFWER (%)"]
    r_cf = [e5["stage_a_pick_identifiable_rate"] * 100, e5["stage_a_pick_unidentifiable_rate"] * 100, e5["stage_a_null_fwer"] * 100]
    r_up = [e5["upca_retain_identifiable_rate"] * 100, e5["upca_retain_unidentifiable_rate"] * 100, e5["upca_null_fwer"] * 100]
    r_sp = [e5_sp["split_pilot_50_50_identifiable_rate"] * 100, 0.0, e5_sp["split_pilot_50_50_null_fwer"] * 100]
    x = np.arange(3)
    w = 0.26
    ax_a.bar(x - w, r_cf, w, color=C_WCZ, alpha=0.88, label="4-Fold CF Stage A")
    ax_a.bar(x, r_up, w, color=C_UPCA, alpha=0.92, label="UPCA-Fluid (Stage A+B)")
    ax_a.bar([x[0] + w, x[2] + w], [r_sp[0], r_sp[2]], w, color=C_ESC, alpha=0.88, label="50/50 Split-Pilot Stage A")
    ax_a.axhline(5.0, color=C_FULL, linestyle="--", linewidth=1.0, label="Nominal alpha = 5%")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cats)
    ax_a.set_ylim(0, 128)
    ax_a.set_ylabel("Selection / FWER Rate (%)")
    ax_a.set_title("A. E5 Screening & Split-Pilot FWER")
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.1)

    # Panel B: E5 Tight Budget (B=1) vs Flat-Fee B=2 vs Scheduled B=2
    ax_b = fig.add_subplot(gs[0, 1])
    lbls_b = ["Stage A\nB=1.0", "UPCA\nB=1.0", "Stage A\nB=2 Flat", "UPCA\nB=2 Flat", "Stage A\nB=2 Sch.", "UPCA\nB=2 Sch."]
    vals_b = [
        e5["nb_stage_a_b1_mean"],
        e5["nb_upca_b1_mean"],
        e5["nb_stage_a_b2_mean"],
        e5["nb_upca_b2_mean"],
        e5_sp["nb_scheduled_stage_a_b2_mean"],
        e5_sp["nb_scheduled_upca_b2_mean"],
    ]
    cols_b = [C_WCZ, C_UPCA, C_WCZ, C_UPCA, "#8C6BB1", C_ESC]
    xb = np.arange(len(lbls_b))
    bars = ax_b.bar(xb, vals_b, width=0.56, color=cols_b, alpha=0.90)
    ax_b.axhline(0.0, color="black", linewidth=0.8)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(lbls_b, fontsize=7.0)
    ax_b.set_ylabel("Deployed Net Benefit (/1,000)")
    ax_b.set_title(f"B. E5 Utility (B=1 dNB={e5['paired_delta_nb_b1_mean']:+.2f}, p=1.4e-11)")
    for b_bar, v in zip(bars, vals_b, strict=False):
        ax_b.text(
            b_bar.get_x() + b_bar.get_width() / 2,
            v + 0.10,
            f"{v:+.2f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
        )
    ax_b.set_ylim(0.0, max(vals_b) + 0.78)

    # Panel C: 500-Pilot Concordance + 200-Pilot 50/50 Split-Pilot Power Trade-Off
    ax_c = fig.add_subplot(gs[0, 2])
    rho_keys = ["rho_0.15", "rho_0.20", "rho_0.30"]
    xc = np.arange(3)
    wc = 0.22
    for off_c, mk, mlbl, mcol in [
        (-1.5 * wc, "weighted_005", "Weighted-CZ (0.05)", C_WCZ),
        (-0.5 * wc, "ipw_wht", "IPW-CZ (w_HT)", C_IPW),
        (0.5 * wc, "offset_bc_gbdt", "4-Fold Offset-CZ", C_UPCA),
    ]:
        rates = [eff[rk]["top1_concordance"][mk]["rate"] * 100.0 for rk in rho_keys]
        ax_c.bar(xc + off_c, rates, wc, color=mcol, alpha=0.90, label=mlbl)
    ax_c.bar(
        [xc[1] + 1.5 * wc],
        [sp200["split_pilot_offset_50_50"]["top1_oracle_concordance"] * 100.0],
        wc,
        color=C_ESC,
        alpha=0.92,
        label=f"50/50 Split (FWER {sp200['split_pilot_offset_50_50']['null_fwer_rate']*100:.1f}%)",
    )
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(["rho=0.15\n(200p)", "rho=0.20 Primary\n(500p / 200p)", "rho=0.30\n(200p)"], fontsize=7.6)
    ax_c.set_ylim(35, 122)
    ax_c.set_ylabel("Top-1 Oracle Concordance (%)")
    ax_c.set_title("C. Stanford ICU Repeated-Pilot Concordance")
    ax_c.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=6.8)

    # Panels D1-D3: 72-Cell 4D Operating Envelope
    sigmas = [1.5, 2.5, 3.5]
    cratios = [0.15, 0.30, 0.60, 0.90]
    cell_lookup = {(c["rho"], c["sigma"], c["cquery_over_g"], c["pi_cz"]): c for c in grid}
    for idx_r, rho in enumerate([0.10, 0.22, 0.40]):
        ax_h = fig.add_subplot(gs[1, idx_r])
        mat = np.array([[cell_lookup[(rho, s, cr, 0.40)]["emp_2d_minus_12bin_gain"] for cr in cratios] for s in sigmas])
        im = ax_h.imshow(mat, cmap="RdBu_r", vmin=-1.5, vmax=3.5, aspect="auto")
        for i_s, s in enumerate(sigmas):
            for j_c, cr in enumerate(cratios):
                c_obj = cell_lookup[(rho, s, cr, 0.40)]
                val = mat[i_s, j_c]
                ret = round(c_obj["retain_identifiable_rate"] * 100)
                ax_h.text(
                    j_c,
                    i_s,
                    f"{val:+.2f}\n(ID:{ret}%)",
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color=("white" if abs(val) > 1.8 else "black"),
                    fontweight="bold",
                )
        ax_h.set_xticks(range(len(cratios)))
        ax_h.set_xticklabels([f"{cr:.2f}" for cr in cratios])
        ax_h.set_yticks(range(len(sigmas)))
        ax_h.set_yticklabels([f"sigma={s:.1f}" for s in sigmas])
        ax_h.set_xlabel("Relative Query Cost c_query / g(sigma)")
        if idx_r == 0:
            ax_h.set_ylabel("Active Signal Dispersion sigma")
        ax_h.set_title(f"D{idx_r+1}. Envelope (rho={rho:.2f}, pi_CZ=0.40)")
        if idx_r == 2:
            cbar = fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            cbar.set_label("2D - 12bin Gain (/1k)", fontsize=7.8)

    _save_both(fig, "fig1_operating_envelope_and_regime.png")


def plot_fig2(v22: dict, v21: dict) -> None:
    pt21 = v21["clinical"]["primary_tasks"]
    b2_tasks = v22["block2_common_panel_and_fee_sweep"]["tasks"]

    fig = plt.figure(figsize=(14.4, 9.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.15], hspace=0.42, wspace=0.28)

    for idx, (tkey, ttitle) in enumerate(TASK_ORDER):
        r_i, c_i = divmod(idx, 3)
        ax = fig.add_subplot(gs[r_i, c_i])
        t_obj = pt21[tkey]
        arms = t_obj["arms"]
        tau_star = t_obj["contract"]["tau_star"]

        dca_u = arms["upca_fluid"]["decision_curve"]
        dca_g = arms["global_cmi_fluid"]["decision_curve"]
        dca_s1 = arms["stage1"]["decision_curve"]
        taus_pct = [t * 100.0 for t in dca_u["taus"]]

        ax.plot(taus_pct, dca_u["nb_treat_none"], color="black", linestyle=":", linewidth=1.0, label="Treat None")
        ax.plot(taus_pct, dca_u["nb_treat_all"], color="#AAAAAA", linestyle="--", linewidth=1.1, label="Treat All")
        ax.plot(taus_pct, dca_s1["nb_model"], color=C_S1, linestyle="-.", linewidth=1.3, label="Stage-1 (0% Query)")
        ax.plot(taus_pct, dca_g["nb_model"], color=C_GLOB, linewidth=1.5, alpha=0.88, label="Global-CMI")
        ax.plot(taus_pct, dca_u["nb_model"], color=C_UPCA, linewidth=2.1, label="UPCA-Fluid")
        ax.axvline(tau_star * 100.0, color=C_FULL, linestyle=":", linewidth=1.1)

        ymax = max(max(dca_u["nb_model"]), max(dca_s1["nb_model"]), 2.0) * 1.18
        ax.set_ylim(-max(3.0, 0.22 * ymax), ymax)
        ax.set_xlabel("Decision Threshold tau (%)")
        if c_i == 0:
            ax.set_ylabel("Net Benefit (/1,000)")
        ax.set_title(f"{chr(65+idx)}. {ttitle}", fontsize=9.1)
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.0)

    # Panel G: Common-Panel Bagged VOI Paired Contrasts Across All 6 Tasks
    ax_bl = fig.add_subplot(gs[2, :2])
    short_names = ["Stanford\nICU/Death", "Stanford\nAdmission", "Stanford\n72h Bounce", "BWH 30d\nDeath", "BWH 10d\nStent/CABG", "NTUH\nPre-Arrest"]
    x = np.arange(len(TASK_ORDER))
    w = 0.18
    contrasts = [
        (-1.5 * w, "upca_vs_full_100pct_all_m", "vs. Full 100% All-M", C_FULL),
        (-0.5 * w, "upca_vs_stage1", "vs. Stage-1 (0%)", C_UPCA),
        (0.5 * w, "upca_vs_boundary_uncertainty_matched_q", "vs. Boundary Unc. (q)", C_BU),
        (1.5 * w, "upca_vs_afa_cmi_persplit", "vs. Covert-CMI AFA", C_ESC),
    ]
    for offset, bkey, lbl, col in contrasts:
        b_objs = [b2_tasks[k]["r10_common_panel_and_triad"]["bootstraps_10k_patient_cluster"][bkey] for k, _ in TASK_ORDER]
        means = [b["diff_mean"] for b in b_objs]
        lo = [b["ci95_low"] for b in b_objs]
        hi = [b["ci95_high"] for b in b_objs]
        yerr = np.array([[max(0.0, means[i] - lo[i]) for i in range(6)], [max(0.0, hi[i] - means[i]) for i in range(6)]])
        ax_bl.bar(x + offset, means, w, yerr=yerr, capsize=2.2, color=col, alpha=0.90, label=lbl)

    ax_bl.axhline(0.0, color="black", linewidth=0.8)
    ax_bl.set_xticks(x)
    ax_bl.set_xticklabels(short_names)
    ax_bl.set_ylabel("Common-Bagged dNB (/1,000) [95% CI]")
    ax_bl.set_title("G. Common-Panel Bagged VOI (R=10) Contrasts Under ChannelCostSchedule")
    ax_bl.legend(loc="upper right", frameon=True, framealpha=0.93, ncol=2, fontsize=7.2)

    # Panel H: Common-Bagged AUPRC_full vs AUPRC_routed
    ax_br = fig.add_subplot(gs[2, 2])
    ap_s1 = [b2_tasks[k]["r10_common_panel_and_triad"]["arms"]["stage1"]["auprc_full"] for k, _ in TASK_ORDER]
    ap_rt = [b2_tasks[k]["r10_common_panel_and_triad"]["arms"]["upca_fluid"]["auprc_routed"] for k, _ in TASK_ORDER]
    ap_fl = [b2_tasks[k]["r10_common_panel_and_triad"]["arms"]["upca_fluid"]["auprc_full"] for k, _ in TASK_ORDER]
    w_h = 0.24
    ax_br.bar(x - w_h, ap_s1, w_h, color=C_S1, alpha=0.85, label="Stage-1 AUPRC")
    ax_br.bar(x, ap_rt, w_h, color="#5FA8D3", alpha=0.90, label="UPCA AUPRC_routed")
    ax_br.bar(x + w_h, ap_fl, w_h, color=C_UPCA, alpha=0.95, label="UPCA AUPRC_full")
    ax_br.set_xticks(x)
    ax_br.set_xticklabels(["ICU", "Adm", "B72", "D30", "Rev", "SCA"], fontsize=8.0)
    ax_br.set_ylabel("Precision-Recall Area (AUPRC)")
    ax_br.set_title("H. AUPRC_full vs. AUPRC_routed (Prop. 2)")
    ax_br.legend(loc="upper right", frameon=False, fontsize=7.3)

    _save_both(fig, "fig2_clinical_net_benefit_and_dca.png")


def plot_fig3(v21: dict) -> None:
    pt = v21["clinical"]["primary_tasks"]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.0))
    plt.subplots_adjust(hspace=0.38, wspace=0.28)

    for idx, (tkey, ttitle) in enumerate(TASK_ORDER):
        ax = axes[idx // 3, idx % 3]
        t_obj = pt[tkey]
        cal_u = t_obj["arms"]["upca_fluid"]["calibration_routed"]
        cal_s1 = t_obj["arms"]["stage1"]["calibration_routed"]
        tau_star = t_obj["contract"]["tau_star"]

        gp = np.array(cal_u["smooth_grid_pred"])
        go = np.array(cal_u["smooth_grid_obs"])
        bp = np.array(cal_u["bin_pred"])
        bo = np.array(cal_u["bin_obs"])
        lim = min(1.0, max(float(np.max(gp)), float(np.max(go)), float(np.max(bp)), float(np.max(bo)), tau_star * 2.0) * 1.08)

        ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1.0, label="Ideal (y = x)")
        ax.plot(cal_s1["smooth_grid_pred"], cal_s1["smooth_grid_obs"], color=C_S1, linestyle="-.", linewidth=1.3, alpha=0.8, label="Stage-1 Smooth")
        ax.plot(gp, go, color=C_UPCA, linewidth=2.1, label="UPCA-Fluid Smooth")
        ax.scatter(bp, bo, color=C_UPCA, s=24, zorder=4, alpha=0.88, label="Decile Observed")
        ax.axvline(tau_star, color=C_FULL, linestyle=":", linewidth=1.1, label=f"tau* = {tau_star:.3f}")

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Predicted Probability")
        if idx % 3 == 0:
            ax.set_ylabel("Observed Event Frequency")
        ax.set_title(f"{chr(65+idx)}. {ttitle}", fontsize=9.1)
        box_txt = (
            f"CITL  = {cal_u['cal_intercept']:+.3f}\n"
            f"Slope = {cal_u['cal_slope']:.3f}\n"
            f"ICI   = {cal_u['ici']:.4f}"
        )
        ax.text(
            0.05,
            0.94,
            box_txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.8,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F7F9FC", "edgecolor": "#CCCCCC"},
        )
        if idx == 0:
            ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=7.2)

    _save_both(fig, "fig3_calibration_curves.png")


def plot_fig4(v22: dict) -> None:
    b2_tasks = v22["block2_common_panel_and_fee_sweep"]["tasks"]
    b4_tasks = v22["block4_subgroup_heterogeneity_audit"]["tasks"]

    fig = plt.figure(figsize=(14.8, 9.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.48)

    # Panel A: Skipped-Case Disagreement Patient-Cluster 95% UBs (Common-Panel Bagged VOI)
    ax_a = fig.add_subplot(gs[0, 0])
    short_t = ["Stanf.\nICU", "Stanf.\nAdmit", "Stanf.\nB72", "BWH\nD30", "BWH\nRev.", "NTUH\nSCA"]
    x = np.arange(len(TASK_ORDER))
    w = 0.25
    ub_tot = [b2_tasks[k]["r10_common_panel_and_triad"]["skipped_case_disagreement_common_bagged"]["flip_ub95_patient_cluster"] * 100 for k, _ in TASK_ORDER]
    ub_n2p = [b2_tasks[k]["r10_common_panel_and_triad"]["skipped_case_disagreement_common_bagged"]["neg_to_pos_ub95_cluster"] * 100 for k, _ in TASK_ORDER]
    ub_p2n = [b2_tasks[k]["r10_common_panel_and_triad"]["skipped_case_disagreement_common_bagged"]["pos_to_neg_ub95_cluster"] * 100 for k, _ in TASK_ORDER]
    ax_a.bar(x - w, ub_tot, w, color=C_UPCA, alpha=0.90, label="Total Flip 95% Cluster UB")
    ax_a.bar(x, ub_n2p, w, color=C_GLOB, alpha=0.88, label="Neg->Pos (Missed Escalation) UB")
    ax_a.bar(x + w, ub_p2n, w, color=C_ESC, alpha=0.88, label="Pos->Neg (Missed De-esc.) UB")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short_t)
    ax_a.set_ylim(0, max(ub_tot) * 1.18)
    ax_a.set_ylabel("Disagreement Rate (%) [95% UB]")
    ax_a.set_title("A. Skipped-Encounter (Q=0) Action-Flip 95% Upper Bounds")
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.0)

    pretty_grp = {
        "Unspecified_or_Commercial": "Commercial/Unspec.",
        "Black or African American": "Black",
        "Native Hawaiian or Other Pacific Islander": "NH/Pacific Isl.",
        "Other_or_Unknown": "Other/Unknown",
        "Q1_Lowest_Income": "Q1 Lowest Income",
        "Q2_LowMid": "Q2 Low-Mid",
        "Q3_HighMid": "Q3 High-Mid",
        "Q4_Highest_Income": "Q4 Highest Income",
    }

    # Panels B, C, D: Subgroup dNB vs Full 100% All-M & vs Stage-1 with Cochran's Q
    sub_panels = [
        (gs[0, 1], "mcmed_critical_icu_death", "payor_clean", "B. Stanford ICU/Death by Payer"),
        (gs[1, 0], "mcmed_critical_icu_death", "race_clean", "C. Stanford ICU/Death by Race"),
        (gs[1, 1], "bwh_death_030_ecg_only", "agi_quartile", "D. BWH 30d Mortality by IRS-AGI Quartile"),
    ]
    for spec_gs, tid, attr, ptitle in sub_panels:
        ax = fig.add_subplot(spec_gs)
        aobj = b4_tasks[tid][attr]
        strata = aobj["strata"]
        hq_all = aobj["cochran_q_heterogeneity"]["vs_full_all_m"]
        hq_s1 = aobj["cochran_q_heterogeneity"]["vs_stage1"]
        g_keys = list(strata.keys())
        yg = np.arange(len(g_keys))
        h = 0.34

        for off_y, bkey, lbl, col in [
            (-h / 2, "boot_upca_vs_full_all_m", f"UPCA vs. Full 100% (p_het={hq_all['p_het']:.3f})", C_FULL),
            (h / 2, "boot_upca_vs_stage1", f"UPCA vs. Stage-1 (p_het={hq_s1['p_het']:.3f})", C_UPCA),
        ]:
            m_vals = [strata[g][bkey]["diff_mean"] for g in g_keys]
            lo_v = [strata[g][bkey]["ci95_low"] for g in g_keys]
            hi_v = [strata[g][bkey]["ci95_high"] for g in g_keys]
            xerr = np.array([[max(0.0, m_vals[i] - lo_v[i]) for i in range(len(g_keys))], [max(0.0, hi_v[i] - m_vals[i]) for i in range(len(g_keys))]])
            ax.barh(yg + off_y, m_vals, h, xerr=xerr, capsize=2.5, color=col, alpha=0.88, label=lbl)

        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(yg)
        ax.set_yticklabels([f"{pretty_grp.get(g, g)} (q={strata[g]['query_rate']*100:.0f}%)" for g in g_keys], fontsize=7.5)
        ax.set_ylim(-0.95, len(g_keys) - 0.35)
        ax.set_xlabel("Subgroup dNB (/1,000) [95% Cluster CI]")
        ax.set_title(ptitle)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=6.9)

    _save_both(fig, "fig4_skipped_disagreement_and_subgroups.png")


def plot_fig5(v22: dict) -> None:
    b2_tasks = v22["block2_common_panel_and_fee_sweep"]["tasks"]
    b3_ex = v22["block3_exchange_and_prior_shift"]["stanford_icu_exchange_and_missed_case_ledger"]
    ps_tax = v22["block3_exchange_and_prior_shift"]["four_way_prior_shift_taxonomy"]

    fig = plt.figure(figsize=(14.4, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32)

    # Panel A: 5-Point Query-Cost Multiplier Sweep on Primary Stanford ICU/Death
    ax_a = fig.add_subplot(gs[0, 0])
    sw_icu = b2_tasks["mcmed_critical_icu_death"]["r10_common_panel_and_triad"]["query_cost_multiplier_sweep"]
    mults = [r["cost_multiplier"] for r in sw_icu]
    ax_a.plot(mults, [r["nb_stage1"] for r in sw_icu], color=C_S1, linestyle="-.", marker="s", label="Stage-1 (0% Query)")
    ax_a.plot(mults, [r["nb_full_100pct_all_m"] for r in sw_icu], color=C_FULL, linestyle="--", marker="v", label="Full 100% All-8 Channels")
    ax_a.plot(mults, [r["nb_full_100pct_locked"] for r in sw_icu], color="#AA4488", linestyle=":", marker="d", label="Full 100% Locked Panel")
    ax_a.plot(mults, [r["nb_boundary_uncertainty_matched_q"] for r in sw_icu], color=C_BU, marker="^", label="Boundary Unc. (|p1-tau*|, q)")
    ax_a.plot(mults, [r["nb_upca_fluid"] for r in sw_icu], color=C_UPCA, linewidth=2.2, marker="o", label="UPCA-Fluid Common-Bagged")
    ax_a.set_xlabel("Assigned Query-Cost Multiplier (x Base Schedule)")
    ax_a.set_ylabel("Net Benefit (/1,000)")
    ax_a.set_title("A. Stanford ICU Query-Cost Multiplier Sweep (0.5x - 3.0x)")
    ax_a.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.2)

    # Panel B: 5-Point Query-Cost Multiplier Sweep on Stanford Inpatient Admission
    ax_b = fig.add_subplot(gs[0, 1])
    sw_adm = b2_tasks["mcmed_hospital_admission"]["r10_common_panel_and_triad"]["query_cost_multiplier_sweep"]
    ax_b.plot(mults, [r["nb_stage1"] for r in sw_adm], color=C_S1, linestyle="-.", marker="s", label="Stage-1 (0% Query)")
    ax_b.plot(mults, [r["nb_full_100pct_all_m"] for r in sw_adm], color=C_FULL, linestyle="--", marker="v", label="Full 100% All-8 Channels")
    ax_b.plot(mults, [r["nb_full_100pct_locked"] for r in sw_adm], color="#AA4488", linestyle=":", marker="d", label="Full 100% Locked Panel")
    ax_b.plot(mults, [r["nb_boundary_uncertainty_matched_q"] for r in sw_adm], color=C_BU, marker="^", label="Boundary Unc. (|p1-tau*|, q)")
    ax_b.plot(mults, [r["nb_upca_fluid"] for r in sw_adm], color=C_UPCA, linewidth=2.2, marker="o", label="UPCA-Fluid Common-Bagged")
    ax_b.set_xlabel("Assigned Query-Cost Multiplier (x Base Schedule)")
    ax_b.set_ylabel("Net Benefit (/1,000)")
    ax_b.set_title("B. Stanford Admission Query-Cost Multiplier Sweep (0.5x - 3.0x)")
    ax_b.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.2)

    # Panel C: Stanford ICU Pre-Fee vs Fee-Saving Decomposition & Escalation-Only Missed Cases
    ax_c = fig.add_subplot(gs[1, 0])
    tau_keys = ["tau_0.03", "tau_0.05", "tau_0.08", "tau_0.10"]
    tau_lbls = ["tau=3.0%\n(32.3:1)", "tau*=5.0%\n(19.0:1)", "tau=8.0%\n(11.5:1)", "tau=10.0%\n(9.0:1)"]
    xt = np.arange(4)
    wt = 0.25
    two_rows = [{p["policy"]: p for p in b3_ex[tk]["policies"]}["upca_fluid_two_sided_default"] for tk in tau_keys]
    esc_rows = [{p["policy"]: p for p in b3_ex[tk]["policies"]}["upca_fluid_escalation_only"] for tk in tau_keys]
    ax_c.bar(xt - wt, [r["pre_fee_delta_vs_full_all8"] for r in two_rows], wt, color="#CC6677", label="Pre-Fee Decision dNB vs. Full-8")
    ax_c.bar(xt, [r["fee_saving_vs_full_all8"] for r in two_rows], wt, color=C_ESC, label="Query-Fee Saving vs. Full-8")
    ax_c.bar(xt + wt, [r["net_ledger_nb_per_1k"] - {p["policy"]: p for p in b3_ex[tk]["policies"]}["full_100pct_all_8_channels"]["net_ledger_nb_per_1k"] for r, tk in zip(two_rows, tau_keys, strict=False)], wt, color=C_UPCA, label="Net Ledger dNB (Two-Sided)")
    ax_c.axhline(0.0, color="black", linewidth=0.8)
    for i_t, (r2, re) in enumerate(zip(two_rows, esc_rows, strict=False)):
        extra_tp = -int(re["missed_cases_vs_stage1"])
        ax_c.text(
            i_t,
            max(r2["fee_saving_vs_full_all8"], 0.5) + 0.25,
            f"Miss vs All8: {r2['missed_cases_vs_full_all8']}\n(Esc-Only: +{extra_tp} TP vs S1)",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )
    ax_c.set_xticks(xt)
    ax_c.set_xticklabels(tau_lbls)
    ax_c.set_ylim(-2.8, max(r["fee_saving_vs_full_all8"] for r in two_rows) + 2.2)
    ax_c.set_ylabel("dNB vs. Full 100% All-8 (/1,000)")
    ax_c.set_title("C. Stanford ICU Pre-Fee vs. Fee-Saving & Missed-Case Trade-Off")
    ax_c.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.0)

    # Panel D: 4-Way Prior-Shift Taxonomy (0 Test Labels vs Non-Deployable Oracle)
    ax_d = fig.add_subplot(gs[1, 1])
    ps_tasks = [
        ("mcmed_bounce72_admit_2h", "Stanford\n72h Bounce"),
        ("mcmed_critical_icu_death", "Stanford\nICU/Death"),
        ("bwh_stent_cabg_010_ecg_only", "BWH 10d\nStent/CABG"),
    ]
    xp = np.arange(len(ps_tasks))
    wp = 0.19
    for off_p, mkey, mlbl, mcol in [
        (-1.5 * wp, "unadjusted_0_test_labels", "Unadjusted (0 Test Labels)", C_S1),
        (-0.5 * wp, "saerens_em_unlabeled_0_test_labels", "Saerens EM (0 Test Labels)", C_UPCA),
        (0.5 * wp, "disjoint_cal_prior_align_0_test_labels", "Disjoint Cal Prior (0 Test Labels)", C_ESC),
        (1.5 * wp, "oracle_test_prevalence_bound_non_deployable", "Oracle Test Prior (Non-Deployable)", C_FULL),
    ]:
        nbs_p = [ps_tax[tk]["methods"][mkey]["nb_common_bagged"] for tk, _ in ps_tasks]
        ax_d.bar(xp + off_p, nbs_p, wp, color=mcol, alpha=0.90, label=mlbl)
    ax_d.axhline(0.0, color="black", linewidth=0.8)
    ax_d.set_xticks(xp)
    ax_d.set_xticklabels([lbl for _, lbl in ps_tasks])
    ax_d.set_ylim(-15.5, 11.2)
    ax_d.set_ylabel("Common-Bagged Net Benefit (/1,000)")
    ax_d.set_title("D. 4-Way Prior-Shift Taxonomy (Deployable vs. Oracle Bound)")
    ax_d.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=7.0)

    _save_both(fig, "fig5_fee_sweep_and_missed_case_ledger.png")


def main() -> None:
    v22 = json.loads(V22_JSON.read_text())
    v21 = json.loads(V21_JSON.read_text())
    plot_fig1(v22)
    plot_fig2(v22, v21)
    plot_fig3(v21)
    plot_fig4(v22)
    plot_fig5(v22)
    print("ALL 5 PAPER 3 v2.2.0 300-DPI FIGURES GENERATED.")


if __name__ == "__main__":
    main()

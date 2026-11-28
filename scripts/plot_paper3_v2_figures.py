"""Generate the 4 Focused, Publication-Grade 300-DPI Figures for Paper 3 v2 (`R3, R4, R5, R11, R12, R13, R16, R22, R24`).

Every figure uses consistent denominators, explicit error bars (95% patient-cluster bootstrap or exact Clopper-Pearson CIs),
neutral comparator names (`Weighted-Pilot CZ`, `Global-CMI + Fluid`), and zero overlapping labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REBUILD_ROOT = Path(__file__).resolve().parents[1]
RES_JSON = REBUILD_ROOT / "results" / "paper3_v2_remediation_results.json"
AUDIT_JSON = (
    REBUILD_ROOT / "results" / "cohort_provenance_and_flow_audit.json"
    if (REBUILD_ROOT / "results" / "cohort_provenance_and_flow_audit.json").exists()
    else REBUILD_ROOT / "data" / "cohort_provenance_and_flow_audit.json"
)
MAN_DIR = REBUILD_ROOT / "manuscript"
FIG_DIR = REBUILD_ROOT / "results"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.0,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_UPCA = "#0B5394"
C_GLOB = "#D95F02"
C_WCZ = "#7570B3"
C_IPW = "#E7298A"
C_S1 = "#666666"
C_ALL = "#999999"
C_GOOD = "#1B9E77"

TASK_ORDER = [
    ("mcmed_critical_icu_death", "Stanford ICU/Death\n(2h, Prev 2.18%)"),
    ("mcmed_hospital_admission", "Stanford Admission\n(2h, Prev 28.8%)"),
    ("mcmed_bounce72_admit_2h", "Stanford 72h Bounce\n(2h, Prev 1.67%)"),
    ("bwh_death_030_ecg_only", "BWH 30d Death\n(ECG, Prev 2.41%)"),
    ("bwh_stent_cabg_010_ecg_only", "BWH 10d Stent/CABG\n(ECG, Prev 1.51%)"),
    ("ntuh_pre_arrest_sca_pi002", "NTUH Pre-Arrest SCA\n(CC Adj. Prev 2.0%)"),
]


def _save_both(fig: plt.Figure, fname: str) -> None:
    MAN_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAN_DIR / fname, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 300-DPI figure -> {MAN_DIR / fname}")


def plot_fig1_envelope_and_regime(res: dict) -> None:
    e5 = res["synthetic"]["experiment_e5_1000_reps"]
    grid = res["synthetic"]["operating_envelope_4d_grid"]["cells"]
    pt = res["clinical"]["primary_tasks"]

    fig = plt.figure(figsize=(14.2, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.08], hspace=0.42, wspace=0.35)

    # Panel A: 1,000-rep Experiment E5 Selection & FWER Rates with Exact Clopper-Pearson 95% CIs
    ax_a = fig.add_subplot(gs[0, 0])
    cats = ["Identifiable\nRetained", "Unidentifiable\nRetained", "Null Noise\nFWER (<=5%)"]
    r_stga = [e5["stage_a_pick_identifiable_rate"] * 100, e5["stage_a_pick_unidentifiable_rate"] * 100, e5["stage_a_null_fwer"] * 100]
    r_upca = [e5["upca_retain_identifiable_rate"] * 100, e5["upca_retain_unidentifiable_rate"] * 100, e5["upca_null_fwer"] * 100]
    ci_stga = [e5["stage_a_pick_identifiable_ci95"], e5["stage_a_pick_unidentifiable_ci95"], e5["stage_a_null_fwer_ci95_clopper_pearson"]]
    ci_upca = [e5["upca_retain_identifiable_ci95_clopper_pearson"], e5["upca_retain_unidentifiable_ci95_clopper_pearson"], e5["upca_null_fwer_ci95_clopper_pearson"]]

    x = np.arange(len(cats))
    w = 0.34
    err_a = np.array([[max(0, r_stga[i] - ci_stga[i][0] * 100) for i in range(3)], [max(0, ci_stga[i][1] * 100 - r_stga[i]) for i in range(3)]])
    err_u = np.array([[max(0, r_upca[i] - ci_upca[i][0] * 100) for i in range(3)], [max(0, ci_upca[i][1] * 100 - r_upca[i]) for i in range(3)]])
    ax_a.bar(x - w / 2, r_stga, w, yerr=err_a, capsize=3, color=C_WCZ, alpha=0.88, label="Offset Stage A Only")
    ax_a.bar(x + w / 2, r_upca, w, yerr=err_u, capsize=3, color=C_UPCA, alpha=0.92, label="UPCA-Fluid (Stage A+B)")
    ax_a.axhline(5.0, color="#B22222", linestyle="--", linewidth=1.0, label="Nominal alpha = 5%")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cats)
    ax_a.set_ylabel("Selection Rate (%) [95% CP CI]")
    ax_a.set_ylim(0, 132)
    ax_a.set_title("A. E5 Channel Screening (1,000 Replicates)")
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.5)

    # Panel B: 1,000-rep Experiment E5 Paired Net Benefit Difference at Tight (B=1) vs Full (B=2) Budget
    ax_b = fig.add_subplot(gs[0, 1])
    b_labels = ["Stage-1\n(0% Query)", "Stage A\n(B = 1.0)", "UPCA-Fluid\n(B = 1.0)", "Stage A\n(B = 2.0)", "UPCA-Fluid\n(B = 2.0)"]
    b_vals = [e5["nb_stage1_mean"], e5["nb_stage_a_b1_mean"], e5["nb_upca_b1_mean"], e5["nb_stage_a_b2_mean"], e5["nb_upca_b2_mean"]]
    b_cols = [C_S1, C_WCZ, C_UPCA, C_WCZ, C_UPCA]
    se_b1 = e5["paired_delta_nb_b1_se"]
    b_errs = [0.0, se_b1 * 1.96, se_b1 * 1.96, abs(e5["paired_delta_nb_b2_ci95"][1] - e5["paired_delta_nb_b2_mean"]), abs(e5["paired_delta_nb_b2_ci95"][1] - e5["paired_delta_nb_b2_mean"])]
    xb = np.arange(len(b_labels))
    bars = ax_b.bar(xb, b_vals, width=0.56, yerr=b_errs, capsize=3.5, color=b_cols, alpha=0.90)
    ax_b.axhline(0.0, color="black", linewidth=0.8)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(b_labels, fontsize=7.8)
    ax_b.set_ylabel("Deployed Net Benefit (/1,000)")
    ax_b.set_title(f"B. E5 Deployed Utility (B=1 dNB = {e5['paired_delta_nb_b1_mean']:+.2f})")
    for bar, val, err in zip(bars, b_vals, b_errs):
        ypos = (val + err + 0.08) if val >= 0 else (val - err - 0.28)
        ax_b.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:+.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ymin_b, ymax_b = min(b_vals) - 0.6, max(b_vals) + 0.85
    ax_b.set_ylim(ymin_b, ymax_b)

    # Panel C: Primary 500-Pilot Two-Phase Discovery Concordance Across rho_pilot in {0.15, 0.20, 0.30}
    ax_c = fig.add_subplot(gs[0, 2])
    eff = res["clinical"]["two_phase_efficiency_r15"]["rho_sensitivity_repeated_pilots"]
    rhos_lbl = ["rho=0.15\n(200 pilots)", "rho=0.20 Primary\n(500 pilots)", "rho=0.30\n(200 pilots)"]
    rho_keys = ["rho_0.15", "rho_0.20", "rho_0.30"]
    xc = np.arange(len(rho_keys))
    wc = 0.26
    for off_c, mk, mlbl, mcol in [
        (-wc, "weighted_005", "Weighted-Pilot CZ (0.05)", C_WCZ),
        (0.0, "ipw_wht", "IPW-Pilot CZ (w_HT)", C_IPW),
        (wc, "offset_bc_gbdt", "UPCA Offset (Breslow-Cain)", C_UPCA),
    ]:
        rates = [eff[rk]["top1_concordance"][mk]["rate"] * 100.0 for rk in rho_keys]
        cis = [eff[rk]["top1_concordance"][mk]["ci95_clopper_pearson"] for rk in rho_keys]
        yerr_c = np.array([
            [max(0.0, rates[i] - cis[i][0] * 100.0) for i in range(3)],
            [max(0.0, cis[i][1] * 100.0 - rates[i]) for i in range(3)],
        ])
        ax_c.bar(xc + off_c, rates, wc, yerr=yerr_c, capsize=3, color=mcol, alpha=0.90, label=mlbl)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(rhos_lbl, fontsize=7.8)
    ax_c.set_ylim(40, 114)
    ax_c.set_ylabel("Top-1 Oracle Concordance (%) [95% CP]")
    ax_c.set_title("C. Repeated-Pilot Concordance (McNemar p=4.7e-12)")
    ax_c.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=7.0)

    # Bottom Row (Panels D1, D2, D3): 4D Operating-Envelope Heatmaps across rho in {0.10, 0.22, 0.40} (pi_CZ = 0.40)
    sigmas = [1.5, 2.5, 3.5]
    cratios = [0.15, 0.30, 0.60, 0.90]
    cell_lookup = {(c["rho"], c["sigma"], c["cquery_over_g"], c["pi_cz"]): c for c in grid}

    for idx_r, rho in enumerate([0.10, 0.22, 0.40]):
        ax_h = fig.add_subplot(gs[1, idx_r])
        mat = np.zeros((len(sigmas), len(cratios)), dtype=float)
        for i_s, sig in enumerate(sigmas):
            for j_c, cr in enumerate(cratios):
                c_obj = cell_lookup[(rho, sig, cr, 0.40)]
                mat[i_s, j_c] = c_obj["emp_2d_minus_12bin_gain"]
        im = ax_h.imshow(mat, cmap="RdBu_r", vmin=-1.5, vmax=3.5, aspect="auto")
        for i_s, sig in enumerate(sigmas):
            for j_c, cr in enumerate(cratios):
                c_obj = cell_lookup[(rho, sig, cr, 0.40)]
                val = mat[i_s, j_c]
                ret = int(round(c_obj["retain_identifiable_rate"] * 100))
                txt_col = "white" if abs(val) > 1.8 else "black"
                ax_h.text(j_c, i_s, f"{val:+.2f}\n(ID:{ret}%)", ha="center", va="center", fontsize=7.6, color=txt_col, fontweight="bold")
        ax_h.set_xticks(range(len(cratios)))
        ax_h.set_xticklabels([f"{cr:.2f}" for cr in cratios])
        ax_h.set_yticks(range(len(sigmas)))
        ax_h.set_yticklabels([f"sigma={s:.1f}" for s in sigmas])
        ax_h.set_xlabel("Relative Query Cost c_query / g(sigma)")
        if idx_r == 0:
            ax_h.set_ylabel("Active Signal Dispersion sigma")
        ax_h.set_title(f"D{idx_r+1}. Envelope (rho={rho:.2f}, pi_CZ=0.40)", fontsize=9.4)
        if idx_r == 2:
            cbar = fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            cbar.set_label("2D - 12bin Gain (/1k)", fontsize=8.0)

    _save_both(fig, "fig1_operating_envelope_and_regime.png")


def plot_fig2_dca_and_comparators(res: dict) -> None:
    pt = res["clinical"]["primary_tasks"]
    fig = plt.figure(figsize=(14.0, 9.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.12], hspace=0.42, wspace=0.28)

    # Top 2 rows: 6 Vickers Decision Curves NB(tau)
    for idx, (tkey, ttitle) in enumerate(TASK_ORDER):
        r_i, c_i = divmod(idx, 3)
        ax = fig.add_subplot(gs[r_i, c_i])
        t_obj = pt[tkey]
        arms = t_obj["arms"]
        tau_star = t_obj["contract"]["tau_star"]

        dca_u = arms["upca_fluid"]["decision_curve"]
        dca_g = arms["global_cmi_fluid"]["decision_curve"]
        dca_w = arms["weighted_pilot_cz_fluid"]["decision_curve"]
        dca_s1 = arms["stage1"]["decision_curve"]
        taus_pct = [t * 100.0 for t in dca_u["taus"]]

        ax.plot(taus_pct, dca_u["nb_treat_none"], color="black", linestyle=":", linewidth=1.0, label="Treat None")
        ax.plot(taus_pct, dca_u["nb_treat_all"], color=C_ALL, linestyle="--", linewidth=1.2, label="Treat All")
        ax.plot(taus_pct, dca_s1["nb_model"], color=C_S1, linestyle="-.", linewidth=1.3, label="Stage-1 (0% Query)")
        ax.plot(taus_pct, dca_g["nb_model"], color=C_GLOB, linewidth=1.5, alpha=0.88, label="Global-CMI + Fluid")
        ax.plot(taus_pct, dca_w["nb_model"], color=C_WCZ, linestyle="--", linewidth=1.5, alpha=0.88, label="Weighted-Pilot CZ + Fluid")
        ax.plot(taus_pct, dca_u["nb_model"], color=C_UPCA, linewidth=2.1, label="UPCA-Fluid")
        ax.axvline(tau_star * 100.0, color="#B22222", linestyle=":", linewidth=1.1)

        ymax = max(max(dca_u["nb_model"]), max(dca_s1["nb_model"]), 2.0) * 1.18
        ymin = -max(3.0, 0.22 * ymax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Decision Threshold tau (%)")
        if c_i == 0:
            ax.set_ylabel("Net Benefit (/1,000)")
        ax.set_title(f"{chr(65+idx)}. {ttitle.replace(chr(10), ' ')}", fontsize=9.2)
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.0)

    # Bottom Left (cols 0..1): R=10 Primary Deployable Bagged VOI & Covert-AFA 10k Patient-Cluster Bootstrap Differences
    ax_bl = fig.add_subplot(gs[2, :2])
    short_names = ["Stanford\nICU/Death", "Stanford\nAdmission", "Stanford\n72h Bounce", "BWH 30d\nDeath", "BWH 10d\nStent/CABG", "NTUH\nPre-Arrest"]
    x = np.arange(len(TASK_ORDER))
    w = 0.21

    series_bl = [
        (-1.5 * w, "r10", "bagged_upca_vs_stage1", "R=10 Bagged UPCA vs. Stage-1", C_UPCA),
        (-0.5 * w, "r10", "bagged_upca_vs_weighted_cz", "R=10 Bagged UPCA vs. Weighted-CZ", C_WCZ),
        (0.5 * w, "r10", "bagged_upca_vs_global_cmi_gbdt", "R=10 Bagged UPCA vs. CMI+GBDT", C_GLOB),
        (1.5 * w, "can", "upca_vs_covert_afa", "UPCA vs. Covert-AFA (Greedy CMI)", C_GOOD),
    ]
    for offset, src_t, bkey, lbl, col in series_bl:
        if src_t == "r10":
            b_objs = [pt[k]["r10_estimator_triad"]["bootstraps_10k_patient_cluster"][bkey] for k, _ in TASK_ORDER]
        else:
            b_objs = [pt[k]["bootstraps_10k_patient_cluster"][bkey] for k, _ in TASK_ORDER]
        means = [b["diff_mean"] for b in b_objs]
        lo = [b["ci95_low"] for b in b_objs]
        hi = [b["ci95_high"] for b in b_objs]
        yerr = np.array([[max(0.0, means[i] - lo[i]) for i in range(6)], [max(0.0, hi[i] - means[i]) for i in range(6)]])
        ax_bl.bar(x + offset, means, w, yerr=yerr, capsize=2.5, color=col, alpha=0.90, label=lbl)

    ax_bl.axhline(0.0, color="black", linewidth=0.8)
    ax_bl.set_xticks(x)
    ax_bl.set_xticklabels(short_names)
    ax_bl.set_ylabel("Paired dNB (/1,000) [10k Patient-Cluster 95% CI]")
    ax_bl.set_title("G. Deployable Bagged VOI (R=10) & Covert-AFA Paired Differences")
    ax_bl.legend(loc="upper right", frameon=False, ncol=2, fontsize=7.2)

    # Bottom Right (col 2): Both AUPRC_full and AUPRC_routed Side-by-Side (`R14, R22`)
    ax_br = fig.add_subplot(gs[2, 2])
    ap_s1 = [pt[k]["arms"]["stage1"]["auprc_full"] for k, _ in TASK_ORDER]
    ap_rt = [pt[k]["arms"]["upca_fluid"]["auprc_routed"] for k, _ in TASK_ORDER]
    ap_fl = [pt[k]["arms"]["upca_fluid"]["auprc_full"] for k, _ in TASK_ORDER]
    ax_br.bar(x - w, ap_s1, w, color=C_S1, alpha=0.85, label="Stage-1 AUPRC")
    ax_br.bar(x, ap_rt, w, color="#5FA8D3", alpha=0.90, label="UPCA AUPRC_routed")
    ax_br.bar(x + w, ap_fl, w, color=C_UPCA, alpha=0.95, label="UPCA AUPRC_full")
    ax_br.set_xticks(x)
    ax_br.set_xticklabels(["ICU", "Adm", "B72", "D30", "Rev", "SCA"], fontsize=8.0)
    ax_br.set_ylabel("Precision-Recall Area (AUPRC)")
    ax_br.set_title("H. AUPRC_full vs. AUPRC_routed (Prop. 1)")
    ax_br.legend(loc="upper right", frameon=False, fontsize=7.4)

    _save_both(fig, "fig2_clinical_net_benefit_and_dca.png")


def plot_fig3_calibration_curves(res: dict) -> None:
    pt = res["clinical"]["primary_tasks"]
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
        ax.axvline(tau_star, color="#B22222", linestyle=":", linewidth=1.1, label=f"tau* = {tau_star:.3f}")

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Predicted Probability")
        if idx % 3 == 0:
            ax.set_ylabel("Observed Event Frequency")
        ax.set_title(f"{chr(65+idx)}. {ttitle.replace(chr(10), ' ')}", fontsize=9.3)

        box_txt = (
            f"Intercept = {cal_u['cal_intercept']:+.3f}\n"
            f"Slope     = {cal_u['cal_slope']:.3f}\n"
            f"ICI       = {cal_u['ici']:.4f}"
        )
        ax.text(
            0.05, 0.94, box_txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.8, family="monospace",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F7F9FC", "edgecolor": "#CCCCCC"},
        )
        if idx == 0:
            ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=7.2)

    _save_both(fig, "fig3_calibration_curves.png")


def plot_fig4_skipped_disagreement_and_subgroups(res: dict, audit: dict) -> None:
    pt = res["clinical"]["primary_tasks"]
    sub = res["clinical"]["subgroup_and_missingness_audits"]
    sens = res["clinical"]["sensitivity_sweeps_r10"]

    fig = plt.figure(figsize=(14.4, 8.8))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.46)

    # Panel A: Directional Patient-Cluster Skipped-Case Action-Disagreement 95% Upper Bounds (`R5, R22`)
    ax_a = fig.add_subplot(gs[0, 0])
    short_t = ["Stanford\nICU", "Stanford\nAdmit", "Stanford\nBounce72", "BWH 30d\nDeath", "BWH 10d\nRev.", "NTUH\nPre-Arrest"]
    x = np.arange(len(TASK_ORDER))
    w = 0.26

    ub_tot = [pt[k]["arms"]["upca_fluid"]["skip_disagreement"]["flip_ub95_patient_cluster"] * 100 for k, _ in TASK_ORDER]
    ub_n2p = [pt[k]["arms"]["upca_fluid"]["skip_disagreement"]["neg_to_pos_ub95_cluster"] * 100 for k, _ in TASK_ORDER]
    ub_p2n = [pt[k]["arms"]["upca_fluid"]["skip_disagreement"]["pos_to_neg_ub95_cluster"] * 100 for k, _ in TASK_ORDER]
    emp_tot = [pt[k]["arms"]["upca_fluid"]["skip_disagreement"]["flip_rate"] * 100 for k, _ in TASK_ORDER]

    ax_a.bar(x - w, ub_tot, w, color=C_UPCA, alpha=0.90, label="Total Disagreement 95% Cluster UB")
    ax_a.bar(x, ub_n2p, w, color=C_GLOB, alpha=0.88, label="Neg->Pos (Missed Escalation) 95% UB")
    ax_a.bar(x + w, ub_p2n, w, color=C_WCZ, alpha=0.88, label="Pos->Neg (Missed De-esc.) 95% UB")
    ax_a.scatter(x - w, emp_tot, color="white", edgecolor="black", s=22, zorder=5, label="Empirical Flip Rate")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short_t, fontsize=8.0)
    ax_a.set_ylabel("Skipped-Case Disagreement (% of Q=0)")
    ax_a.set_title("A. Directional Skipped-Case Action-Disagreement Bounds")
    ax_a.legend(loc="upper right", frameon=False, fontsize=7.3)

    # Panel B: Query-Cost Sensitivity Sweep c_query = gamma * C_FN on Stanford ICU/Death (`R10`)
    ax_b = fig.add_subplot(gs[0, 1])
    cs_icu = sens["stanford_icu_death"]["c_query_sweep"]
    cqs = [r["c_query"] for r in cs_icu]
    nb_u_sw = [r["nb_upca"] for r in cs_icu]
    nb_100_sw = [r["nb_100pct"] for r in cs_icu]
    nb_s1_sw = [r["nb_stage1"] for r in cs_icu]
    q_sw = [r["query_rate"] * 100 for r in cs_icu]

    ax_b.plot(cqs, nb_u_sw, marker="o", color=C_UPCA, linewidth=2.1, label="UPCA-Fluid (Selective)")
    ax_b.plot(cqs, nb_100_sw, marker="s", color=C_GLOB, linestyle="--", linewidth=1.6, label="100% Unselective Query")
    ax_b.plot(cqs, nb_s1_sw, color=C_S1, linestyle=":", linewidth=1.4, label="Stage-1 (0% Query)")
    ax_b_r = ax_b.twinx()
    ax_b_r.spines["top"].set_visible(False)
    ax_b_r.plot(cqs, q_sw, marker="^", color=C_GOOD, linestyle="-.", linewidth=1.4, label="Query Rate q (%)")
    ax_b.set_xlabel("Per-Encounter Query Cost c_query (C_FP = 1, C_FN = 19)")
    ax_b.set_ylabel("Net Benefit (/1,000)", color=C_UPCA)
    ax_b_r.set_ylabel("Query Rate (%)", color=C_GOOD)
    ax_b_r.set_ylim(0, 105)
    ax_b.set_title("B. Query-Cost Sweep on Stanford ICU/Death")
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_b_r.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc="center right", frameon=True, framealpha=0.9, fontsize=7.3)

    # Panel C: BWH 30d Mortality Subgroup Delta-NB with 10k Patient-Cluster 95% CIs (`R11, R24`)
    ax_c = fig.add_subplot(gs[1, 0])
    bwh_agi = sub["bwh_death30_by_agi_quartile"]["strata"]
    bwh_race = sub["bwh_death30_by_race"]["strata"]
    c_items = [
        ("AGI Q1 (Low)", bwh_agi.get("Q1_Lowest_Income")),
        ("AGI Q2", bwh_agi.get("Q2_LowMid")),
        ("AGI Q3", bwh_agi.get("Q3_HighMid")),
        ("AGI Q4 (High)", bwh_agi.get("Q4_Highest_Income")),
        ("Race: Black", bwh_race.get("Black")),
        ("Race: White", bwh_race.get("White")),
        ("Race: Hisp.", bwh_race.get("Hispanic")),
    ]
    c_items = [(lbl, d) for lbl, d in c_items if d is not None]
    yc = np.arange(len(c_items))
    c_means = [d["delta_nb"] for _, d in c_items]
    c_err = np.array([
        [max(0.0, d["delta_nb"] - d["delta_nb_ci95"][0]) for _, d in c_items],
        [max(0.0, d["delta_nb_ci95"][1] - d["delta_nb"]) for _, d in c_items],
    ])
    ax_c.barh(yc, c_means, xerr=c_err, capsize=3, color=C_UPCA, alpha=0.88, height=0.60)
    ax_c.axvline(0.0, color="black", linewidth=0.8)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([f"{lbl} (N={d['n_encounters']}, q={d['query_rate']*100:.0f}%)" for lbl, d in c_items], fontsize=7.8)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("dNB vs. Stage-1 (/1,000) [10k Patient-Cluster 95% CI]")
    p_het_bwh = sub["bwh_death30_by_agi_quartile"]["heterogeneity"]["p_heterogeneity"]
    ax_c.set_title(f"C. BWH 30d Mortality Subgroups (AGI Q p_het={p_het_bwh:.2f})")

    # Panel D: Stanford Admission / ICU Subgroups + Prior-CDW EHR Payer Coverage (`R11, R19`)
    ax_d = fig.add_subplot(gs[1, 1])
    mc_pay = sub["mcmed_admission_by_payor"]["strata"]
    mc_rc = sub["mcmed_admission_by_race"]["strata"]
    cdw_pay = audit["mcmed_stanford"]["cdw_prior_ehr_coverage_by_payor"]
    d_items = [
        ("Medicaid", mc_pay.get("Medicaid"), cdw_pay["Medicaid"]["has_prior_cdw_meds_rate"]),
        ("Medicare", mc_pay.get("Medicare"), cdw_pay["Medicare"]["has_prior_cdw_meds_rate"]),
        ("Unspec./Comm.", mc_pay.get("Unspecified_or_Commercial"), cdw_pay["Unspecified_or_Commercial"]["has_prior_cdw_meds_rate"]),
        ("Black", mc_rc.get("Black or African American"), None),
        ("Asian", mc_rc.get("Asian"), None),
        ("White", mc_rc.get("White"), None),
    ]
    d_items = [(lbl, d, cov) for lbl, d, cov in d_items if d is not None]
    yd = np.arange(len(d_items))
    d_means = [d["delta_nb"] for _, d, _ in d_items]
    d_err = np.array([
        [max(0.0, d["delta_nb"] - d["delta_nb_ci95"][0]) for _, d, _ in d_items],
        [max(0.0, d["delta_nb_ci95"][1] - d["delta_nb"]) for _, d, _ in d_items],
    ])
    ax_d.barh(yd, d_means, xerr=d_err, capsize=3, color=C_GOOD, alpha=0.88, height=0.60)
    ax_d.axvline(0.0, color="black", linewidth=0.8)
    ax_d.set_yticks(yd)
    ylabels_d = []
    for lbl, d, cov in d_items:
        cov_str = f", CDW-Med={cov*100:.0f}%" if cov is not None else ""
        ylabels_d.append(f"{lbl} (N={d['n_encounters']}{cov_str})")
    ax_d.set_yticklabels(ylabels_d, fontsize=7.6)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("dNB vs. Stage-1 (/1,000) [10k Patient-Cluster 95% CI]")
    p_het_mc = sub["mcmed_admission_by_payor"]["heterogeneity"]["p_heterogeneity"]
    p_het_str = "p_het < 0.0001" if p_het_mc < 1e-4 else f"p_het={p_het_mc:.3f}"
    ax_d.set_title(f"D. Stanford Admission Subgroups & CDW Coverage ({p_het_str})")

    _save_both(fig, "fig4_skipped_disagreement_and_subgroups.png")


def main() -> None:
    res = json.loads(RES_JSON.read_text())
    audit = json.loads(AUDIT_JSON.read_text())
    plot_fig1_envelope_and_regime(res)
    plot_fig2_dca_and_comparators(res)
    plot_fig3_calibration_curves(res)
    plot_fig4_skipped_disagreement_and_subgroups(res, audit)
    print("\nALL 4 FOCUSED 300-DPI FIGURES GENERATED CLEANLY.")


if __name__ == "__main__":
    main()

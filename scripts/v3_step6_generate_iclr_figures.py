import os
import glob
import json
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Native ICLR \textwidth target (~5.50 in canvas; exact tight-cropped PDF bounding box verified at build time),
# Type-42 TrueType fonts (zero Type-3 fonts) & legible 6.4-8.2pt vector typography.
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.6,
    "axes.titlesize": 8.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.4,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.pad": 1.8,
    "ytick.major.pad": 1.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.24,
    "grid.linestyle": "--",
    "grid.linewidth": 0.45,
})

COLORS = {
    "cedar": "#0072B2",        # Deep Blue (CEDAR Split-Op)
    "cedar_alt": "#56B4E9",    # Sky Blue
    "strong_no_nsd": "#332288",# Indigo (Strong Combined w/o NSD)
    "supcon": "#009E73",       # Bluish Green
    "aug_erm": "#E69F00",      # Orange
    "uned_erm": "#CC79A7",     # Reddish Purple
    "desklib": "#D55E00",      # Vermilion
    "bino": "#7F7F7F",         # Neutral Gray
    "fastd": "#999933",        # Olive
    "radar": "#882255",        # Wine
    "hc3": "#AA4499",          # Violet
}


def r_pct(val: float, digits: int = 1) -> str:
    q = Decimal("10") ** (-digits)
    return str((Decimal(str(val)) * Decimal("100")).quantize(q, rounding=ROUND_HALF_UP))


def save_both(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    pdf_p = os.path.join(out_dir, f"{stem}.pdf")
    png_p = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(pdf_p, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_p, bbox_inches="tight", pad_inches=0.02, dpi=260)
    plt.close(fig)
    print(f"Saved {pdf_p} & {png_p}")


def plot_fig1(summary: dict, per_seed: dict, out_dir: str):
    pre = summary["pretrained_and_probes"]
    agg = summary["trained_multi_seed_agg"]
    bridge = summary["attribution_bridge_2x2"]

    def seed_id_e2_e0(prefix):
        vals = [v["test_id_recoverability"]["E2"]["fpr_at_tau_e0_5"] for k, v in per_seed.items() if k.startswith(prefix + "_seed")]
        return float(np.mean(vals)), float(np.std(vals, ddof=0))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.5, 1.32), gridspec_kw={"width_ratios": [1.08, 0.98, 1.04]})

    # (a) Edited Human FPR Inflation at Unedited Calibration Threshold tau_E0
    methods = [
        ("Fast-D.", pre["fast_detectgpt_qwen3b"]["test_id_recoverability"]["E2"]["fpr_at_tau_e0_5"], 0.0),
        ("Binoc.", pre["binoculars_qwen3b"]["test_id_recoverability"]["E2"]["fpr_at_tau_e0_5"], 0.0),
        ("HC3", pre["hc3_roberta_detector"]["test_id_recoverability"]["E2"]["fpr_at_tau_e0_5"], 0.0),
        ("Desklib", pre["desklib_raid_v1"]["test_id_recoverability"]["E2"]["fpr_at_tau_e0_5"], 0.0),
        ("Uned.", *seed_id_e2_e0("mb_v3_unedited_erm")),
        ("AugERM", *seed_id_e2_e0("mb_v3_augmented_erm")),
        ("NoNSD", *seed_id_e2_e0("mb_v3_strong_combined_no_nsd")),
        ("CEDAR", *seed_id_e2_e0("mb_v3_cedar_split_op")),
    ]
    x1 = np.arange(len(methods))
    vals1 = [100.0 * m[1] for m in methods]
    errs1 = [100.0 * m[2] for m in methods]
    cols1 = [COLORS["fastd"], COLORS["bino"], COLORS["hc3"], COLORS["desklib"], COLORS["uned_erm"], COLORS["aug_erm"], COLORS["strong_no_nsd"], COLORS["cedar"]]
    ax1.bar(x1, vals1, width=0.62, yerr=errs1, capsize=1.3, error_kw={"lw": 0.55}, color=cols1, edgecolor="black", linewidth=0.35)
    ax1.axhline(5.0, color="#D55E00", linestyle="--", linewidth=0.85, label="5% Target")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([m[0] for m in methods], rotation=32, ha="right")
    ax1.set_ylabel("Edited Human FPR (%)")
    ax1.set_title("(a) $E_2$ FPR at $\\tau_{E_0}$")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, borderpad=0.2)

    # (b) 2x2 Register Symmetry Bridge: Unidirectional (v1) vs. Bidirectional (v2)
    b_labels = ["v1\nAug", "v1\nCEDAR", "v2\nAug", "v2\nCEDAR"]
    b_keys = ["bridge_v1data_aug_erm", "bridge_v1data_v1cedar", "mb_v3_augmented_erm", "mb_v3_cedar_split_op"]
    xb = np.arange(len(b_keys))
    wb = 0.36
    ms_v = [100.0 * bridge[k]["mixset_pure_auroc"]["mean"] for k in b_keys]
    ms_e = [100.0 * bridge[k]["mixset_pure_auroc"]["std"] for k in b_keys]
    cf_v = [100.0 * bridge[k]["confirm_colloquial_auroc"]["mean"] for k in b_keys]
    cf_e = [100.0 * bridge[k]["confirm_colloquial_auroc"]["std"] for k in b_keys]
    ax2.bar(xb - wb / 2, ms_v, width=wb, yerr=ms_e, capsize=1.3, error_kw={"lw": 0.55}, color="#FC8D62", edgecolor="black", linewidth=0.35, label="MixSet Pure")
    ax2.bar(xb + wb / 2, cf_v, width=wb, yerr=cf_e, capsize=1.3, error_kw={"lw": 0.55}, color=COLORS["cedar"], edgecolor="black", linewidth=0.35, label="Confirm Colloq.")
    ax2.axhline(50.0, color="#444444", linestyle=":", linewidth=0.8)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(b_labels)
    ax2.set_ylim(35, 104)
    ax2.set_ylabel("AUROC (%)")
    ax2.set_title("(b) Register Bridge")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9, borderpad=0.18, handlelength=1.0)

    # (c) Frontier: Pristine test_confirm WG Acc vs MixSet Pure AUROC
    pts = [
        ("Binoc.", pre["binoculars_qwen3b"]["external_mixset"]["pure_origin_auroc"], pre["binoculars_qwen3b"]["test_confirm"]["worst_group_acc_E0_E3"], 0.0, 0.0, COLORS["bino"], "s"),
        ("Desklib", pre["desklib_raid_v1"]["external_mixset"]["pure_origin_auroc"], pre["desklib_raid_v1"]["test_confirm"]["worst_group_acc_E0_E3"], 0.0, 0.0, COLORS["desklib"], "^"),
        ("Uned.", agg["mb_v3_unedited_erm"]["mixset_pure_auroc"]["mean"], agg["mb_v3_unedited_erm"]["confirm_wg_E0_E3"]["mean"], agg["mb_v3_unedited_erm"]["mixset_pure_auroc"]["std"], agg["mb_v3_unedited_erm"]["confirm_wg_E0_E3"]["std"], COLORS["uned_erm"], "P"),
        ("AugERM", agg["mb_v3_augmented_erm"]["mixset_pure_auroc"]["mean"], agg["mb_v3_augmented_erm"]["confirm_wg_E0_E3"]["mean"], agg["mb_v3_augmented_erm"]["mixset_pure_auroc"]["std"], agg["mb_v3_augmented_erm"]["confirm_wg_E0_E3"]["std"], COLORS["aug_erm"], "o"),
        ("SupCon", agg["mb_v3_supcon_orth_disentangle"]["mixset_pure_auroc"]["mean"], agg["mb_v3_supcon_orth_disentangle"]["confirm_wg_E0_E3"]["mean"], agg["mb_v3_supcon_orth_disentangle"]["mixset_pure_auroc"]["std"], agg["mb_v3_supcon_orth_disentangle"]["confirm_wg_E0_E3"]["std"], COLORS["supcon"], "X"),
        ("NoNSD", agg["mb_v3_strong_combined_no_nsd"]["mixset_pure_auroc"]["mean"], agg["mb_v3_strong_combined_no_nsd"]["confirm_wg_E0_E3"]["mean"], agg["mb_v3_strong_combined_no_nsd"]["mixset_pure_auroc"]["std"], agg["mb_v3_strong_combined_no_nsd"]["confirm_wg_E0_E3"]["std"], COLORS["strong_no_nsd"], "D"),
        ("CEDAR", agg["mb_v3_cedar_split_op"]["mixset_pure_auroc"]["mean"], agg["mb_v3_cedar_split_op"]["confirm_wg_E0_E3"]["mean"], agg["mb_v3_cedar_split_op"]["mixset_pure_auroc"]["std"], agg["mb_v3_cedar_split_op"]["confirm_wg_E0_E3"]["std"], COLORS["cedar"], "*"),
    ]
    for lbl, xv, yv, xe, ye, col, mk in pts:
        sz = 68 if lbl == "CEDAR" else 26
        ax3.errorbar(100 * xv, 100 * yv, xerr=100 * xe, yerr=100 * ye, fmt="none", ecolor=col, alpha=0.65, capsize=1.3, elinewidth=0.65)
        ax3.scatter([100 * xv], [100 * yv], s=sz, color=col, marker=mk, edgecolors="black", linewidths=0.45, zorder=4, label=lbl)
    ax3.set_xlabel("MixSet Pure AUC (%)")
    ax3.set_ylabel("Confirm WG Acc (%)")
    ax3.set_title("(c) Untouched Frontier")
    ax3.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.88, fontsize=5.9, columnspacing=0.35, handletextpad=0.18, borderpad=0.18, labelspacing=0.12)

    fig.tight_layout(pad=0.30, w_pad=0.75)
    save_both(fig, out_dir, "fig1_editorial_paradox_overview")


def plot_fig2(summary: dict, root_dir: str, out_dir: str):
    pre = summary["pretrained_and_probes"]
    agg = summary["trained_multi_seed_agg"]
    lvs = ["E0", "E1", "E2", "E3", "E4"]
    lv_labels = ["$E_0$", "$E_1$", "$E_2$", "$E_3$", "$E_4$"]
    x = np.arange(len(lvs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 1.22))

    series_a = [
        ("Binoc.", [pre["binoculars_qwen3b"]["test_id_recoverability"][k]["bal_acc"] for k in lvs], [0] * 5, COLORS["bino"], "--", "s"),
        ("Desklib", [pre["desklib_raid_v1"]["test_id_recoverability"][k]["bal_acc"] for k in lvs], [0] * 5, COLORS["desklib"], "--", "^"),
        ("Uned. ERM", [agg["mb_v3_unedited_erm"]["id_recoverability"][k]["bal_acc"]["mean"] for k in lvs],
         [agg["mb_v3_unedited_erm"]["id_recoverability"][k]["bal_acc"]["std"] for k in lvs], COLORS["uned_erm"], "-.", "P"),
        ("Aug. ERM", [agg["mb_v3_augmented_erm"]["id_recoverability"][k]["bal_acc"]["mean"] for k in lvs],
         [agg["mb_v3_augmented_erm"]["id_recoverability"][k]["bal_acc"]["std"] for k in lvs], COLORS["aug_erm"], "-", "o"),
        ("Strong w/o NSD", [agg["mb_v3_strong_combined_no_nsd"]["id_recoverability"][k]["bal_acc"]["mean"] for k in lvs],
         [agg["mb_v3_strong_combined_no_nsd"]["id_recoverability"][k]["bal_acc"]["std"] for k in lvs], COLORS["strong_no_nsd"], "-", "D"),
        ("CEDAR", [agg["mb_v3_cedar_split_op"]["id_recoverability"][k]["bal_acc"]["mean"] for k in lvs],
         [agg["mb_v3_cedar_split_op"]["id_recoverability"][k]["bal_acc"]["std"] for k in lvs], COLORS["cedar"], "-", "*"),
    ]
    for lbl, ys, es, col, ls, mk in series_a:
        ys_a = 100.0 * np.asarray(ys)
        es_a = 100.0 * np.asarray(es)
        lw = 1.6 if lbl == "CEDAR" else 1.05
        ax1.plot(x, ys_a, label=lbl, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=3.6)
        if np.any(es_a > 0):
            ax1.fill_between(x, ys_a - es_a, ys_a + es_a, color=col, alpha=0.13)

    cedar_e4_auc = agg["mb_v3_cedar_split_op"]["id_recoverability"]["E4"]["auroc"]["mean"]
    ax1.scatter([4.06], [100.0 * cedar_e4_auc], marker="D", s=22, facecolors="white", edgecolors=COLORS["cedar"], linewidths=1.1, zorder=5,
                label=f"$E_4$ AUC ({r_pct(cedar_e4_auc, 1)}%)")
    ax1.axhline(50.0, color="black", linestyle=":", linewidth=0.8)
    ax1.axvspan(3.55, 4.3, color="#EEEEEE", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lv_labels)
    ax1.set_xlabel("Editorial Intervention Level (`test_id`)")
    ax1.set_ylim(44, 101)
    ax1.set_ylabel("Balanced Acc (%)")
    ax1.set_title("(a) Recoverability ($E_0 \\to E_4$)")
    ax1.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.92, fontsize=6.0, columnspacing=0.4, handlelength=1.1, handletextpad=0.22, borderpad=0.18, labelspacing=0.15)

    tid_rows = json.load(open(os.path.join(root_dir, "data", "split_test_id.json")))
    y_tid = np.array([r["origin_label"] for r in tid_rows])
    cov_grid = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])

    def eval_dense_rc(scores, tau, a_hat=None):
        s = np.asarray(scores, dtype=np.float64)
        if np.min(s) >= -1e-6 and np.max(s) <= 1.0 + 1e-6 and 0.0 < tau < 1.0:
            denom = np.where(s > tau, max(1.0 - tau, 1e-6), max(tau, 1e-6))
        else:
            denom = np.where(s > tau, max(float(np.max(s) - tau), 1e-6), max(float(tau - np.min(s)), 1e-6))
        conf = np.abs(s - tau) / denom
        if a_hat is not None and len(a_hat) == len(s):
            conf = conf * (1.0 - 0.5 * np.clip(np.asarray(a_hat, dtype=np.float64), 0.0, 1.0))
        order = np.argsort(-conf)
        errs = ((s > tau).astype(np.int64) != y_tid).astype(np.float64)
        return np.array([100.0 * float(errs[order[:max(1, int(round(0.01 * c * len(y_tid))))]].mean()) for c in cov_grid])

    pre_p = json.load(open(os.path.join(root_dir, "results_v3", "preds_pretrained_and_probes_v3.json")))
    desk_tau = pre["desklib_raid_v1"]["tau_cal_all5"]
    desk_curve = eval_dense_rc(pre_p["desklib_raid_v1"]["test_id"], desk_tau)
    ax2.plot(cov_grid, desk_curve, label="Desklib ($m_\\tau$)", color=COLORS["desklib"], linestyle="--", marker="^", linewidth=1.05, markersize=3.4)

    pred_dir = os.path.join(root_dir, "results_v3", "model_preds_v3")
    aurc_m = r_pct(agg["mb_v3_cedar_split_op"]["rc_margin_aurc"]["mean"], 2)
    aurc_c = r_pct(agg["mb_v3_cedar_split_op"]["rc_intens_aurc"]["mean"], 2)
    for pref, lbl, col, ls, mk, use_int in [
        ("mb_v3_augmented_erm", "Aug. ERM ($m_\\tau$)", COLORS["aug_erm"], "-", "o", False),
        ("mb_v3_strong_combined_no_nsd", "Strong w/o NSD ($m_\\tau$)", COLORS["strong_no_nsd"], "-", "D", False),
        ("mb_v3_cedar_split_op", f"CEDAR ($m_\\tau$: {aurc_m}%)", COLORS["cedar_alt"], "--", "d", False),
        ("mb_v3_cedar_split_op", f"CEDAR ($c_\\tau$: {aurc_c}%)", COLORS["cedar"], "-", "*", True),
    ]:
        curves = []
        fps = sorted(glob.glob(os.path.join(pred_dir, f"{pref}_seed*.json")), key=lambda p: int(p.split("_seed")[-1].split(".")[0]))
        for fp in fps:
            pl = json.load(open(fp))
            ah = pl.get("edit_intensity", {}).get("test_id") if use_int else None
            curves.append(eval_dense_rc(pl["preds"]["test_id"], float(pl["locked_thresholds"]["tau_cal_all5"]), ah))
        arr = np.asarray(curves)
        m_rc, s_rc = arr.mean(axis=0), arr.std(axis=0, ddof=0)
        lw = 1.55 if "CEDAR" in lbl else 1.05
        ax2.plot(cov_grid, m_rc, label=lbl, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=3.5)
        ax2.fill_between(cov_grid, m_rc - s_rc, m_rc + s_rc, color=col, alpha=0.12)

    ax2.set_xlabel("Retained Coverage (%)")
    ax2.set_ylabel("Selective Error (%)")
    ax2.set_title("(b) Risk--Coverage ($20\\%$--$100\\%$)")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=6.0, handlelength=1.15, handletextpad=0.22, borderpad=0.18, labelspacing=0.15)

    fig.tight_layout(pad=0.32, w_pad=0.8)
    save_both(fig, out_dir, "fig2_recoverability_phase_curves")


def plot_fig3(summary: dict, per_seed: dict, out_dir: str):
    pre = summary["pretrained_and_probes"]
    agg = summary["trained_multi_seed_agg"]
    crc = summary["few_shot_human_anchor_crc"]["mb_v3_cedar_split_op"]["benchmarks"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.5, 1.24), gridspec_kw={"width_ratios": [1.0, 0.96, 1.15]})

    # (a) Pristine test_confirm Balanced Accuracy by Stylistic Register
    regs = ["unedited_E0", "formal_edited", "colloquial_humanized"]
    reg_lbls = ["Raw\n$E_0$", "Formal\nEdit", "Colloq.\nHumanize"]
    xr = np.arange(len(regs))
    wr = 0.20
    r_bars = [
        ("Desklib", [100 * pre["desklib_raid_v1"]["test_confirm"]["by_register"][r]["bal_acc_tau_all5"] for r in regs], [0] * 3, COLORS["desklib"]),
        ("AugERM", [100 * agg["mb_v3_augmented_erm"]["confirm_by_register"][r]["bal_acc"]["mean"] for r in regs],
         [100 * agg["mb_v3_augmented_erm"]["confirm_by_register"][r]["bal_acc"]["std"] for r in regs], COLORS["aug_erm"]),
        ("NoNSD", [100 * agg["mb_v3_strong_combined_no_nsd"]["confirm_by_register"][r]["bal_acc"]["mean"] for r in regs],
         [100 * agg["mb_v3_strong_combined_no_nsd"]["confirm_by_register"][r]["bal_acc"]["std"] for r in regs], COLORS["strong_no_nsd"]),
        ("CEDAR", [100 * agg["mb_v3_cedar_split_op"]["confirm_by_register"][r]["bal_acc"]["mean"] for r in regs],
         [100 * agg["mb_v3_cedar_split_op"]["confirm_by_register"][r]["bal_acc"]["std"] for r in regs], COLORS["cedar"]),
    ]
    for off, (lbl, vals, errs, col) in zip([-1.5 * wr, -0.5 * wr, 0.5 * wr, 1.5 * wr], r_bars):
        ax1.bar(xr + off, vals, width=wr, yerr=errs, capsize=1.2, error_kw={"lw": 0.55}, label=lbl, color=col, edgecolor="black", linewidth=0.35)
    ax1.axhline(50.0, color="#444444", linestyle=":", linewidth=0.8)
    ax1.set_xticks(xr)
    ax1.set_xticklabels(reg_lbls)
    ax1.set_ylim(42, 106)
    ax1.set_ylabel("Bal. Acc / AUC (%)")
    ax1.set_title("(a) Pristine Confirm")
    ax1.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.9, fontsize=5.9, columnspacing=0.35, handlelength=0.9, borderpad=0.16)

    # (b) Crossed-OOD Masked Diffusion (LLaDA-8B) Per-Level AUROC
    def seed_ll_lv(prefix, lv):
        vals = [100.0 * v["test_ood_llada"]["by_level"][lv]["auroc"] for k, v in per_seed.items() if k.startswith(prefix + "_seed")]
        return float(np.mean(vals)), float(np.std(vals, ddof=0))

    x_ll = np.arange(3)
    lvs_ll = ["E0", "E1", "E2"]
    ax2.plot(x_ll, [100 * pre["binoculars_qwen3b"]["test_ood_llada"]["by_level"][l]["auroc"] for l in lvs_ll],
             label="Binoc.", color=COLORS["bino"], linestyle="--", marker="s", linewidth=1.0, markersize=3.2)
    ax2.plot(x_ll, [100 * pre["desklib_raid_v1"]["test_ood_llada"]["by_level"][l]["auroc"] for l in lvs_ll],
             label="Desklib", color=COLORS["desklib"], linestyle="--", marker="^", linewidth=1.0, markersize=3.2)
    for pref, lbl, col, mk in [
        ("mb_v3_augmented_erm", "AugERM", COLORS["aug_erm"], "o"),
        ("mb_v3_strong_combined_no_nsd", "NoNSD", COLORS["strong_no_nsd"], "D"),
        ("mb_v3_cedar_split_op", "CEDAR", COLORS["cedar"], "*"),
    ]:
        ms_v = [seed_ll_lv(pref, l) for l in lvs_ll]
        ym = np.array([z[0] for z in ms_v])
        ys = np.array([z[1] for z in ms_v])
        ax2.plot(x_ll, ym, label=lbl, color=col, marker=mk, linewidth=(1.6 if lbl == "CEDAR" else 1.05), markersize=3.6)
        ax2.fill_between(x_ll, ym - ys, ym + ys, color=col, alpha=0.13)
    ax2.set_xticks(x_ll)
    ax2.set_xticklabels(["$E_0$", "$E_1$", "$E_2$"])
    ax2.set_xlabel("Edit Level (LLaDA-8B)")
    ax2.set_title("(b) OOD Diffusion AUC")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=5.9, handlelength=1.0, borderpad=0.16)

    # (c) Zero-Shot vs K=30 Human-Only Domain-Anchor Split-CRC Realized Cluster Human FPR
    c_tasks = ["Confirm", "ESL", "MixSet", "Beemo", "MAGE"]
    c_keys = ["test_confirm", "external_esl_full", "external_mixset_pure", "external_beemo", "external_mage_wild"]
    xc = np.arange(len(c_tasks))
    wc = 0.34
    zs_fpr = [100 * crc[k]["zero_shot_locked_val_calib"]["cluster_human_fpr"]["mean"] for k in c_keys]
    zs_err = [100 * crc[k]["zero_shot_locked_val_calib"]["cluster_human_fpr"]["std"] for k in c_keys]
    fs_fpr = [100 * crc[k]["few_shot_K30_human_anchor_crc"]["marginal_cluster_human_fpr"]["mean"] for k in c_keys]
    fs_err = [100 * crc[k]["few_shot_K30_human_anchor_crc"]["marginal_cluster_human_fpr"]["std"] for k in c_keys]
    ax3.bar(xc - wc / 2, zs_fpr, width=wc, yerr=zs_err, capsize=1.2, error_kw={"lw": 0.55}, color="#8DA0CB", edgecolor="black", linewidth=0.35, label="Zero-Shot $\\tau_{\\mathrm{cal}}$")
    ax3.bar(xc + wc / 2, fs_fpr, width=wc, yerr=fs_err, capsize=1.2, error_kw={"lw": 0.55}, color=COLORS["cedar"], edgecolor="black", linewidth=0.35, label="$K{=}30$ Anchor CRC")
    ax3.axhline(5.0, color="#D55E00", linestyle="--", linewidth=0.9, label="5% $\\alpha_{\\mathrm{fpr}}$ Bound")
    ax3.set_xticks(xc)
    ax3.set_xticklabels(c_tasks, rotation=20, ha="right")
    ax3.set_ylabel("Cluster Human FPR (%)")
    ax3.set_title("(c) $K{=}30$ Human-Anchor CRC")
    ax3.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=5.9, handlelength=1.0, borderpad=0.16)

    fig.tight_layout(pad=0.30, w_pad=0.75)
    save_both(fig, out_dir, "fig3_crossed_ood_and_external")


def plot_fig4(summary: dict, geom: dict, out_dir: str):
    agg = summary["trained_multi_seed_agg"]
    t2 = geom["theorem2_empirical_verification"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.5, 1.22))

    # (a) Theorem 2 Empirical Finite-Sample Convergence on 520 Untouched Octets
    N_vals = [16, 32, 64, 128, 256]
    op_m = np.array([t2["by_N"][str(n)]["op_err_mean"] for n in N_vals])
    op_s = np.array([t2["by_N"][str(n)]["op_err_sd"] for n in N_vals])
    sh_m = np.array([t2["by_N"][str(n)]["shrinker_err_mean"] for n in N_vals])
    sh_s = np.array([t2["by_N"][str(n)]["shrinker_err_sd"] for n in N_vals])
    eff_f = np.array([t2["by_N"][str(n)]["effective_inv_sqrt_n_factor"] for n in N_vals])
    ref_curve = op_m[0] * (eff_f / eff_f[0])

    ax1.plot(N_vals, op_m, color=COLORS["cedar"], marker="o", markersize=3.4, linewidth=1.4, label="$\\|\\hat{M}_N - M_{\\mathrm{ref}}\\|_{\\mathrm{op}}$")
    ax1.fill_between(N_vals, np.maximum(1e-4, op_m - op_s), op_m + op_s, color=COLORS["cedar"], alpha=0.14)
    ax1.plot(N_vals, sh_m, color=COLORS["supcon"], marker="s", markersize=3.2, linewidth=1.3, label="$\\|\\hat{P}_{\\mathcal{N}} - P_{\\mathcal{N},\\mathrm{ref}}\\|_{\\mathrm{op}}$")
    ax1.fill_between(N_vals, np.maximum(1e-4, sh_m - sh_s), sh_m + sh_s, color=COLORS["supcon"], alpha=0.14)
    ax1.plot(N_vals, ref_curve, color="#444444", linestyle="--", linewidth=0.95, label="Theory $\\propto (N^{-1}{+}N_{\\mathrm{ref}}^{-1})^{1/2}$")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(N_vals)
    ax1.set_xticklabels([str(n) for n in N_vals])
    ax1.set_xlabel("Operator Octets $N$")
    ax1.set_ylabel("Spectral Error (log)")
    ax1.set_title("(a) Theorem 1 Rate")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.90, fontsize=5.7, handlelength=1.0, borderpad=0.16, labelspacing=0.12)

    # (b) Cross-Editor Energy Capture vs Origin Preservation on test_confirm & test_id
    ranks = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    c_cf = geom["cross_editor_variance_capture_by_rank"]["test_confirm"]
    c_id = geom["cross_editor_variance_capture_by_rank"]["test_id"]
    ax2.plot(ranks, [100 * c_cf[str(r)]["edit_energy_captured"] for r in ranks],
             color=COLORS["cedar"], linestyle="-", marker="o", markersize=2.8, linewidth=1.35, label="Conf. Edit")
    ax2.plot(ranks, [100 * c_cf[str(r)]["orig_energy_captured"] for r in ranks],
             color=COLORS["cedar"], linestyle=":", marker="^", markersize=2.6, linewidth=1.05, label="Conf. Orig")
    ax2.plot(ranks, [100 * c_id[str(r)]["edit_energy_captured"] for r in ranks],
             color=COLORS["supcon"], linestyle="-", marker="s", markersize=2.8, linewidth=1.2, label="ID Edit")
    ax2.plot(ranks, [100 * c_id[str(r)]["orig_energy_captured"] for r in ranks],
             color=COLORS["supcon"], linestyle=":", marker="d", markersize=2.6, linewidth=1.0, label="ID Orig")
    ax2.axvline(24, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Subspace Rank $r$")
    ax2.set_ylabel("Energy Captured (%)")
    ax2.set_title("(b) Spectral Selectivity")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.90, fontsize=5.9, ncol=2, columnspacing=0.35, handlelength=0.9, borderpad=0.16)

    # (c) Rank & Protection Sensitivity at Identical N_fit=600 Parents
    r_keys = [
        (0, "mb_v3_strong_combined_fit600_no_nsd"),
        (8, "mb_v3_cedar_rank8"),
        (24, "mb_v3_cedar_split_op"),
        (48, "mb_v3_cedar_rank48"),
    ]
    rx = [p[0] for p in r_keys]
    cf_wg_m = np.array([100 * agg[p[1]]["confirm_wg_E0_E3"]["mean"] for p in r_keys])
    cf_wg_s = np.array([100 * agg[p[1]]["confirm_wg_E0_E3"]["std"] for p in r_keys])
    id_wg_m = np.array([100 * agg[p[1]]["id_wg_E0_E3"]["mean"] for p in r_keys])
    id_wg_s = np.array([100 * agg[p[1]]["id_wg_E0_E3"]["std"] for p in r_keys])
    ms_m = np.array([100 * agg[p[1]]["mixset_pure_auroc"]["mean"] for p in r_keys])
    ms_s = np.array([100 * agg[p[1]]["mixset_pure_auroc"]["std"] for p in r_keys])

    ax3.plot(rx, cf_wg_m, color=COLORS["cedar"], marker="*", markersize=4.8, linewidth=1.6, label="Conf. WG")
    ax3.fill_between(rx, cf_wg_m - cf_wg_s, cf_wg_m + cf_wg_s, color=COLORS["cedar"], alpha=0.14)
    ax3.plot(rx, id_wg_m, color=COLORS["supcon"], marker="o", markersize=3.2, linewidth=1.2, label="ID WG")
    ax3.fill_between(rx, id_wg_m - id_wg_s, id_wg_m + id_wg_s, color=COLORS["supcon"], alpha=0.14)
    ax3.plot(rx, ms_m, color=COLORS["aug_erm"], marker="s", markersize=3.2, linewidth=1.2, label="MixSet AUC")
    ax3.fill_between(rx, ms_m - ms_s, ms_m + ms_s, color=COLORS["aug_erm"], alpha=0.14)
    ax3.set_xticks(rx)
    ax3.set_xlabel("Deflation Rank $r$ ($N_{\\mathrm{fit}}{=}600$)")
    ax3.set_ylabel("Acc / AUC (%)")
    ax3.set_title("(c) Rank Sensitivity")
    ax3.legend(loc="lower right", frameon=True, framealpha=0.90, fontsize=5.9, handlelength=0.95, borderpad=0.16)

    fig.tight_layout(pad=0.30, w_pad=0.75)
    save_both(fig, out_dir, "fig4_nsd_geometry_and_ablations")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res_dir = os.path.join(root, "results_v3")
    fig_dir = os.path.join(root, "paper", "figures")

    summary = json.load(open(os.path.join(res_dir, "master_evaluation_summary_v3.json")))
    geom = json.load(open(os.path.join(res_dir, "nsd_v3_theorem2_and_geometry.json")))

    plot_fig1(summary, summary["trained_per_seed"], fig_dir)
    plot_fig2(summary, root, fig_dir)
    plot_fig3(summary, summary["trained_per_seed"], fig_dir)
    plot_fig4(summary, geom, fig_dir)
    print("\n=== ALL 4 V3 PUBLICATION VECTOR PDF FIGURES GENERATED (TYPE-42)! ===")


if __name__ == "__main__":
    main()

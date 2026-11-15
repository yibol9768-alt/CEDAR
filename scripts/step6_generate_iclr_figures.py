import os
import glob
import json
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Approximately native ICLR \textwidth (5.5 in canvas, ~5.465 in tight-cropped PDF included at width=\textwidth):
# Type-42 TrueType fonts (zero Type-3) & 6.2-8.2pt vector text!
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.8,
    "axes.titlesize": 8.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.8,
    "xtick.labelsize": 7.1,
    "ytick.labelsize": 7.1,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.45,
})

COLORS = {
    "cedar": "#0072B2",        # Deep Blue (Ours)
    "cedar_alt": "#56B4E9",    # Sky Blue
    "supcon": "#009E73",       # Bluish Green
    "aug_erm": "#E69F00",      # Orange
    "uned_erm": "#CC79A7",     # Reddish Purple
    "desklib": "#D55E00",      # Vermilion
    "bino": "#7F7F7F",         # Neutral Gray
    "fastd": "#999933",        # Olive
    "radar": "#882255",        # Wine
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

    def seed_ms_hwt_e0(prefix):
        vals = [
            v["external_mixset"]["hwt_llm_polished_fpr_tau_e0_5"]
            for k, v in per_seed.items()
            if k.startswith(prefix + "_seed")
        ]
        return float(np.mean(vals)), float(np.std(vals, ddof=0))

    def seed_id_e2_e0(prefix):
        vals = [
            v["test_id_recoverability"]["E2"]["fpr_at_tau_e0_5"]
            for k, v in per_seed.items()
            if k.startswith(prefix + "_seed")
        ]
        return float(np.mean(vals)), float(np.std(vals, ddof=0))

    methods = [
        ("Fast-Det.", pre["fast_detectgpt_qwen3b"]["test_id_recoverability"]["E2"]["fpr_at_tau_e0_5"], 0.0,
         pre["fast_detectgpt_qwen3b"]["external_mixset"]["hwt_llm_polished_fpr_tau_e0_5"], 0.0, COLORS["fastd"]),
        ("Binoc.", pre["binoculars_qwen3b"]["test_id_recoverability"]["E2"]["fpr_at_tau_e0_5"], 0.0,
         pre["binoculars_qwen3b"]["external_mixset"]["hwt_llm_polished_fpr_tau_e0_5"], 0.0, COLORS["bino"]),
        ("RADAR", pre["radar_vicuna7b"]["test_id_recoverability"]["E2"]["fpr_at_tau_e0_5"], 0.0,
         pre["radar_vicuna7b"]["external_mixset"]["hwt_llm_polished_fpr_tau_e0_5"], 0.0, COLORS["radar"]),
        ("Desklib", pre["desklib_raid_v1"]["test_id_recoverability"]["E2"]["fpr_at_tau_e0_5"], 0.0,
         pre["desklib_raid_v1"]["external_mixset"]["hwt_llm_polished_fpr_tau_e0_5"], 0.0, COLORS["desklib"]),
        ("Uned. ERM", *seed_id_e2_e0("mb_unedited_erm"), *seed_ms_hwt_e0("mb_unedited_erm"), COLORS["uned_erm"]),
        ("SupCon", *seed_id_e2_e0("mb_supcon_orth_disentangle"), *seed_ms_hwt_e0("mb_supcon_orth_disentangle"), COLORS["supcon"]),
        ("Aug. ERM", *seed_id_e2_e0("mb_augmented_erm"), *seed_ms_hwt_e0("mb_augmented_erm"), COLORS["aug_erm"]),
        ("CEDAR", *seed_id_e2_e0("mb_cedar_full"), *seed_ms_hwt_e0("mb_cedar_full"), COLORS["cedar"]),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 1.28), gridspec_kw={"width_ratios": [1.14, 1.0]})

    names = [m[0] for m in methods]
    id_fpr = [100.0 * m[1] for m in methods]
    id_err = [100.0 * m[2] for m in methods]
    ms_fpr = [100.0 * m[3] for m in methods]
    ms_err = [100.0 * m[4] for m in methods]

    x = np.arange(len(names))
    w = 0.36
    ax1.bar(x - w / 2, id_fpr, width=w, yerr=id_err, capsize=1.4, error_kw={"lw": 0.6},
            color="#8DA0CB", edgecolor="black", linewidth=0.4, label="ID Hum $E_2$ (@$\\tau_{E_0}$)")
    ax1.bar(x + w / 2, ms_fpr, width=w, yerr=ms_err, capsize=1.4, error_kw={"lw": 0.6},
            color="#FC8D62", edgecolor="black", linewidth=0.4, label="MixSet Pol (@$\\tau_{E_0}$)")
    ax1.axhline(5.0, color="#D55E00", linestyle="--", linewidth=0.9, label="5% Target")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=24, ha="right")
    ax1.set_ylabel("Human FPR (%)")
    ax1.set_title("(a) FPR Inflation at $\\tau_{E_0}$")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, handlelength=1.2, borderpad=0.25, labelspacing=0.2)

    pts = [
        ("Fast-Det.", 1.0 - pre["fast_detectgpt_qwen3b"]["external_mixset"]["hwt_llm_polished_fpr_tau_e0_5"],
         pre["fast_detectgpt_qwen3b"]["test_id_worst_group_acc_E0_E3"], 0.0, 0.0, COLORS["fastd"], "D"),
        ("Binoc.", 1.0 - pre["binoculars_qwen3b"]["external_mixset"]["hwt_llm_polished_fpr_tau_e0_5"],
         pre["binoculars_qwen3b"]["test_id_worst_group_acc_E0_E3"], 0.0, 0.0, COLORS["bino"], "s"),
        ("RADAR", 1.0 - pre["radar_vicuna7b"]["external_mixset"]["hwt_llm_polished_fpr_tau_e0_5"],
         pre["radar_vicuna7b"]["test_id_worst_group_acc_E0_E3"], 0.0, 0.0, COLORS["radar"], "v"),
        ("Desklib", 1.0 - pre["desklib_raid_v1"]["external_mixset"]["hwt_llm_polished_fpr_tau_e0_5"],
         pre["desklib_raid_v1"]["test_id_worst_group_acc_E0_E3"], 0.0, 0.0, COLORS["desklib"], "^"),
        ("Uned. ERM", 1.0 - seed_ms_hwt_e0("mb_unedited_erm")[0],
         agg["mb_unedited_erm"]["id_wg_E0_E3"]["mean"], seed_ms_hwt_e0("mb_unedited_erm")[1],
         agg["mb_unedited_erm"]["id_wg_E0_E3"]["std"], COLORS["uned_erm"], "P"),
        ("SupCon", 1.0 - seed_ms_hwt_e0("mb_supcon_orth_disentangle")[0],
         agg["mb_supcon_orth_disentangle"]["id_wg_E0_E3"]["mean"], seed_ms_hwt_e0("mb_supcon_orth_disentangle")[1],
         agg["mb_supcon_orth_disentangle"]["id_wg_E0_E3"]["std"], COLORS["supcon"], "X"),
        ("Aug. ERM", 1.0 - seed_ms_hwt_e0("mb_augmented_erm")[0],
         agg["mb_augmented_erm"]["id_wg_E0_E3"]["mean"], seed_ms_hwt_e0("mb_augmented_erm")[1],
         agg["mb_augmented_erm"]["id_wg_E0_E3"]["std"], COLORS["aug_erm"], "o"),
        ("CEDAR", 1.0 - seed_ms_hwt_e0("mb_cedar_full")[0],
         agg["mb_cedar_full"]["id_wg_E0_E3"]["mean"], seed_ms_hwt_e0("mb_cedar_full")[1],
         agg["mb_cedar_full"]["id_wg_E0_E3"]["std"], COLORS["cedar"], "*"),
    ]

    for lbl, xv, yv, xe, ye, col, mk in pts:
        sz = 72 if "CEDAR" in lbl else 30
        ax2.errorbar(100 * xv, 100 * yv, xerr=100 * xe, yerr=100 * ye, fmt="none", ecolor=col, alpha=0.65, capsize=1.5, elinewidth=0.7)
        ax2.scatter([100 * xv], [100 * yv], s=sz, color=col, marker=mk, edgecolors="black", linewidths=0.5, zorder=4, label=lbl)

    ax2.set_ylim(-2, 76)
    ax2.set_xlabel("MixSet Specificity at $\\tau_{E_0}$ (%)")
    ax2.set_ylabel("ID WG Acc at $\\tau_{\\mathrm{cal}}^{\\mathrm{all}}$ (%)")
    ax2.set_title("(b) Operating Trade-Off")
    ax2.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.88, fontsize=6.5, columnspacing=0.4, handletextpad=0.2, borderpad=0.2, labelspacing=0.15)

    fig.tight_layout(pad=0.35, w_pad=0.8)
    save_both(fig, out_dir, "fig1_editorial_paradox_overview")


def plot_fig2(summary: dict, per_seed: dict, root_dir: str, out_dir: str):
    pre = summary["pretrained_and_probes"]
    agg = summary["trained_multi_seed_agg"]
    lvs = ["E0", "E1", "E2", "E3", "E4"]
    lv_labels = ["$E_0$", "$E_1$", "$E_2$", "$E_3$", "$E_4$"]
    x = np.arange(len(lvs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 1.18))

    series_a = [
        ("Binoc.", [pre["binoculars_qwen3b"]["test_id_recoverability"][k]["bal_acc"] for k in lvs], [0]*5, COLORS["bino"], "--", "s"),
        ("Desklib", [pre["desklib_raid_v1"]["test_id_recoverability"][k]["bal_acc"] for k in lvs], [0]*5, COLORS["desklib"], "--", "^"),
        ("Uned. ERM", [agg["mb_unedited_erm"]["id_recoverability"][k]["bal_acc"]["mean"] for k in lvs],
         [agg["mb_unedited_erm"]["id_recoverability"][k]["bal_acc"]["std"] for k in lvs], COLORS["uned_erm"], "-.", "P"),
        ("Aug. ERM", [agg["mb_augmented_erm"]["id_recoverability"][k]["bal_acc"]["mean"] for k in lvs],
         [agg["mb_augmented_erm"]["id_recoverability"][k]["bal_acc"]["std"] for k in lvs], COLORS["aug_erm"], "-", "o"),
        ("SupCon", [agg["mb_supcon_orth_disentangle"]["id_recoverability"][k]["bal_acc"]["mean"] for k in lvs],
         [agg["mb_supcon_orth_disentangle"]["id_recoverability"][k]["bal_acc"]["std"] for k in lvs], COLORS["supcon"], "-", "X"),
        ("CEDAR", [agg["mb_cedar_full"]["id_recoverability"][k]["bal_acc"]["mean"] for k in lvs],
         [agg["mb_cedar_full"]["id_recoverability"][k]["bal_acc"]["std"] for k in lvs], COLORS["cedar"], "-", "*"),
    ]

    for lbl, ys, es, col, ls, mk in series_a:
        ys_a = 100.0 * np.asarray(ys)
        es_a = 100.0 * np.asarray(es)
        lw = 1.6 if "CEDAR" in lbl else 1.05
        ax1.plot(x, ys_a, label=lbl, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=3.8)
        if np.any(es_a > 0):
            ax1.fill_between(x, ys_a - es_a, ys_a + es_a, color=col, alpha=0.14)

    cedar_e4_auc = agg["mb_cedar_full"]["id_recoverability"]["E4"]["auroc"]["mean"]
    ax1.scatter([4.06], [100.0 * cedar_e4_auc], marker="D", s=24, facecolors="white", edgecolors=COLORS["cedar"], linewidths=1.1, zorder=5,
                label=f"$E_4$ AUC ({r_pct(cedar_e4_auc, 1)}%)")

    ax1.axhline(50.0, color="black", linestyle=":", linewidth=0.8)
    ax1.axvspan(3.55, 4.3, color="#EEEEEE", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lv_labels)
    ax1.set_xlabel("Editorial Intervention Level")
    ax1.set_ylim(44, 101)
    ax1.set_ylabel("Balanced Acc (%)")
    ax1.set_title("(a) Recoverability ($E_0 \\to E_4$)")
    ax1.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.92, columnspacing=0.45, handlelength=1.2, handletextpad=0.25, borderpad=0.22, labelspacing=0.18)

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
        if a_hat is not None:
            conf = conf * (1.0 - 0.5 * np.clip(np.asarray(a_hat, dtype=np.float64), 0.0, 1.0))
        order = np.argsort(-conf)
        errs = ((s > tau).astype(np.int64) != y_tid).astype(np.float64)
        return np.array([100.0 * float(errs[order[:max(1, int(round(0.01 * c * len(y_tid))))]].mean()) for c in cov_grid])

    pre_p = json.load(open(os.path.join(root_dir, "results", "preds_pretrained_and_probes.json")))
    desk_tau = pre["desklib_raid_v1"]["tau_cal_all5"]
    desk_curve = eval_dense_rc(pre_p["desklib_raid_v1"]["test_id"], desk_tau)
    ax2.plot(cov_grid, desk_curve, label="Desklib ($m_\\tau$)", color=COLORS["desklib"], linestyle="--", marker="^", linewidth=1.05, markersize=3.5)

    pred_dir = os.path.join(root_dir, "results", "model_preds_v2")
    for pref, lbl, col, ls, mk, use_int in [
        ("mb_augmented_erm", "Aug. ERM ($m_\\tau$)", COLORS["aug_erm"], "-", "o", False),
        ("mb_supcon_orth_disentangle", "SupCon ($m_\\tau$)", COLORS["supcon"], "-", "X", False),
        ("mb_cedar_full", "CEDAR ($m_\\tau$: 12.58%)", COLORS["cedar_alt"], "--", "d", False),
        ("mb_cedar_full", "CEDAR ($c_\\tau$: 12.72%)", COLORS["cedar"], "-", "*", True),
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
        ax2.plot(cov_grid, m_rc, label=lbl, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=3.6)
        ax2.fill_between(cov_grid, m_rc - s_rc, m_rc + s_rc, color=col, alpha=0.12)

    ax2.set_xlabel("Retained Coverage (%)")
    ax2.set_ylabel("Selective Error (%)")
    ax2.set_title("(b) Risk--Coverage ($20\\%$--$100\\%$)")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92, handlelength=1.2, handletextpad=0.25, borderpad=0.22, labelspacing=0.18)

    fig.tight_layout(pad=0.35, w_pad=0.8)
    save_both(fig, out_dir, "fig2_recoverability_phase_curves")


def plot_fig3(summary: dict, per_seed: dict, out_dir: str):
    pre = summary["pretrained_and_probes"]
    agg = summary["trained_multi_seed_agg"]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.18), gridspec_kw={"width_ratios": [0.92, 0.92, 1.25]})

    def seed_ood_lv(prefix, ood_key, lv):
        vals = [
            100.0 * v[ood_key]["by_level"][lv]["auroc"]
            for k, v in per_seed.items()
            if k.startswith(prefix + "_seed")
        ]
        return float(np.mean(vals)), float(np.std(vals, ddof=0))

    for ax_idx, (ood_k, title_str) in enumerate([
        ("test_ood_ar", "(a) OOD-AR (OLMo-2)"),
        ("test_ood_llada", "(b) OOD-Diff (LLaDA)"),
    ]):
        ax = axes[ax_idx]
        lvs = ["E0", "E1", "E2"]
        x = np.arange(3)
        ax.plot(x, [100 * pre["binoculars_qwen3b"][ood_k]["by_level"][l]["auroc"] for l in lvs],
                label="Binoc.", color=COLORS["bino"], linestyle="--", marker="s", linewidth=1.05, markersize=3.4)
        ax.plot(x, [100 * pre["desklib_raid_v1"][ood_k]["by_level"][l]["auroc"] for l in lvs],
                label="Desklib", color=COLORS["desklib"], linestyle="--", marker="^", linewidth=1.05, markersize=3.4)
        for pref, lbl, col, mk in [
            ("mb_augmented_erm", "Aug. ERM", COLORS["aug_erm"], "o"),
            ("mb_supcon_orth_disentangle", "SupCon", COLORS["supcon"], "X"),
            ("mb_cedar_full", "CEDAR", COLORS["cedar"], "*"),
        ]:
            ms_v = [seed_ood_lv(pref, ood_k, l) for l in lvs]
            ym = np.array([z[0] for z in ms_v])
            ys = np.array([z[1] for z in ms_v])
            lw = 1.6 if "CEDAR" in lbl else 1.05
            ax.plot(x, ym, label=lbl, color=col, marker=mk, linewidth=lw, markersize=3.8)
            ax.fill_between(x, ym - ys, ym + ys, color=col, alpha=0.14)
        ax.set_xticks(x)
        ax.set_xticklabels(["$E_0$", "$E_1$", "$E_2$"])
        ax.set_xlabel("Edit Level")
        if ax_idx == 0:
            ax.set_ylabel("AUROC (%)")
            ax.legend(loc="lower left", frameon=True, framealpha=0.92, handlelength=1.1, borderpad=0.2, labelspacing=0.18)
        ax.set_title(title_str)

    ax3 = axes[2]
    ext_tasks = ["BM\nExp", "BM\nG4o", "MS\nPure", "MS\nAll", "MAGE\nWild"]
    x3 = np.arange(len(ext_tasks))
    w = 0.19

    bars = [
        ("Desklib", [
            100 * pre["desklib_raid_v1"]["external_beemo"]["auroc_expert_edited_vs_human"],
            100 * pre["desklib_raid_v1"]["external_beemo"]["auroc_gpt4o_edited_vs_human"],
            100 * pre["desklib_raid_v1"]["external_mixset"]["pure_origin_auroc"],
            100 * pre["desklib_raid_v1"]["external_mixset"]["all_720_auroc"],
            100 * pre["desklib_raid_v1"]["external_mage_wild"]["auroc"],
        ], [0]*5, COLORS["desklib"]),
        ("Aug. ERM", [
            100 * agg["mb_augmented_erm"]["beemo_exp_auroc"]["mean"],
            100 * agg["mb_augmented_erm"]["beemo_gpt4o_auroc"]["mean"],
            100 * agg["mb_augmented_erm"]["mixset_pure_auroc"]["mean"],
            100 * agg["mb_augmented_erm"]["mixset_all_auroc"]["mean"],
            100 * agg["mb_augmented_erm"]["mage_wild_auroc"]["mean"],
        ], [
            100 * agg["mb_augmented_erm"]["beemo_exp_auroc"]["std"],
            100 * agg["mb_augmented_erm"]["beemo_gpt4o_auroc"]["std"],
            100 * agg["mb_augmented_erm"]["mixset_pure_auroc"]["std"],
            100 * agg["mb_augmented_erm"]["mixset_all_auroc"]["std"],
            100 * agg["mb_augmented_erm"]["mage_wild_auroc"]["std"],
        ], COLORS["aug_erm"]),
        ("SupCon", [
            100 * agg["mb_supcon_orth_disentangle"]["beemo_exp_auroc"]["mean"],
            100 * agg["mb_supcon_orth_disentangle"]["beemo_gpt4o_auroc"]["mean"],
            100 * agg["mb_supcon_orth_disentangle"]["mixset_pure_auroc"]["mean"],
            100 * agg["mb_supcon_orth_disentangle"]["mixset_all_auroc"]["mean"],
            100 * agg["mb_supcon_orth_disentangle"]["mage_wild_auroc"]["mean"],
        ], [
            100 * agg["mb_supcon_orth_disentangle"]["beemo_exp_auroc"]["std"],
            100 * agg["mb_supcon_orth_disentangle"]["beemo_gpt4o_auroc"]["std"],
            100 * agg["mb_supcon_orth_disentangle"]["mixset_pure_auroc"]["std"],
            100 * agg["mb_supcon_orth_disentangle"]["mixset_all_auroc"]["std"],
            100 * agg["mb_supcon_orth_disentangle"]["mage_wild_auroc"]["std"],
        ], COLORS["supcon"]),
        ("CEDAR", [
            100 * agg["mb_cedar_full"]["beemo_exp_auroc"]["mean"],
            100 * agg["mb_cedar_full"]["beemo_gpt4o_auroc"]["mean"],
            100 * agg["mb_cedar_full"]["mixset_pure_auroc"]["mean"],
            100 * agg["mb_cedar_full"]["mixset_all_auroc"]["mean"],
            100 * agg["mb_cedar_full"]["mage_wild_auroc"]["mean"],
        ], [
            100 * agg["mb_cedar_full"]["beemo_exp_auroc"]["std"],
            100 * agg["mb_cedar_full"]["beemo_gpt4o_auroc"]["std"],
            100 * agg["mb_cedar_full"]["mixset_pure_auroc"]["std"],
            100 * agg["mb_cedar_full"]["mixset_all_auroc"]["std"],
            100 * agg["mb_cedar_full"]["mage_wild_auroc"]["std"],
        ], COLORS["cedar"]),
    ]

    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    for off, (lbl, vals, errs, col) in zip(offsets, bars):
        ax3.bar(x3 + off, vals, width=w, yerr=errs, capsize=1.2, error_kw={"lw": 0.55}, label=lbl, color=col, edgecolor="black", linewidth=0.35)
    ax3.axhline(50.0, color="#444444", linestyle=":", linewidth=0.8, zorder=2)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(ext_tasks)
    ax3.set_ylim(28, 118)
    ax3.set_title("(c) External AUROC")
    ax3.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92, columnspacing=0.45, handlelength=1.0, handletextpad=0.25, borderpad=0.2, labelspacing=0.18)

    fig.tight_layout(pad=0.35, w_pad=0.95)
    save_both(fig, out_dir, "fig3_crossed_ood_and_external")


def plot_fig4(summary: dict, geom: dict, out_dir: str):
    agg = summary["trained_multi_seed_agg"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.5, 1.16))

    ranks = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    c_id = geom["cross_editor_variance_capture_by_rank"]["test_id"]
    c_ar = geom["cross_editor_variance_capture_by_rank"]["test_ood_ar"]
    c_ll = geom["cross_editor_variance_capture_by_rank"]["test_ood_llada"]

    ax1.plot(ranks, [100 * c_id[str(r)]["edit_energy_captured"] for r in ranks],
             color=COLORS["cedar"], linestyle="-", marker="o", markersize=2.8, linewidth=1.3, label="ID Ed. ($3.69\\times$)")
    ax1.plot(ranks, [100 * c_id[str(r)]["orig_energy_captured"] for r in ranks],
             color=COLORS["cedar"], linestyle=":", marker="^", markersize=2.6, linewidth=1.0, label="ID Orig")
    ax1.plot(ranks, [100 * c_ar[str(r)]["edit_energy_captured"] for r in ranks],
             color=COLORS["supcon"], linestyle="-", marker="s", markersize=2.8, linewidth=1.2, label="AR Ed. ($2.73\\times$)")
    ax1.plot(ranks, [100 * c_ar[str(r)]["orig_energy_captured"] for r in ranks],
             color=COLORS["supcon"], linestyle=":", marker="d", markersize=2.6, linewidth=1.0, label="AR Orig")
    ax1.plot(ranks, [100 * c_ll[str(r)]["edit_energy_captured"] for r in ranks],
             color=COLORS["desklib"], linestyle="-", marker="P", markersize=3.0, linewidth=1.2, label="Diff Ed. ($1.01\\times$)")
    ax1.plot(ranks, [100 * c_ll[str(r)]["orig_energy_captured"] for r in ranks],
             color=COLORS["desklib"], linestyle="--", marker="v", markersize=2.6, linewidth=1.0, label="Diff Orig")

    ax1.axvline(32, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_xlabel("Subspace Rank $r$")
    ax1.set_ylim(-1, 31)
    ax1.set_ylabel("Energy Capture (%)")
    ax1.set_title("(a) Orthogonal Capture")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.90, fontsize=6.2, ncol=2, columnspacing=0.35, handlelength=0.9, handletextpad=0.2, borderpad=0.16, labelspacing=0.12)

    nsd_d = summary["nsd_subspace_diagnostics"]
    ab_keys = [
        ("Unprot.", "mb_ablation_unprotected_nsd"),
        ("w/o DTE", "mb_ablation_no_dte"),
        ("MLP", "mb_ablation_param_matched_mlp"),
        ("CEDAR", "mb_cedar_full"),
    ]
    xb = np.arange(len(ab_keys))
    wb = 0.26
    edit_rem = [100 * nsd_d[k]["frac_edit_energy_removed"]["mean"] for _, k in ab_keys]
    edit_err = [100 * nsd_d[k]["frac_edit_energy_removed"]["std"] for _, k in ab_keys]
    orig_rem = [100 * nsd_d[k]["frac_orig_energy_removed"]["mean"] for _, k in ab_keys]
    orig_err = [100 * nsd_d[k]["frac_orig_energy_removed"]["std"] for _, k in ab_keys]
    wg_vals = [100 * agg[k]["id_wg_E0_E3"]["mean"] for _, k in ab_keys]
    wg_errs = [100 * agg[k]["id_wg_E0_E3"]["std"] for _, k in ab_keys]

    ax2.bar(xb - wb, edit_rem, width=wb, yerr=edit_err, capsize=1.2, error_kw={"lw": 0.55}, color=COLORS["cedar_alt"], edgecolor="black", linewidth=0.35, label="Edit Atten.")
    ax2.bar(xb, orig_rem, width=wb, yerr=orig_err, capsize=1.2, error_kw={"lw": 0.55}, color=COLORS["desklib"], edgecolor="black", linewidth=0.35, label="Orig Atten.")
    ax2.bar(xb + wb, wg_vals, width=wb, yerr=wg_errs, capsize=1.2, error_kw={"lw": 0.55}, color=COLORS["cedar"], edgecolor="black", linewidth=0.35, label="ID WG Acc")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(["Unpr.", "NoDTE", "MLP", "CEDAR"], rotation=15, ha="right")
    ax2.set_ylim(0, 126)
    ax2.set_ylabel("Atten. / Acc (%)")
    ax2.set_title("(b) Signal Protection")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.90, fontsize=6.4, handlelength=0.95, borderpad=0.16, labelspacing=0.12)

    r_keys = [
        (0, "mb_ablation_no_nsd"),
        (8, "mb_cedar_rank8"),
        (16, "mb_cedar_rank16"),
        (32, "mb_cedar_full"),
        (64, "mb_cedar_rank64"),
    ]
    present = [(rv, rk) for rv, rk in r_keys if rk in agg]
    rx = [p[0] for p in present]
    wg_m = np.array([100 * agg[p[1]]["id_wg_E0_E3"]["mean"] for p in present])
    wg_s = np.array([100 * agg[p[1]]["id_wg_E0_E3"]["std"] for p in present])
    id_m = np.array([100 * agg[p[1]]["id_mean_bal_E0_E3"]["mean"] for p in present])
    id_s = np.array([100 * agg[p[1]]["id_mean_bal_E0_E3"]["std"] for p in present])
    ms_m = np.array([100 * agg[p[1]]["mixset_pure_auroc"]["mean"] for p in present])
    ms_s = np.array([100 * agg[p[1]]["mixset_pure_auroc"]["std"] for p in present])

    ax3.plot(rx, id_m, color=COLORS["supcon"], marker="o", markersize=3.2, linewidth=1.2, label="ID Bal")
    ax3.fill_between(rx, id_m - id_s, id_m + id_s, color=COLORS["supcon"], alpha=0.14)
    ax3.plot(rx, wg_m, color=COLORS["cedar"], marker="*", markersize=4.8, linewidth=1.6, label="ID WG")
    ax3.fill_between(rx, wg_m - wg_s, wg_m + wg_s, color=COLORS["cedar"], alpha=0.14)
    ax3.plot(rx, ms_m, color=COLORS["aug_erm"], marker="s", markersize=3.2, linewidth=1.2, label="MixSet")
    ax3.fill_between(rx, ms_m - ms_s, ms_m + ms_s, color=COLORS["aug_erm"], alpha=0.14)

    ax3.set_xticks(rx)
    ax3.set_xlabel("Deflation Rank $r$")
    ax3.set_ylabel("Acc / AUC (%)")
    ax3.set_title("(c) Rank Sensitivity")
    ax3.legend(loc="center right", frameon=True, framealpha=0.92, handlelength=1.0, borderpad=0.18, labelspacing=0.15)

    fig.tight_layout(pad=0.35, w_pad=0.95)
    save_both(fig, out_dir, "fig4_nsd_geometry_and_ablations")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res_dir = os.path.join(root, "results")
    fig_dir = os.path.join(root, "paper", "figures")

    with open(os.path.join(res_dir, "master_evaluation_summary_v2.json")) as f:
        summary = json.load(f)
    with open(os.path.join(res_dir, "nsd_spectrum_and_geometry.json")) as f:
        geom = json.load(f)

    plot_fig1(summary, summary["trained_per_seed"], fig_dir)
    plot_fig2(summary, summary["trained_per_seed"], root, fig_dir)
    plot_fig3(summary, summary["trained_per_seed"], fig_dir)
    plot_fig4(summary, geom, fig_dir)
    print("\n=== ALL 4 PUBLICATION VECTOR PDF FIGURES GENERATED (NATIVE 5.5IN, TYPE-42)! ===")


if __name__ == "__main__":
    main()

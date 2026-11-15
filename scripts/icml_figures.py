#!/usr/bin/env python3
"""ICML figures (vector PDF, ICML column width 3.25 in). Usage: python scripts/icml_figures.py [--dev] [--unseal]
fig_cells:   per-cell accuracy at the locked threshold, 8B pair vs 8B aug, on each test split.
fig_shift:   distribution of the edit-induced logit shift s(E3) - s(E0) on human parents (Prop. 1's Delta), held-out editors.
fig_scaling: human-E3 FPR and matched-FPR WG vs backbone size, pair vs aug."""
import json, os, sys
import numpy as np
LEG = {"8B pair": "sibling-aligned", "8B aug": "Augmented ERM"}
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import icml_final_analysis as A  # noqa: E402  (reads --dev / --unseal from argv)

OUT = os.path.join(A.ROOT, "paper_icml", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
C_PAIR, C_AUG = "#1b6ca8", "#d1495b"
NAMES = {"test_confirm": "Confirm", "test_confirm2": "Confirm-2", "test_confirm3": "Confirm-3"}
FIG_SPLITS = [s for s in A.SPLITS if s in NAMES]  # the length-matched subset is reported in tables only


def arrs(sp, method):
    S = A.Split(sp); src, tag = A.METHODS[method]
    return S, A.seed_arrays(S, A.load_seeds(src, tag, sp))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def fig_cells():
    fig, axes = plt.subplots(1, len(FIG_SPLITS), figsize=(3.25 if len(FIG_SPLITS) < 3 else 6.75, 1.7), sharey=True)
    axes = np.atleast_1d(axes); x = np.arange(8); lab = [f"E{c % 4}" for c in range(8)]
    for ax, sp in zip(axes, FIG_SPLITS):
        for k, (m, col) in enumerate((("8B pair", C_PAIR), ("8B aug", C_AUG))):
            S, ar = arrs(sp, m)
            acc = np.array([[100 * np.mean(((sc > tau) == (S.y == 1))[S.cell == c]) for c in range(8)] for sc, tau, _, _ in ar])
            ax.bar(x + (k - 0.5) * 0.38, acc.mean(0), 0.38, yerr=acc.std(0), color=col, label=LEG.get(m, m), capsize=1, lw=0, error_kw={"lw": 0.6})
        ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(40, 101); ax.set_title(NAMES[sp]); ax.axvline(3.5, color="0.7", lw=0.5)
        for xc, t in ((1.5, "human"), (5.5, "machine")):
            ax.text(xc, -0.24, t, ha="center", va="top", fontsize=7, color="0.3", transform=ax.get_xaxis_transform())
    axes[0].set_ylabel("cell accuracy (%)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.3, rect=(0, 0.02, 1, 0.9)); fig.subplots_adjust(bottom=0.3); fig.savefig(os.path.join(OUT, "fig_cells.pdf")); plt.close(fig)


def fig_shift():
    """Survival of the edit-induced human shift Delta = s(E3) - s(E0), relative to the class separation, on calibration
    data (training editors) and on test splits (unseen editors). Pair and aug look alike on training editors; only pair transfers."""
    import icml_prop1_check as P
    sps = [s for s in FIG_SPLITS if s != "test_confirm"]
    panels = [("val_calib", "seen editors (val.)")] + [(s, "unseen: " + NAMES[s]) for s in sps]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.25 if len(panels) < 3 else 6.75, 1.5), sharey=True); axes = np.atleast_1d(axes)
    xs = np.linspace(0, 1.05, 106)
    for m, col in (("8B pair", C_PAIR), ("8B aug", C_AUG)):
        src, tag = A.METHODS[m]
        for ax, (sp, title) in zip(axes, panels):
            S = A.Split(sps[0] if sp == "val_calib" else sp); seeds = A.load_seeds(src, tag, S.sp); d = []
            for (mp, tau, sd), (sc, *_) in zip(seeds, A.seed_arrays(S, seeds)):
                sc = P.to_logit(sc); sep = np.median(sc[S.y == 1]) - np.median(sc[S.y == 0])
                D = P.val_deltas(src, tag, sd) if sp == "val_calib" else P.deltas(S, sc)[0]
                d += list(D / sep)
            d = np.asarray(d)
            ax.plot(xs, [100 * np.mean(d > t) for t in xs], color=col, lw=1.2, label=LEG.get(m, m))
            ax.set_title(title)
    for ax in axes:
        ax.set_yscale("log"); ax.set_ylim(0.3, 100); ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.5, 1.0]); ax.set_xlabel(r"$\delta$ / class separation")
    axes[0].set_ylabel(r"$\Pr[\Delta>\delta]$ (%)"); axes[0].legend(frameon=False)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_shift.pdf")); plt.close(fig)


def fig_scaling():
    sp = FIG_SPLITS[-1]; sizes = [s for s in ("1.7B", "4B", "8B") if f"{s} pair" in A.METHODS and A.load_seeds(*A.METHODS[f"{s} pair"], sp)]
    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.45))
    for loss, col in (("pair", C_PAIR), ("aug", C_AUG)):
        xs, f3, w5 = [], [], []
        for i, s in enumerate(sizes):
            S = A.Split(sp); src, tag = A.METHODS[f"{s} {loss}"]; seeds = A.load_seeds(src, tag, sp)
            if not seeds:
                continue
            ar = A.seed_arrays(S, seeds); ms = [A.metrics(S, sc, tau, t5=t5) for sc, tau, t5, _ in ar]
            xs.append(i); f3.append((np.mean([m["fpr_hE3"] for m in ms]), np.std([m["fpr_hE3"] for m in ms])))
            w5.append((np.mean([m["wg_t5"] for m in ms]), np.std([m["wg_t5"] for m in ms])))
        for ax, v in zip(axes, (f3, w5)):
            v = np.array(v); ax.errorbar(xs, v[:, 0], v[:, 1], color=col, marker="o", ms=3, lw=1, capsize=1.5, label=loss)
    for ax, t in zip(axes, ("human-E3 FPR (%)", "WG at 5% FPR (%)")):
        ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes); ax.set_ylabel(t); ax.set_xlabel(f"backbone ({NAMES[sp]})")
    axes[0].axhline(5, color="0.6", lw=0.5, ls="--"); axes[0].legend(frameon=False)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_scaling.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_cells(); fig_shift(); fig_scaling(); print("figures in", OUT)

#!/usr/bin/env python3
"""Paper tables and figures for the reframed v3 paper (two-sided editing study).

Reads results_v3/master_evaluation_summary_v3_final.json (138 runs, anchored on paired
training) and writes paper/tables/t_*.tex and paper/figures/f_*.pdf.
Run:  uv run --with numpy --with scipy --with matplotlib python scripts/v3_paper_assets.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.environ.get("CEDAR_SUMMARY", os.path.join(ROOT, "results_v3", "master_evaluation_summary_v3_final.json"))
_OUT = os.environ.get("CEDAR_ASSET_DIR", ROOT)
TAB = os.path.join(_OUT, "paper", "tables")
FIG = os.path.join(_OUT, "paper", "figures")
OUT_JSON = os.path.join(_OUT, "results_v3", "paper_numbers.json")

D = json.load(open(SUMMARY))
PS = D["trained_per_seed"]
PT = D["pretrained_and_probes"]
TESTS = dict(D["paired_significance_tests"])
TESTS.update(D.get("nsd_and_dte_effect_tests", {}))
_E1 = os.path.join(ROOT, "results_v3", "e1_analysis.json")
_KMAP = {"tc_wg": "test_confirm_worst_group_E0_E3", "tid_wg": "test_id_worst_group_E0_E3", "tc_auc": "test_confirm_auroc",
         "ms_auc": "external_mixset_pure_auroc", "ll_lm_auc": "test_ood_llada_len120_220_auroc", "bm_auc": "external_beemo_auroc"}
if os.path.exists(_E1):
    for _k, _v in json.load(open(_E1))["paired_tests"].items():
        TESTS.setdefault(_k.replace("__vs__", "_vs_"), {_KMAP.get(a, a): b for a, b in _v.items()})

# ----------------------------------------------------------------------------- metric access

def fit_width(text):
    """Scale a table body to the text width (used for the wide contrast/factorial/ablation tables)."""
    return text.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}}", 1)

def seeds_of(run):
    out = sorted([k for k in PS if k.rsplit("_seed", 1)[0] == run], key=lambda k: int(k.rsplit("_seed", 1)[1]))
    return [PS[k] for k in out]


def g(rec, path):
    x = rec
    for p in path.split("/"):
        if x is None:
            return None
        x = x.get(p) if isinstance(x, dict) else None
    return x


M = {  # name -> path inside one evaluated record (trained seed record or pretrained record)
    "c_auc": "test_confirm/overall_auroc",
    "c_wg": "test_confirm/worst_group_acc_E0_E3",
    "c_fpr": "test_confirm/human_fpr_tau_all5",
    "c_tpr": "test_confirm/machine_tpr_tau_all5",
    "c_unseen_wg": "test_confirm/unseen_genres_wg_E0_E3",
    "id_auc": "test_id_auroc_E0_E3",
    "id_wg": "test_id_worst_group_acc_E0_E3",
    "ar_auc": "test_ood_ar/overall_auroc",
    "ll_auc": "test_ood_llada/overall_auroc",
    "ll_len": "test_ood_llada/len120_220_auroc",
    "ms_auc": "external_mixset/pure_origin_auroc",
    "ms_pol_fpr": "external_mixset/hwt_llm_polished_fpr_tau_all5",
    "bm_auc": "external_beemo/overall_auroc",
    "bm_exp_auc": "external_beemo/auroc_expert_edited_vs_human",
    "esl1_fpr": "external_esl_fairness/esl_E1_polished_fpr_at_tau_all5",
    "esl1_fpr_e0": "external_esl_fairness/esl_E1_polished_fpr_at_tau_e0_5",
    "mw_auc": "external_mage_wild/overall_auroc",
    "vsel": "best_val_select/val_select_worst_edit_bal_acc",
}


def vals(run, key):
    if run in PT:
        v = g(PT[run], M[key])
        return [] if v is None else [float(v)]
    return [float(g(r, M[key])) for r in seeds_of(run) if g(r, M[key]) is not None]


def ms(run, key):
    v = vals(run, key)
    if not v:
        return None, None, 0
    return float(np.mean(v)) * 100, float(np.std(v, ddof=1)) * 100 if len(v) > 1 else None, len(v)


def lvl_fpr(rec, lvl, cal):
    return g(rec, f"test_id_recoverability/{lvl}/fpr_at_tau_{cal}")


# ----------------------------------------------------------------------------- display names
PRETRAINED = [
    ("fast_detectgpt_qwen3b", "Fast-DetectGPT"),
    ("binoculars_qwen3b", "Binoculars"),
    ("radar_vicuna7b", "RADAR"),
    ("hc3_roberta_detector", "ChatGPT-Det.\\ (HC3)"),
    ("desklib_raid_v1", "Desklib"),
]
MB_BASE = [
    ("mb_v3_unedited_erm", "Unedited ERM"),
    ("mb_v3_augmented_erm", "Augmented ERM"),
    ("mb_v3_joint_8class_ce", "Joint 8-class CE"),
    ("mb_v3_multitask_2head", "Multi-task 2-head"),
    ("mb_v3_supcon_orth_disentangle", "SupCon + orth."),
    ("mb_v3_erm_group_dro", "ERM + Group-DRO"),
    ("mb_v3_erm_paired_consist", "ERM + pair consist."),
]
MB_OURS = [
    ("mb_v3_strong_combined_no_nsd", "\\textsc{Spot}"),
    ("mb_v3_ablation_no_dte", "\\textsc{Spot}-PD"),
]
DB_ROWS = [
    ("db_v3_unedited_erm", "Unedited ERM"),
    ("db_v3_augmented_erm", "Augmented ERM"),
    ("db_v3_supcon_orth_disentangle", "SupCon + orth."),
    ("db_v3_erm_paired_consist", "ERM + pair consist."),
    ("db_v3_strong_combined_no_nsd", "\\textsc{Spot}"),
    ("db_v3_cedar_no_dte", "\\textsc{Spot}-PD"),
]


def fmt(m, s, bold=False, sd=True):
    if m is None:
        return "--"
    core = f"{m:.1f}"
    if bold:
        core = f"\\textbf{{{core}}}"
    if sd and s is not None:
        return f"{core}\\std{{{s:.1f}}}"
    return core


# ----------------------------------------------------------------------------- Table: main
MAIN_COLS = [
    ("c_auc", "AUROC", True), ("c_wg", "WG", True), ("c_fpr", "FPR", False),
    ("id_wg", "ID WG", True), ("ar_auc", "AR", True), ("ll_len", "Diff.", True),
    ("ms_auc", "MixSet", True), ("bm_auc", "Beemo", True), ("esl1_fpr", "ESL FPR", False),
]


def best_by_col(runs):
    best = {}
    for k, _, hi in MAIN_COLS:
        xs = [(ms(r, k)[0], r) for r in runs if ms(r, k)[0] is not None]
        if not xs:
            continue
        best[k] = (max if hi else min)(xs)[1]
    return best


def main_table():
    lines = []
    lines.append("\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Main results} (\\%, mean\\std{SD} over 5 seeds for trained models). "
                 "\\textit{Confirm}: unseen generator, editor and 3 genres. WG: worst-group accuracy over the 8 (origin, edit) cells and FPR: "
                 "cluster-averaged human FPR, both at the \\texttt{val\\_calib} threshold ($\\alpha=5\\%$). AR / Diff.: AUROC on unseen "
                 "\\textsc{OLMo-2} / \\textsc{LLaDA} (120--220 words) drafts. ESL FPR: polished IELTS essays. \\textsc{Spot} includes DTE; "
                 "\\textsc{Spot}-PD has no DTE and 600 training parents (Table~\\ref{tab:config}). Bold: best trained value per column.}\n"
                 "\\label{tab:main}\n\\small\n\\setlength{\\tabcolsep}{2.2pt}\n"
                 "\\begin{tabular}{l ccc c cc ccc}\n\\toprule\n"
                 "& \\multicolumn{3}{c}{\\textit{Confirm}} & \\textit{In-dist.} & \\multicolumn{2}{c}{\\textit{Crossed OOD}} & "
                 "\\multicolumn{3}{c}{\\textit{External}} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}\n"
                 "Method & AUROC$\\uparrow$ & WG$\\uparrow$ & FPR$\\downarrow$ & WG$\\uparrow$ & AR$\\uparrow$ & Diff.$\\uparrow$ & "
                 "MixSet$\\uparrow$ & Beemo$\\uparrow$ & ESL FPR$\\downarrow$ \\\\\n\\midrule\n")
    lines.append("\\multicolumn{10}{l}{\\textit{Pretrained / zero-shot detectors}} \\\\\n")
    for r, n in PRETRAINED:
        cells = [fmt(ms(r, k)[0], None, sd=False) for k, _, _ in MAIN_COLS]
        lines.append(n + " & " + " & ".join(cells) + " \\\\\n")
    for title, rows in (("\\textsc{ModernBERT-large}, trained on our symmetric octets", MB_BASE + MB_OURS),
                        ("\\textsc{DeBERTa-v3-large}", DB_ROWS)):
        lines.append("\\midrule\n\\multicolumn{10}{l}{" + title + "} \\\\\n")
        best = best_by_col([r for r, _ in rows])
        for r, n in rows:
            if r in dict(MB_OURS) or r in ("db_v3_strong_combined_no_nsd", "db_v3_cedar_no_dte"):
                pass
            cells = []
            for k, _, _ in MAIN_COLS:
                m, s, _n = ms(r, k)
                cells.append(fmt(m, s, bold=(best.get(k) == r), sd=k in ("c_wg", "id_wg", "ll_len")))
            pre = "\\rowcolor{gray!10}" if "Spot" in n else ""
            lines.append(pre + n + " & " + " & ".join(cells) + " \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(TAB, "t_main.tex"), "w").write("".join(lines))


# ----------------------------------------------------------------------------- Table: factorial
FACT = [
    # (run, DTE, PD, N_fit)
    ("mb_v3_strong_combined_no_nsd", "\\cmark", "--", "800"),
    ("mb_v3_strong_combined_fit600_no_nsd", "\\cmark", "--", "600"),
    ("mb_v3_cedar_split_op", "\\cmark", "\\cmark", "600+200"),
    ("mb_v3_cedar_full800", "\\cmark", "\\cmark", "800"),
    ("mb_v3_nodte_no_nsd", "--", "--", "800"),
    ("mb_v3_nodte_fit600_no_nsd", "--", "--", "600"),
    ("mb_v3_ablation_no_dte", "--", "\\cmark", "600+200"),
    ("mb_v3_ablation_unprotected_nsd", "\\cmark", "$\\lambda{=}0$", "600+200"),
]
FACT_DB = [
    ("db_v3_strong_combined_no_nsd", "\\cmark", "--", "800"),
    ("db_v3_cedar_split_op", "\\cmark", "\\cmark", "600+200"),
    ("db_v3_nodte_no_nsd", "--", "--", "800"),
    ("db_v3_cedar_no_dte", "--", "\\cmark", "600+200"),
]
FCOLS = [("vsel", "Val"), ("c_wg", "Confirm WG"), ("id_wg", "ID WG"), ("ll_len", "Diff."), ("ms_auc", "MixSet")]


def factorial_table():
    L = ["\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Factorial over the paired recipe} (\\%, mean\\std{SD}; $n$ seeds). "
         "DTE: discourse-trajectory branch. PD: protected deflation ($\\lambda_{\\mathrm{prot}}{=}0.5$, rank 24, operator fitted on "
         "200 held-out parents). $N$: training parents. Val: \\texttt{val\\_select} worst-edit balanced accuracy. "
         "Diff.: \\textsc{LLaDA} AUROC on 120--220-word passages.}\n\\label{tab:factorial}\n\\small\n"
         "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l ccc c ccccc}\n\\toprule\n"
         "Backbone & DTE & PD & $N$ & $n$ & Val & Confirm WG & ID WG & Diff. & MixSet \\\\\n\\midrule\n"]
    for bb, rows in (("MB", FACT), ("DB", FACT_DB)):
        for i, (r, dte, pd, n) in enumerate(rows):
            nseed = ms(r, "c_wg")[2]
            cells = [fmt(*ms(r, k)[:2]) for k, _ in FCOLS]
            L.append(("\\textsc{ModernBERT}" if bb == "MB" else "\\textsc{DeBERTa}") * (i == 0) +
                     f" & {dte} & {pd} & {n} & {nseed} & " + " & ".join(cells) + " \\\\\n")
        if bb == "MB":
            L.append("\\midrule\n")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(TAB, "t_factorial.tex"), "w").write(fit_width("".join(L)))


# ----------------------------------------------------------------------------- Table: ablations (appendix)
ABL = [
    ("mb_v3_cedar_split_op", "Full (DTE + PD, split operator)"),
    ("mb_v3_ablation_ema_only", "PD from online EMA only (no held-out operator)"),
    ("mb_v3_ablation_unweighted_pair", "Unweighted pair alignment"),
    ("mb_v3_ablation_no_xmargin", "No within-parent cross-margin"),
    ("mb_v3_ablation_param_matched_mlp", "DTE $\\to$ parameter-matched MLP"),
    ("mb_v3_ablation_no_dte", "No DTE"),
    ("mb_v3_cedar_rank8", "Deflation rank 8"),
    ("mb_v3_cedar_rank48", "Deflation rank 48"),
    ("mb_v3_ablation_unprotected_nsd", "Unprotected deflation ($\\lambda_{\\mathrm{prot}}{=}0$)"),
]


def ablation_table():
    L = ["\\begin{table}[h]\n\\centering\n\\caption{Component ablations of the deflation variant on \\textsc{ModernBERT} (\\%, mean\\std{SD}).}\n"
         "\\label{tab:app_ablation}\n\\small\n\\begin{tabular}{l c ccccc}\n\\toprule\n"
         "Variant & $n$ & Val & Confirm WG & ID WG & Diff. & MixSet \\\\\n\\midrule\n"]
    for r, n in ABL:
        L.append(n + f" & {ms(r, 'c_wg')[2]} & " + " & ".join(fmt(*ms(r, k)[:2]) for k, _ in FCOLS) + " \\\\\n")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(TAB, "t_app_ablation.tex"), "w").write(fit_width("".join(L)))


# ----------------------------------------------------------------------------- paired tests table
def find_test(a, b):
    for key in (f"{a}_vs_{b}",):
        if key in TESTS:
            return TESTS[key], 1
    key = f"{b}_vs_{a}"
    if key in TESTS:
        return TESTS[key], -1
    return None, 0


CONTRASTS = [
    ("Paired vs.\\ augmented training", "mb_v3_strong_combined_no_nsd", "mb_v3_augmented_erm"),
    ("PD effect, no DTE, $N{=}600$ (MB)", "mb_v3_ablation_no_dte", "mb_v3_nodte_fit600_no_nsd"),
    ("PD effect, with DTE, $N{=}600$ (MB)", "mb_v3_cedar_split_op", "mb_v3_strong_combined_fit600_no_nsd"),
    ("PD effect, no DTE (DB)", "db_v3_cedar_no_dte", "db_v3_nodte_no_nsd"),
    ("DTE effect, no PD, $N{=}800$ (MB)", "mb_v3_strong_combined_no_nsd", "mb_v3_nodte_no_nsd"),
    ("DTE effect, no PD, $N{=}800$ (DB)", "db_v3_strong_combined_no_nsd", "db_v3_nodte_no_nsd"),
    ("DTE effect, no PD, $N{=}600$ (MB)", "mb_v3_strong_combined_fit600_no_nsd", "mb_v3_nodte_fit600_no_nsd"),
    ("Protection ($\\lambda{=}0.5$ vs.\\ 0)", "mb_v3_cedar_split_op", "mb_v3_ablation_unprotected_nsd"),
]
EXTRA = [("SPOT vs pair", "mb_v3_strong_combined_no_nsd", "mb_v3_erm_paired_consist"),
         ("SPOT vs GDRO", "mb_v3_strong_combined_no_nsd", "mb_v3_erm_group_dro"),
         ("SPOT vs SupCon", "mb_v3_strong_combined_no_nsd", "mb_v3_supcon_orth_disentangle"),
         ("SPOT vs aug DB", "db_v3_strong_combined_no_nsd", "db_v3_augmented_erm"),
         ("SPOT vs SupCon DB", "db_v3_strong_combined_no_nsd", "db_v3_supcon_orth_disentangle"),
         ("PD vs SPOT MB", "mb_v3_ablation_no_dte", "mb_v3_strong_combined_no_nsd"),
         ("PD vs SPOT DB", "db_v3_cedar_no_dte", "db_v3_strong_combined_no_nsd")]
TCOLS = [("test_confirm_worst_group_E0_E3", "Confirm WG"), ("test_id_worst_group_E0_E3", "ID WG"),
         ("test_ood_llada_len120_220_auroc", "Diff."), ("external_mixset_pure_auroc", "MixSet")]


def tests_table(numbers):
    L = ["\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Paired contrasts} (A$-$B, percentage points) with 95\\% parent-cluster "
         "bootstrap CIs (5{,}000 resamples, seed-averaged scores). Bold: CI excludes 0.}\n\\label{tab:contrasts}\n\\small\n"
         "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{l cccc}\n\\toprule\n"
         "Contrast (A vs.\\ B) & " + " & ".join(n for _, n in TCOLS) + " \\\\\n\\midrule\n"]
    numbers["contrasts"] = {}
    for name, a, b in CONTRASTS + EXTRA:
        t, sgn = find_test(a, b)
        cells = []
        rec = {}
        for k, _ in TCOLS:
            if t is None or k not in t:
                cells.append("--")
                continue
            dlt = sgn * t[k]["diff"] * 100
            lo, hi = sorted([sgn * t[k]["ci95"][0] * 100, sgn * t[k]["ci95"][1] * 100])
            sig = lo > 0 or hi < 0
            s = f"{dlt:+.1f} [{lo:+.1f}, {hi:+.1f}]"
            rec[k] = [round(dlt, 2), round(lo, 2), round(hi, 2)]
            cells.append(f"\\textbf{{{s}}}" if sig else s)
        numbers["contrasts"][f"{a}__vs__{b}"] = rec
        if (name, a, b) in CONTRASTS:
            L.append(name + " & " + " & ".join(cells) + " \\\\\n")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(TAB, "t_contrasts.tex"), "w").write(fit_width("".join(L)))


# ----------------------------------------------------------------------------- figures
def figures(numbers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    C_PRE, C_BASE, C_OURS, C_PD, C_AUG = "#8c8c8c", "#4c78a8", "#e45756", "#f58518", "#72b7b2"

    # ---------------- Figure 1
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.1, 1.0, 1.05]})
    a = ax[0]
    lab = {"desklib_raid_v1": ("Desklib", (-30, 5)), "radar_vicuna7b": ("RADAR", (4, 2)),
           "hc3_roberta_detector": ("HC3", (4, 2)), "fast_detectgpt_qwen3b": ("Fast-DetectGPT", (-58, 4)),
           "binoculars_qwen3b": ("Binoculars", (3, -9)), "mb_v3_unedited_erm": ("Unedited ERM", (-50, -3)),
           "mb_v3_augmented_erm": ("Augmented ERM", (-58, -2)), "mb_v3_strong_combined_no_nsd": ("SPOT", (-22, 2))}
    for r, _ in PRETRAINED:
        a.scatter(ms(r, "c_auc")[0], ms(r, "c_wg")[0], c=C_PRE, marker="s", s=18, zorder=3, edgecolors="none")
    for r, _ in MB_BASE:
        a.scatter(ms(r, "c_auc")[0], ms(r, "c_wg")[0], c=C_BASE, marker="o", s=16, zorder=3, edgecolors="none")
    a.scatter(ms("mb_v3_strong_combined_no_nsd", "c_auc")[0], ms("mb_v3_strong_combined_no_nsd", "c_wg")[0], c=C_OURS,
              marker="*", s=70, zorder=4, edgecolors="none")
    for r, (n, off) in lab.items():
        a.annotate(n, (ms(r, "c_auc")[0], ms(r, "c_wg")[0]), xytext=off, textcoords="offset points", fontsize=6.3)
    a.scatter([], [], c=C_PRE, marker="s", s=14, label="pretrained / zero-shot")
    a.scatter([], [], c=C_BASE, marker="o", s=14, label="trained on SymEdit")
    a.legend(frameon=False, loc="upper left", handletextpad=0.1, borderaxespad=0.0)
    a.set_xlabel("Confirm AUROC (%)")
    a.set_ylabel("Confirm worst-group acc. (%)")
    a.set_title("(a) AUROC hides worst-group failure")
    a.set_ylim(-4, 92)

    b = ax[1]
    lv = ["E0", "E1", "E2", "E3"]
    for r, n, c in (("mb_v3_unedited_erm", "Unedited ERM", C_BASE), ("mb_v3_augmented_erm", "Augmented ERM", C_AUG),
                    ("mb_v3_strong_combined_no_nsd", "SPOT", C_OURS)):
        recs = seeds_of(r)
        e0 = [np.mean([lvl_fpr(x, l, "e0_5") for x in recs]) * 100 for l in lv]
        al = [np.mean([lvl_fpr(x, l, "all5") for x in recs]) * 100 for l in lv]
        b.plot(range(4), e0, "-o", c=c, ms=3, lw=1.2, label=n)
        b.plot(range(4), al, "--", c=c, lw=1.0, alpha=0.9)
        numbers.setdefault("calib_inflation", {})[r] = {"tau_e0": [round(v, 1) for v in e0], "tau_all": [round(v, 1) for v in al]}
    b.plot([], [], "-", c="k", lw=1.0, label="$\\tau_{E_0}$ (unedited humans)")
    b.plot([], [], "--", c="k", lw=1.0, label="$\\tau_{\\mathrm{cal}}$ (edited clusters)")
    b.axhline(5, c="k", lw=0.6, ls=":")
    b.set_xticks(range(4), ["$E_0$", "$E_1$", "$E_2$", "$E_3$"])
    b.set_ylabel("Human FPR on test\\_id (%)")
    b.set_xlabel("Edit level of the human text")
    b.set_title("(b) $E_0$ calibration inflates FPR")
    b.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.0)

    c = ax[2]
    ladder = [("mb_v3_unedited_erm", "Unedited"), ("mb_v3_augmented_erm", "Augm."),
              ("mb_v3_erm_group_dro", "+GDRO"), ("mb_v3_erm_paired_consist", "+Pair"),
              ("mb_v3_strong_combined_no_nsd", "SPOT")]
    for i, (r, n) in enumerate(ladder):
        v = np.array(vals(r, "c_wg")) * 100
        col = C_OURS if n == "SPOT" else C_BASE
        c.scatter(np.full(len(v), i) + np.linspace(-0.14, 0.14, len(v)), v, s=8, c=col, zorder=3, edgecolors="none", alpha=0.8)
        c.hlines(v.mean(), i - 0.28, i + 0.28, colors=col, lw=1.6)
        c.text(i + 0.32, v.mean(), f"{v.mean():.0f}", ha="left", va="center", fontsize=6.3)
    c.set_xticks(range(len(ladder)), [n for _, n in ladder], fontsize=6.3)
    c.set_xlim(-0.5, len(ladder) - 0.2)
    c.set_ylabel("Confirm worst-group acc. (%)")
    c.set_title("(c) Using the octet structure")
    c.set_ylim(10, 88)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(FIG, "f_hero.pdf"))
    plt.close(fig)

    # ---------------- Figure 2: deflation trade-off (means +- SD), arrows from matched no-PD to PD
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.35))
    groups = {
        "MB": dict(points=[("mb_v3_augmented_erm", "Augmented ERM", C_AUG, "s"),
                           ("mb_v3_strong_combined_no_nsd", "SPOT", C_OURS, "o"),
                           ("mb_v3_strong_combined_fit600_no_nsd", "SPOT, $N$=600", "#f2a2a1", "o"),
                           ("mb_v3_nodte_no_nsd", "SPOT w/o DTE", C_BASE, "o"),
                           ("mb_v3_nodte_fit600_no_nsd", "SPOT w/o DTE, $N$=600", "#9ecae9", "o"),
                           ("mb_v3_cedar_split_op", "+PD", "#b279a2", "^"),
                           ("mb_v3_ablation_no_dte", "+PD (w/o DTE)", C_PD, "^")],
                   arrows=[("mb_v3_nodte_fit600_no_nsd", "mb_v3_ablation_no_dte"),
                           ("mb_v3_strong_combined_fit600_no_nsd", "mb_v3_cedar_split_op")]),
        "DB": dict(points=[("db_v3_augmented_erm", "Augmented ERM", C_AUG, "s"),
                           ("db_v3_strong_combined_no_nsd", "SPOT", C_OURS, "o"),
                           ("db_v3_nodte_no_nsd", "SPOT w/o DTE", C_BASE, "o"),
                           ("db_v3_cedar_split_op", "+PD", "#b279a2", "^"),
                           ("db_v3_cedar_no_dte", "+PD (w/o DTE)", C_PD, "^")],
                   arrows=[("db_v3_nodte_no_nsd", "db_v3_cedar_no_dte"), ("db_v3_strong_combined_no_nsd", "db_v3_cedar_split_op")]),
    }
    for j, (bb, gdef) in enumerate(groups.items()):
        a = ax[j]
        for r, n, col, mk in gdef["points"]:
            x = np.array(vals(r, "id_wg")) * 100
            y = np.array(vals(r, "ll_len")) * 100
            if len(x) == 0:
                continue
            a.errorbar(x.mean(), y.mean(), xerr=x.std(ddof=1) / np.sqrt(len(x)), yerr=y.std(ddof=1) / np.sqrt(len(y)),
                       fmt=mk, color=col, ms=6, mec="k", mew=0.4, elinewidth=0.8, capsize=1.5, label=f"{n} ({len(x)})", zorder=4)
        for r0, r1 in gdef["arrows"]:
            x0, y0 = ms(r0, "id_wg")[0], ms(r0, "ll_len")[0]
            x1, y1 = ms(r1, "id_wg")[0], ms(r1, "ll_len")[0]
            if None in (x0, y0, x1, y1):
                continue
            a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="k", lw=0.8, shrinkA=4, shrinkB=4))
        a.set_xlabel("In-distribution worst-group acc. (%)")
        a.set_ylabel("Unseen diffusion-draft AUROC (%)")
        a.set_title("(a) ModernBERT-large" if bb == "MB" else "(b) DeBERTa-v3-large")
        a.legend(frameon=False, loc="lower left", fontsize=5.8, handletextpad=0.2, borderaxespad=0.1, ncol=1)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(os.path.join(FIG, "f_tradeoff.pdf"))
    plt.close(fig)

    # ---------------- Figure 3: seed instability + validation/confirm agreement over configurations
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.15), gridspec_kw={"width_ratios": [1.45, 1.0]})
    a = ax[0]
    strip = [("mb_v3_strong_combined_no_nsd", "SPOT"), ("mb_v3_nodte_no_nsd", "SPOT\nw/o DTE"),
             ("mb_v3_ablation_no_dte", "+PD\nw/o DTE"), ("mb_v3_cedar_split_op", "+PD\n(DTE)"),
             ("mb_v3_cedar_full800", "+PD (DTE)\n$N$=800"), ("mb_v3_ablation_unprotected_nsd", "+PD\n$\\lambda$=0")]
    for i, (r, n) in enumerate(strip):
        v = np.array(vals(r, "c_wg")) * 100
        a.scatter(np.full(len(v), i) + np.linspace(-0.13, 0.13, max(len(v), 1)), v, s=11,
                  c=C_OURS if i == 0 else (C_PD if "PD" in n else C_BASE), edgecolors="none", zorder=3)
        if len(v):
            a.hlines(np.median(v), i - 0.25, i + 0.25, colors="k", lw=1.0)
    a.set_xticks(range(len(strip)), [n for _, n in strip], fontsize=6.3)
    a.set_ylabel("Confirm WG (%)")
    a.set_title("(a) Per-seed Confirm WG, ModernBERT (bar: median)")
    b = ax[1]
    xs, ys, names = [], [], []
    for r in sorted({k.rsplit("_seed", 1)[0] for k in PS}):
        if not (r.startswith("mb_v3_") or r.startswith("db_v3_")):
            continue
        v, cw = ms(r, "vsel")[0], ms(r, "c_wg")[0]
        if v is None or cw is None:
            continue
        xs.append(v); ys.append(cw); names.append(r)
    xs, ys = np.array(xs), np.array(ys)
    rho = spearmanr(xs, ys).statistic
    from scipy.stats import pearsonr
    pr = pearsonr(xs, ys).statistic
    numbers["val_confirm_pearson_configs"] = round(float(pr), 3)
    top = xs >= 75
    numbers["val_confirm_spearman_configs_val_ge75"] = round(float(spearmanr(xs[top], ys[top]).statistic), 3)
    numbers["n_configs_val_ge75"] = int(top.sum())
    cols = [C_OURS if "strong_combined_no_nsd" in n else (C_BASE if n.startswith("mb") else C_AUG) for n in names]
    b.scatter(xs, ys, s=14, c=cols, edgecolors="none")
    b.scatter([], [], c=C_BASE, s=12, label="ModernBERT configs")
    b.scatter([], [], c=C_AUG, s=12, label="DeBERTa configs")
    b.scatter([], [], c=C_OURS, s=12, label="SPOT")
    b.legend(frameon=False, loc="lower right", handletextpad=0.1)
    b.set_xlabel("val\\_select score, mean over seeds (%)")
    b.set_ylabel("Confirm WG, mean (%)")
    b.set_title(f"(b) Validation vs. Confirm (Pearson $r$={pr:.2f})")
    numbers["val_confirm_spearman_configs"] = round(float(rho), 3)
    numbers["n_configs_val_confirm"] = int(len(xs))
    vr, cr = [], []
    for k in PS:
        if k.startswith("mb_v3_") or k.startswith("db_v3_"):
            v1, c1 = g(PS[k], M["vsel"]), g(PS[k], M["c_wg"])
            if v1 is not None and c1 is not None:
                vr.append(v1); cr.append(c1)
    numbers["val_confirm_spearman_runs"] = round(float(spearmanr(vr, cr).statistic), 3)
    numbers["n_runs_val_confirm"] = len(vr)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(os.path.join(FIG, "f_seeds.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------- appendix: calibration & E4
def calib_table(numbers):
    crc = D.get("few_shot_human_anchor_crc", {})
    names = [("mb_v3_strong_combined_no_nsd", "\\textsc{Spot} (MB)"), ("mb_v3_augmented_erm", "Augmented ERM (MB)"),
             ("db_v3_strong_combined_no_nsd", "\\textsc{Spot} (DB)"), ("desklib_raid_v1", "Desklib")]
    bms = [("test_confirm", "\\textit{Confirm}"), ("external_esl_full", "IELTS"), ("external_mixset_pure", "\\textsc{MixSet}"),
           ("external_beemo", "\\textsc{Beemo}"), ("external_mage_wild", "\\textsc{MAGE} wild")]
    L = ["\\begin{table}[h]\n\\centering\n\\caption{\\textbf{Human false-positive rate (\\%) at target $\\alpha=5\\%$.} "
         "Zero-shot: threshold locked on \\texttt{val\\_calib}. Anchor: recalibrated on $K{=}30$ human-only parent clusters of the target "
         "domain (mean over 200 random draws; p90 in parentheses; remaining clusters evaluated). Machine TPR after the anchor is shown in brackets.}\n"
         "\\label{tab:app_calib}\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{l " + "c" * len(bms) + "}\n\\toprule\n"
         "Model / threshold & " + " & ".join(n for _, n in bms) + " \\\\\n\\midrule\n"]
    numbers["calibration"] = {}
    for r, n in names:
        if r not in crc:
            continue
        z, a = [], []
        for b, _ in bms:
            x = crc[r]["benchmarks"].get(b)
            if x is None:
                z.append("--"); a.append("--"); continue
            zz = x["zero_shot_locked_val_calib"]["cluster_human_fpr"]
            zz = zz["mean"] if isinstance(zz, dict) else zz
            aa = x["few_shot_K30_human_anchor_crc"]
            fm = aa["marginal_cluster_human_fpr"]["mean"] * 100
            p90 = aa["p90_cluster_human_fpr_across_draws"]["mean"] * 100
            tp = aa.get("machine_tpr")
            tps = "" if not tp else f" [{tp['mean'] * 100:.0f}]"
            z.append(f"{zz * 100:.1f}")
            a.append(f"{fm:.1f} ({p90:.1f}){tps}")
            numbers["calibration"].setdefault(r, {})[b] = {"zero_shot": round(zz * 100, 2), "anchor": round(fm, 2), "anchor_p90": round(p90, 2),
                                                            "anchor_tpr": None if not tp else round(tp["mean"] * 100, 1)}
        L.append(n + ", zero-shot & " + " & ".join(z) + " \\\\\n")
        L.append(n + ", anchor & " + " & ".join(a) + " \\\\\n")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(TAB, "t_app_calib.tex"), "w").write("".join(L))


def e4_table(numbers):
    rows = [(r, n) for r, n in PRETRAINED] + [("mb_v3_unedited_erm", "Unedited ERM"), ("mb_v3_augmented_erm", "Augmented ERM"),
                                              ("mb_v3_strong_combined_no_nsd", "\\textsc{Spot} (MB)"), ("db_v3_strong_combined_no_nsd", "\\textsc{Spot} (DB)")]
    L = ["\\begin{table}[h]\n\\centering\n\\caption{\\textbf{Recoverability by edit level} on \\texttt{test\\_id}: AUROC (\\%) per level and "
         "human FPR at $E_4$ under the two calibration rules. Every detector is near chance at $E_4$ (summary regeneration).}\n"
         "\\label{tab:app_levels}\n\\small\n\\begin{tabular}{l ccccc cc}\n\\toprule\n"
         "Detector & $E_0$ & $E_1$ & $E_2$ & $E_3$ & $E_4$ & $E_4$ FPR $\\tau_{\\mathrm{cal}}$ & $E_4$ FPR $\\tau_{E_0}$ \\\\\n\\midrule\n"]
    numbers["levels"] = {}
    for r, n in rows:
        recs = [PT[r]] if r in PT else seeds_of(r)
        aucs = [np.mean([x["test_id_recoverability"][l]["auroc"] for x in recs]) * 100 for l in ["E0", "E1", "E2", "E3", "E4"]]
        f_all = np.mean([x["test_id_recoverability"]["E4"]["fpr_at_tau_all5"] for x in recs]) * 100
        f_e0 = np.mean([x["test_id_recoverability"]["E4"]["fpr_at_tau_e0_5"] for x in recs]) * 100
        numbers["levels"][r] = [round(a, 1) for a in aucs] + [round(f_all, 1), round(f_e0, 1)]
        L.append(n + " & " + " & ".join(f"{a:.1f}" for a in aucs) + f" & {f_all:.1f} & {f_e0:.1f} \\\\\n")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(TAB, "t_app_levels.tex"), "w").write("".join(L))


def register_numbers(numbers):
    out = {}
    for r in ["mb_v3_unedited_erm", "mb_v3_augmented_erm", "mb_v3_strong_combined_no_nsd", "db_v3_unedited_erm",
              "db_v3_augmented_erm", "db_v3_strong_combined_no_nsd"]:
        recs = seeds_of(r)
        if not recs:
            continue
        out[r] = {reg: {k: round(float(np.mean([x["test_confirm"]["by_register"][reg][k] for x in recs])) * 100, 1)
                        for k in recs[0]["test_confirm"]["by_register"][reg]} for reg in recs[0]["test_confirm"]["by_register"]}
        out[r]["esl"] = {k: round(float(np.mean([x["external_esl_fairness"][k] for x in recs])) * 100, 1)
                         for k in recs[0]["external_esl_fairness"]}
        out[r]["mixset"] = {k: round(float(np.mean([x["external_mixset"][k] for x in recs])) * 100, 1)
                            for k in recs[0]["external_mixset"] if not isinstance(recs[0]["external_mixset"][k], dict)}
    numbers["register_and_external"] = out
    numbers["holm"] = D.get("holm_bonferroni_primary_family")


def dump_numbers(numbers):
    allruns = sorted({k.rsplit("_seed", 1)[0] for k in PS})
    numbers["runs"] = {}
    for r in allruns + [p for p, _ in PRETRAINED]:
        numbers["runs"][r] = {k: (None if ms(r, k)[0] is None else [round(ms(r, k)[0], 2),
                                                                    None if ms(r, k)[1] is None else round(ms(r, k)[1], 2),
                                                                    ms(r, k)[2]]) for k in M}
    numbers["n_trained_runs"] = len(PS)
    numbers["summary_file"] = os.path.relpath(SUMMARY, ROOT)
    json.dump(numbers, open(OUT_JSON, "w"), indent=1)


if __name__ == "__main__":
    numbers = {}
    main_table()
    factorial_table()
    ablation_table()
    tests_table(numbers)
    calib_table(numbers)
    e4_table(numbers)
    register_numbers(numbers)
    figures(numbers)
    dump_numbers(numbers)
    print("wrote tables, figures and", OUT_JSON)

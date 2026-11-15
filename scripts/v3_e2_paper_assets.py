#!/usr/bin/env python3
"""Confirm-2 table for the paper (paper/tables/t_confirm2.tex) and numbers (results_e2/paper_numbers_e2.json).
Inputs: results_e2/e2_analysis.json, results_e2/e2_anchor.json, results_e2/model_preds (for seed SDs)."""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(os.path.join(ROOT, "results_e2", "e2_analysis.json")))
N = json.load(open(os.path.join(ROOT, "results_e2", "e2_anchor.json")))["runs"]
R = json.load(open(os.path.join(ROOT, "data_e2", "split_test_confirm2.json")))
ids = [r["id"] for r in R]
y = np.array([r["origin_label"] for r in R])
lv = np.array([r["edit_level"] for r in R])
LV = ["E0", "E1", "E2", "E3"]

PRE = [("fast_detectgpt_qwen3b", "Fast-DetectGPT"), ("binoculars_qwen3b", "Binoculars"), ("radar_vicuna7b", "RADAR"),
       ("hc3_roberta_detector", "ChatGPT-Det.\\ (HC3)"), ("desklib_raid_v1", "Desklib")]
MB = [("unedited", "Unedited ERM"), ("aug", "Augmented ERM"), ("joint8", "Joint 8-class CE"), ("multitask", "Multi-task 2-head"),
      ("supcon", "SupCon + orth."), ("gdro", "ERM + Group-DRO"), ("pair", "ERM + pair consist."), ("spot", "\\textsc{Spot}"),
      ("spot_nodte", "\\textsc{Spot} w/o DTE"), ("gappd_split", "\\textsc{Spot}-gPD (selected)")]
DB = [("unedited", "Unedited ERM"), ("aug", "Augmented ERM"), ("supcon", "SupCon + orth."), ("pair", "ERM + pair consist."),
      ("spot", "\\textsc{Spot}"), ("spot_nodte", "\\textsc{Spot} w/o DTE"), ("gappd_split", "\\textsc{Spot}-gPD (selected)")]
GEN_AR, GEN_DIFF = "test_confirm2/auroc_gpt-oss-20b", "test_confirm2/auroc_Dream-v0-Instruct-7B"


def seed_wg_sd(run):
    v = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results_e2", "model_preds", f"{run}_seed*.json"))):
        d = json.load(open(f))
        m = dict(zip(d["row_ids"]["test_confirm2"], d["preds"]["test_confirm2"]))
        s = np.array([m[i] for i in ids])
        t = d["locked_thresholds"]["tau_cal_all5"]
        v.append(min(np.mean((s[(y == o) & (lv == e)] > t) == (o == 1)) for o in (0, 1) for e in LV) * 100)
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def row(name, m, an, sd=None, bold=()):
    def f(x, k):
        s = f"{x:.1f}"
        return f"\\textbf{{{s}}}" if k in bold else s
    wg = f(m["test_confirm2/wg"], "wg") + (f"\\std{{{sd:.1f}}}" if sd is not None else "")
    cells = [f(m["test_confirm2/auroc"], "auc"), f(m[GEN_AR], "ar"), f(m[GEN_DIFF], "diff"), wg,
             f(an["zero_shot"]["cluster_human_fpr"], "fpr"), f(m["test_confirm2/human_fpr_E3"], "fe3"),
             f(an["anchor_k30"]["wg"], "awg")]
    return f"{name} & " + " & ".join(cells) + " \\\\"


def best(keys, cfgs):
    out = {}
    for k, path, lower in [("auc", "test_confirm2/auroc", False), ("ar", GEN_AR, False), ("diff", GEN_DIFF, False),
                           ("wg", "test_confirm2/wg", False), ("fe3", "test_confirm2/human_fpr_E3", True)]:
        vals = {c: A["configs"][c][path] for c in cfgs}
        out[k] = (min if lower else max)(vals, key=vals.get)
    vals = {c: N[f"e2_{c}"]["anchor_k30"]["wg"] for c in cfgs}
    out["awg"] = max(vals, key=vals.get)
    vals = {c: N[f"e2_{c}"]["zero_shot"]["cluster_human_fpr"] for c in cfgs}
    out["fpr"] = min(vals, key=vals.get)
    return out


def main():
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{\textbf{Confirm-2}, specified before it was built (\%; trained: mean over 5 seeds, \std{SD} for WG). "
             r"Drafts by \textsc{gpt-oss-20b} (AR) or \textsc{Dream-7B} (Diff.), edits by \textsc{Yi-1.5-9B}. FPR: cluster-averaged at the "
             r"\texttt{val\_calib} threshold; FPR$_{E_3}$: restructured human text. Anchored: threshold re-locked on 30 target human parents "
             r"(post hoc, 200 draws, FPR 2.5--3.3\%). Bold: best trained value per block.}",
             r"\label{tab:confirm2}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l ccc cc c c}", r"\toprule",
             r"& \multicolumn{3}{c}{AUROC$\uparrow$} & \multicolumn{3}{c}{Locked threshold} & Anchored \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}",
             r"Method & All & AR & Diff. & WG$\uparrow$ & FPR$\downarrow$ & FPR$_{E_3}\downarrow$ & WG$\uparrow$ \\", r"\midrule",
             r"\multicolumn{8}{l}{\textit{Pretrained / zero-shot detectors}} \\"]
    for k, nm in PRE:
        lines.append(row(nm, A["pretrained"][k], N[k]))
    for bb, rows, title in (("mb", MB, r"\textsc{ModernBERT-large}"), ("db", DB, r"\textsc{DeBERTa-v3-large}")):
        lines += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{{title}}} \\"]
        cfgs = [f"{bb}_{t}" for t, _ in rows]
        b = best(None, cfgs)
        for t, nm in rows:
            c = f"{bb}_{t}"
            pre = r"\rowcolor{gray!10}" if t in ("spot", "spot_nodte", "gappd_split") else ""
            lines.append(pre + row(nm, A["configs"][c], N[f"e2_{c}"], seed_wg_sd(f"e2_{c}"), bold={k for k, v in b.items() if v == c}))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(ROOT, "paper", "tables", "t_confirm2.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[14:]))


if __name__ == "__main__":
    main()

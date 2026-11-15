#!/usr/bin/env python3
"""LaTeX tables for the ICML paper from results_icml/final_analysis[_dev][_unsealed].json.
Usage: python scripts/icml_tables.py [--dev] [--unseal]   ->  paper_icml/tables/t_main.tex, t_claims.tex"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV, UNSEAL = "--dev" in sys.argv, "--unseal" in sys.argv
RES = json.load(open(os.path.join(ROOT, "results_icml", f"final_analysis{'_dev' if DEV else ''}{'_unsealed' if UNSEAL else ''}.json")))
OUT = os.path.join(ROOT, "paper_icml", "tables"); os.makedirs(OUT, exist_ok=True)
SPL = [s for s in ("test_confirm3", "test_confirm2", "test_confirm") if s in RES["splits"]]
SNAME = {"test_confirm3": r"\textit{Confirm-3} (sealed)", "test_confirm2": r"\textit{Confirm-2}", "test_confirm": r"\textit{Confirm}"}
ROWS = [("Qwen3-8B, same data", [("8B pair", r"\textbf{Sibling-aligned (ours)}"), ("8B aug", "Augmented ERM"), ("8B SupCon", "SupCon"),
                                ("8B GroupDRO", "Group-DRO"), ("8B logit-pair", "Logit pair")]),
        ("ModernBERT-large", [("MB pair", "Sibling-aligned"), ("MB aug", "Augmented ERM")]),
        ("Pretrained, no retraining", [("fast_detectgpt_qwen3b", "Fast-DetectGPT"), ("binoculars_qwen3b", "Binoculars"), ("radar_vicuna7b", "RADAR"),
                                       ("desklib_raid_v1", "Desklib"), ("hc3_roberta_detector", "HC3 RoBERTa"), ("word_count_probe", "Word count")])]
COLS = [("wg", "WG", max), ("fpr_hE3", r"FPR$_{E_3}$", None), ("auroc", "AUROC", max), ("wg_t5", r"WG$_{5\%}$", max)]  # no bold for FPR: 0% at 0% recall is trivial


def cell(m, k, best, show_sd=True):
    if m is None:
        return "--"
    v, sd = m[k]; s = f"{v:.1f}"
    s = rf"\textbf{{{s}}}" if best is not None and abs(v - best) < 0.05 else s
    return s + (rf"\std{{{sd:.1f}}}" if show_sd and m.get("n_seeds", 1) > 1 else "")


def t_main():
    L = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
         r"\caption{Detection under symmetric editing with a threshold locked on \texttt{val\_calib} (5\% human cluster FPR). WG: worst-group accuracy over the eight (provenance, level) cells; FPR$_{E_3}$: false accusation of restructured human text; WG$_{5\%}$: WG at a threshold set to 5\% FPR on the split itself (ranking quality without operating-point transfer). SupCon: supervised contrastive loss in the style of DeTeCtive; logit pair: pair consistency on the logit. Mean (and, for WG and FPR$_{E_3}$ on \textit{Confirm-3}, standard deviation) over seeds (five for the 8B pair and Augmented-ERM rows and the encoders, three otherwise). \textit{Confirm-3} was scored once; \textit{Confirm} and \textit{Confirm-2} informed development. Best WG, AUROC and WG$_{5\%}$ per split in bold.}",
         r"\label{tab:main}", r"\begin{tabular}{l" + r"@{\hspace{10pt}}".join(["rrrr"] * len(SPL)) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{SNAME[s]}}}" for s in SPL) + r" \\",
         " ".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(SPL))),
         "Method & " + " & ".join(c[1] for _ in SPL for c in COLS) + r" \\", r"\midrule"]
    best = {(s, k): f([RES["splits"][s][n][k][0] for _, rs in ROWS for n, _ in rs if n in RES["splits"][s]]) if f else None for s in SPL for k, _, f in COLS}
    for gi, (grp, rs) in enumerate(ROWS):
        if gi:
            L.append(r"\midrule")
        L.append(rf"\multicolumn{{{1 + 4 * len(SPL)}}}{{l}}{{\textit{{{grp}}}}} \\")
        for n, lab in rs:
            if not any(n in RES["splits"][s] for s in SPL):
                continue
            L.append(lab + " & " + " & ".join(cell(RES["splits"][s].get(n), k, best[(s, k)], show_sd=(len(SPL) < 3 or s == SPL[0]) and k in ("wg", "fpr_hE3"))
                                              for s in SPL for k, _, _ in COLS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(OUT, "t_main.tex"), "w").write("\n".join(L) + "\n")


def ci(c, k):
    d = c[k]; return rf"${d['diff']:+.1f}$ & $[{d['ci95'][0]:.1f}, {d['ci95'][1]:.1f}]$"


def t_claims():
    if "test_confirm3" not in RES["splits"]:
        return
    C, sp = RES["contrasts"], "test_confirm3"; best = RES.get("best_pretrained", {}).get(sp)
    bname = dict(r for _, rs in ROWS for r in rs).get(best, best)
    rows = [("H2a", "WG vs.\\ Aug.\\ ERM", f"{sp}/8B pair vs 8B aug", "wg", 1),
            ("H2b", "WG vs.\\ ModernBERT", f"{sp}/8B pair vs MB pair", "wg", 1),
            ("H2c", f"WG vs.\\ {bname}", f"{sp}/8B pair vs {best}", "wg", 1),
            ("H3a", r"FPR$_{E_3}$ vs.\ Aug.\ ERM", f"{sp}/8B pair vs 8B aug", "fpr_hE3", -1),
            ("H3b", r"WG$_{5\%}$ vs.\ Aug.\ ERM", f"{sp}/8B pair vs 8B aug", "wg_t5", 1)]
    L = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{1.6pt}",
         r"\caption{Pre-registered \textit{Confirm-3} claims (C3-H1 to C3-H3). Each contrast is the 8B sibling-aligned detector minus the named method (Aug.\ ERM: Augmented ERM; H2c: the best pretrained detector), or, for H1, its 2025 minus its 2023 parents. Differences in percentage points with 95\% intervals from a two-level bootstrap over seeds and parents. C3-H2 and C3-H3 hold if the interval excludes 0 in the predicted direction; C3-H1 (no evidence that backbone exposure drives detection) holds if its lower end is above $-10$. ``Len.'' repeats the contrast on the length-matched subset (sign only, pre-registered as a robustness check).}",
         r"\label{tab:claims}", r"\begin{tabular}{@{}llrrcc@{}}", r"\toprule", r"& Contrast & $\Delta$ & 95\% CI & Holds & Len. \\", r"\midrule"]
    for tag, lab, key, k, sgn in rows:
        if key not in C:
            continue
        d = C[key][k]; ok = (d["ci95"][0] > 0) if sgn > 0 else (d["ci95"][1] < 0)
        lk = key.replace(sp + "/", sp + "_lenmatch/"); lm = C.get(lk, {}).get(k)
        lsign = "--" if lm is None else (r"\cmark" if lm["diff"] * sgn > 0 else r"\xmark")
        mark = r"\cmark" if ok else r"\xmark"
        L.append(f"{tag} & {lab} & {ci(C[key], k)} & {mark} & {lsign} " + r"\\")
    con = RES.get("contamination", {}).get("8B pair")
    if con:
        d = con["wg"]; ok = d["ci95"][0] > -10
        mark = r"\cmark" if ok else r"\xmark"
        L.append(rf"H1 & WG 2025 vs.\ 2023 & ${d['post_minus_pre']:+.1f}$ & $[{d['ci95'][0]:.1f}, {d['ci95'][1]:.1f}]$ & {mark} & -- \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "t_claims.tex"), "w").write("\n".join(L) + "\n")


def t_external():
    """External corpora at each run's locked threshold (no retraining, no re-calibration)."""
    import glob
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import icml_eval as EV
    import icml_final_analysis as FA
    src = "results_icml/p1" if DEV else "results_icml/final"
    tags = [("8B pair", f"{src}/{'qwen3_8b_pairx' if DEV else 'f_8b_pair'}"), ("8B aug", f"{src}/{'qwen3_8b_augx' if DEV else 'f_8b_aug'}"),
            ("8B SupCon", f"{src}/f_8b_supcon"), ("8B GroupDRO", f"{src}/f_8b_gdro"), ("8B logit-pair", f"{src}/f_8b_logitpair"),
            ("MB pair", ("results_e2/model_preds" if DEV else "results_icml/final_enc") + "/e2_mb_pair"),
            ("MB aug", ("results_e2/model_preds" if DEV else "results_icml/final_enc") + "/e2_mb_aug")]
    COL = [("external_esl_full", "fpr_E0", "FPR raw"), ("external_esl_full", "fpr_E1pol", "FPR pol."),
           ("external_mixset", "auroc_pure", "AUROC"), ("external_mixset", "fpr_llm_polished_h", "FPR pol."),
           ("external_mixset", "tpr_humanized_m", "TPR hum."), ("external_beemo", "auroc", "AUROC"),
           ("external_beemo", "tpr_expert_human_edited", "TPR exp."), ("external_mage_wild", "auroc", "AUROC")]
    res = {}
    for name, g in tags:
        files = sorted(glob.glob(os.path.join(ROOT, g + "_seed*.json")))
        if files:
            res[name] = EV.pooled(files)
    pv3 = json.load(open(os.path.join(ROOT, "results_v3/preds_pretrained_and_probes_v3.json")))
    val = json.load(open(os.path.join(ROOT, "data/split_val.json")))
    cal = set(json.load(open(os.path.join(ROOT, "data_v2/val_manifest.json")))["val_calib_parent_ids"])
    ci_ = [i for i, r in enumerate(val) if r["parent_id"] in cal and int(r["origin_label"]) == 0]
    for det in FA.PRETRAINED:
        tau = FA.conformal(np.asarray(pv3[det]["val"], float)[ci_], [val[i]["parent_id"] for i in ci_]); m = {}
        for sp in {c[0] for c in COL}:
            ids = list(EV.R[sp].keys()); m[sp] = {k: (v, 0.0, 1) for k, v in EV.split_metrics(sp, ids, pv3[det][sp], tau).items()}
        res[det] = m
    lab = dict(r for _, rs in ROWS for r in rs)
    L = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{3.5pt}",
         r"\caption{External corpora, scored at each detector's locked \texttt{val\_calib} threshold without retraining. ESL: IELTS essays by non-native writers, raw and grammar-polished by an LLM (FPR, lower is better). \textsc{MixSet}: pure-origin records; FPR on LLM-polished human text and TPR on humanised machine text. \textsc{Beemo}: AUROC over all machine texts and TPR on expert-edited machine text. \textsc{MAGE}: unseen domains. Trained rows: mean over seeds.}",
         r"\label{tab:external}", r"\begin{tabular}{l" + "r" * len(COL) + "}", r"\toprule",
         r" & \multicolumn{2}{c}{ESL} & \multicolumn{3}{c}{\textsc{MixSet}} & \multicolumn{2}{c}{\textsc{Beemo}} & \textsc{MAGE} \\",
         r"\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-8} \cmidrule(lr){9-9}",
         "Detector & " + " & ".join(c[2] for c in COL) + r" \\", r"\midrule"]
    for name in [t[0] for t in tags] + FA.PRETRAINED:
        if name not in res:
            continue
        XL = {"8B pair": r"\textbf{8B sibling-aligned (ours)}", "8B aug": "8B Augmented ERM", "8B SupCon": "8B SupCon",
              "8B GroupDRO": "8B Group-DRO", "8B logit-pair": "8B logit pair", "MB pair": "ModernBERT sibling-aligned", "MB aug": "ModernBERT Augmented ERM"}
        L.append(XL.get(name, lab.get(name, name)) + " & " +
                 " & ".join(f"{res[name][sp][k][0]:.1f}" if sp in res[name] and k in res[name][sp] else "--" for sp, k, _ in COL) + r" \\")
        if name == "MB aug":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(OUT, "t_external.tex"), "w").write("\n".join(L) + "\n")


def t_prop1():
    P = json.load(open(os.path.join(ROOT, "results_icml", f"prop1_check{'_dev' if DEV else ''}{'_unsealed' if UNSEAL else ''}.json")))
    XL = {"8B pair": r"\textbf{8B sibling-aligned}", "8B aug": "8B Augmented ERM", "8B SupCon": "8B SupCon", "8B GroupDRO": "8B Group-DRO",
          "8B logit-pair": "8B logit pair", "1.7B pair": "1.7B sibling-aligned", "1.7B aug": "1.7B Augmented ERM",
          "MB pair": "ModernBERT sibling-aligned", "MB aug": "ModernBERT Augmented ERM"}
    sps = [s_ for s_ in ["test_confirm3", "test_confirm2"] if s_ in P]
    nm = {"test_confirm3": r"\textit{Confirm-3}", "test_confirm2": r"\textit{Confirm-2}"}
    L = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{3.5pt}",
         r"\caption{Proposition~\ref{prop:transfer} on the two unseen-editor splits. FPR$_{E_3}$: observed false accusation of restructured human text at the locked threshold. Unedited and drift: the two terms of the bound at the minimising $\delta$; bound: their sum. $q_{90}$: 90th percentile of the edit-induced shift $\Delta$ of human texts in units of the class separation, on calibration data edited by the training editors (s, seen) and on the test split (u, unseen). Unedit.: unedited term. All in \%, pooled over seeds.}",
         r"\label{tab:prop1}", r"\begin{tabular}{l" + "rrrrr" * len(sps) + "}", r"\toprule",
         " & " + " & ".join(r"\multicolumn{5}{c}{" + nm[s_] + "}" for s_ in sps) + r" \\",
         " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 5 * i, 6 + 5 * i) for i in range(len(sps))),
         "Model & " + " & ".join([r"FPR$_{E_3}$ & unedit. & drift & bound & $q_{90}$ s/u"] * len(sps)) + r" \\", r"\midrule"]
    for m in XL:
        if not all(m in P[s_] for s_ in sps):
            continue
        row = []
        for s_ in sps:
            r = P[s_][m]
            row += [f"{r['actual_E3_fpr']:.1f}", f"{r['term_unedited']:.1f}", f"{r['term_drift']:.1f}", f"{r['bound']:.1f}",
                    f"{100*r['drift_val_p90_rel']:.0f} / {100*r['drift_test_p90_rel']:.0f}"]
        L.append(XL[m] + " & " + " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(OUT, "t_prop1.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    t_main(); t_claims(); t_external(); t_prop1(); print("tables in", OUT)

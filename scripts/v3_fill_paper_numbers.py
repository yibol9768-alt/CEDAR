#!/usr/bin/env python3
"""Fill {{KEY}} placeholders in paper/sections/sec5_results.tex from results_v3/paper_numbers.json.

Keeps a template copy (sec5_results.tmpl.tex) so the fill can be re-run after the numbers change.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = json.load(open(os.path.join(ROOT, "results_v3", "paper_numbers.json")))
R = N["runs"]
C = N["contrasts"]
SEC = os.path.join(ROOT, "paper", "sections", "sec5_results.tex")
TMPL = os.path.join(ROOT, "paper", "sections", "sec5_results.tmpl.tex")
if not os.path.exists(TMPL):
    open(TMPL, "w").write(open(SEC).read())
text = open(TMPL).read()


def pct(run, key):
    return f"{R[run][key][0]:.1f}\\%"


def con(a, b, key="test_confirm_worst_group_E0_E3"):
    d, lo, hi = C[f"{a}__vs__{b}"][key]
    return f"${d:+.1f}$\\,pp [{lo:+.1f}, {hi:+.1f}]"


ci = N["calib_inflation"]
vals = {
    "UE_E0_E1": f"{ci['mb_v3_unedited_erm']['tau_e0'][1]:.1f}\\%",
    "UE_E0_E2": f"{ci['mb_v3_unedited_erm']['tau_e0'][2]:.1f}\\%",
    "UE_E0_E3": f"{ci['mb_v3_unedited_erm']['tau_e0'][3]:.1f}\\%",
    "SP_E0_E3": f"{ci['mb_v3_strong_combined_no_nsd']['tau_e0'][3]:.1f}\\%",
    "SP_CAL_AVG": f"{sum(ci['mb_v3_strong_combined_no_nsd']['tau_all']) / 4:.1f}\\%",
    "UE_MB": pct("mb_v3_unedited_erm", "c_wg"), "AUG_MB": pct("mb_v3_augmented_erm", "c_wg"),
    "UE_DB": pct("db_v3_unedited_erm", "c_wg"), "AUG_DB": pct("db_v3_augmented_erm", "c_wg"),
    "GDRO_MB": pct("mb_v3_erm_group_dro", "c_wg"), "PAIR_MB": pct("mb_v3_erm_paired_consist", "c_wg"),
    "SUPCON_MB": pct("mb_v3_supcon_orth_disentangle", "c_wg"),
    "SPOT_MB": pct("mb_v3_strong_combined_no_nsd", "c_wg"), "SPOT_DB": pct("db_v3_strong_combined_no_nsd", "c_wg"),
    "SPOT_LL": pct("mb_v3_strong_combined_no_nsd", "ll_len"), "AUG_LL": pct("mb_v3_augmented_erm", "ll_len"),
    "C_SPOT_AUG": con("mb_v3_strong_combined_no_nsd", "mb_v3_augmented_erm"),
    "C_SPOT_AUG_ID": con("mb_v3_strong_combined_no_nsd", "mb_v3_augmented_erm", "test_id_worst_group_E0_E3"),
    "C_SPOT_PAIR": con("mb_v3_strong_combined_no_nsd", "mb_v3_erm_paired_consist"),
    "C_SPOT_GDRO": con("mb_v3_strong_combined_no_nsd", "mb_v3_erm_group_dro"),
    "C_DTE_MB800": con("mb_v3_strong_combined_no_nsd", "mb_v3_nodte_no_nsd"),
    "C_DTE_MB600": con("mb_v3_strong_combined_fit600_no_nsd", "mb_v3_nodte_fit600_no_nsd"),
    "C_DTE_DB": con("db_v3_strong_combined_no_nsd", "db_v3_nodte_no_nsd"),
    "DTE_PD_MB": pct("mb_v3_cedar_split_op", "c_wg"), "NODTE_PD_MB": pct("mb_v3_ablation_no_dte", "c_wg"),
    "C_PD_MB_WG": con("mb_v3_ablation_no_dte", "mb_v3_nodte_fit600_no_nsd"),
    "C_PD_DB_WG": con("db_v3_cedar_no_dte", "db_v3_nodte_no_nsd"),
    "C_PD_MB_ID": con("mb_v3_ablation_no_dte", "mb_v3_nodte_fit600_no_nsd", "test_id_worst_group_E0_E3"),
    "C_PD_DB_ID": con("db_v3_cedar_no_dte", "db_v3_nodte_no_nsd", "test_id_worst_group_E0_E3"),
    "C_PD_MB_LL": con("mb_v3_ablation_no_dte", "mb_v3_nodte_fit600_no_nsd", "test_ood_llada_len120_220_auroc"),
    "C_PD_DB_LL": con("db_v3_cedar_no_dte", "db_v3_nodte_no_nsd", "test_ood_llada_len120_220_auroc"),
    "C_PD_DB_MS": con("db_v3_cedar_no_dte", "db_v3_nodte_no_nsd", "external_mixset_pure_auroc"),
    "UNPROT_WG": pct("mb_v3_ablation_unprotected_nsd", "c_wg"),
    "VAL_R": f"{N['val_confirm_pearson_configs']:.2f}", "N_CONF": str(N["n_configs_val_confirm"]),
}
holm = N.get("holm") or {}
adj = holm.get("holm_adjusted_p", {})
k = "confirm_wg_vs_aug_erm"
vals["HOLM_NOTE"] = (f"Holm-adjusted $p={adj[k]:.3f}$" if k in adj else "Holm-adjusted")
missing = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text)) - set(vals))
if missing:
    sys.exit(f"missing placeholders: {missing}")
for key, v in vals.items():
    text = text.replace("{{" + key + "}}", v)
open(SEC, "w").write(text)
print("filled", len(vals), "placeholders")

#!/usr/bin/env python3
"""Compare P1 LLM-backbone runs with ModernBERT/DeBERTa pair-consistency (same seed) on Confirm / Confirm-2:
AUROC, WG at the val_calib threshold, and WG / machine recall at an in-target 5% cluster-FPR threshold (ceiling),
plus K=30 human-anchor re-locking (100 draws). Usage: python scripts/icml_p1_eval.py results_icml/p1/*.json"""
import glob, json, sys
from collections import defaultdict
import numpy as np
src = open(__file__.replace("icml_p1_eval.py", "v3_seed_ensemble_probe.py")).read().split("out = {}")[0]
exec(src.replace("os.path.dirname(os.path.dirname(os.path.abspath(__file__)))", '"/home/liuyibo/auto-llm-driving/CEDAR"'))
def metrics(d, sp):
    R = SPL[sp]; ids = d["row_ids"][sp]
    keep = [i for i, k in enumerate(ids) if R[k]["edit_level"] in LV]
    y = np.array([int(R[ids[i]]["origin_label"]) for i in keep]); lv = np.array([R[ids[i]]["edit_level"] for i in keep]); pid = np.array([R[ids[i]]["parent_id"] for i in keep])
    s = logit(d["preds"][sp])[keep]; tl = logit([d["locked_thresholds"]["tau_cal_all5"]])[0]
    def wg(t, m=None):
        m = np.ones(len(s), bool) if m is None else m
        return 100 * min(np.mean((s[m & (y == o) & (lv == e)] > t) == (o == 1)) for o in (0, 1) for e in LV)
    t5 = conformal(s[y == 0], list(pid[y == 0]))
    parents = np.unique(pid); an = []
    for r in range(100):
        ch = np.random.default_rng(10000 + r).choice(parents, 30, replace=False); a = np.isin(pid, ch) & (y == 0)
        an.append(wg(conformal(s[a], list(pid[a])), ~np.isin(pid, ch)))
    return {"auroc": 100 * auroc(y, s), "wg_locked": wg(tl), "recall_locked": 100 * np.mean(s[y == 1] > tl),
            "wg_at_target5": wg(t5), "recall_at_target5": 100 * np.mean(s[y == 1] > t5), "wg_anchor30": np.mean(an)}
files = sys.argv[1:] + ["results_e2/model_preds/e2_mb_pair_seed42.json", "results_e2/model_preds/e2_db_pair_seed42.json",
                        "results_e2/model_preds/e2_mb_spot_nodte_seed42.json"]
for f in files:
    d = json.load(open(f))
    for sp in ["test_confirm", "test_confirm2"]:
        m = metrics(d, sp)
        print(f"{f.split('/')[-1][:-5]:30s} {sp[5:]:9s} " + "  ".join(f"{k} {v:5.1f}" for k, v in m.items()), flush=True)

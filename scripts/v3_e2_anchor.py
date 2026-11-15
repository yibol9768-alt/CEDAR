#!/usr/bin/env python3
"""Confirm-2 with the few-shot human-anchor recalibration that v3 already defined (step 5,
evaluate_few_shot_human_anchor_crc; Proposition/Theorem in app_proofs): K=30 human parents of the target split are
drawn, the threshold is re-locked with the same cluster-conformal rule on their 4 human texts (no machine text), all 8
texts of those parents are removed from evaluation, and WG / cluster human FPR are measured on the remaining parents.
200 draws per seed, metrics averaged over draws and seeds. Applied after seeing that zero-shot thresholds were
conservative on Confirm-2 (stated in the paper). Writes results_e2/e2_anchor.json.
"""
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "results_e2", "model_preds")
LEVELS = ["E0", "E1", "E2", "E3"]
K, N_DRAWS, ALPHA = 30, 200, 0.05
R = json.load(open(os.path.join(ROOT, "data_e2", "split_test_confirm2.json")))
ids = [r["id"] for r in R]
y = np.array([r["origin_label"] for r in R])
lv = np.array([r["edit_level"] for r in R])
pid = np.array([r["parent_id"] for r in R])
parents = sorted(set(pid))
by_parent = defaultdict(list)
for i, p in enumerate(pid):
    by_parent[p].append(i)


def conformal(scores, pids):
    cnt = Counter(pids)
    w = np.array([1.0 / cnt[p] for p in pids])
    for c in np.unique(scores):
        if (1.0 + w[scores > c].sum()) / (len(cnt) + 1) <= ALPHA + 1e-12:
            return float(c)
    return float("inf")


def metrics(s, tau, mask):
    wg = min(np.mean((s[mask & (y == o) & (lv == e)] > tau) == (o == 1)) for o in (0, 1) for e in LEVELS)
    h = mask & (y == 0)
    fpr_by_parent = defaultdict(list)
    for i in np.where(h)[0]:
        fpr_by_parent[pid[i]].append(s[i] > tau)
    cfpr = np.mean([np.mean(v) for v in fpr_by_parent.values()])
    rec = np.mean(s[mask & (y == 1)] > tau)
    return wg, cfpr, rec


def score_sets():
    sets = {}
    for f in sorted(glob.glob(os.path.join(PRED, "e2_*_seed*.json"))):
        run = os.path.basename(f).rsplit("_seed", 1)[0]
        d = json.load(open(f))
        m = dict(zip(d["row_ids"]["test_confirm2"], d["preds"]["test_confirm2"]))
        sets.setdefault(run, []).append((np.array([m[i] for i in ids]), d["locked_thresholds"]["tau_cal_all5"]))
    pe = json.load(open(os.path.join(ROOT, "results_e2", "preds_pretrained_e2.json")))
    pv3 = json.load(open(os.path.join(ROOT, "results_v3", "preds_pretrained_and_probes_v3.json")))
    val_rows = json.load(open(os.path.join(ROOT, "data", "split_val.json")))
    cal_set = set(json.load(open(os.path.join(ROOT, "data_v2", "val_manifest.json")))["val_calib_parent_ids"])
    cal_idx = [i for i, r in enumerate(val_rows) if r["parent_id"] in cal_set]
    for det in ["desklib_raid_v1", "fast_detectgpt_qwen3b", "binoculars_qwen3b", "radar_vicuna7b", "hc3_roberta_detector"]:
        cal = np.asarray(pv3[det]["val"], dtype=np.float64)[cal_idx]
        hum = [j for j, i in enumerate(cal_idx) if val_rows[i]["origin_label"] == 0]
        tau = conformal(cal[hum], [val_rows[cal_idx[j]]["parent_id"] for j in hum])
        assert pe["row_ids"]["test_confirm2"] == ids
        sets[det] = [(np.asarray(pe[det]["test_confirm2"], dtype=np.float64), tau)]
    return sets


def main():
    out = {"K_human_parents": K, "n_draws": N_DRAWS, "alpha": ALPHA, "runs": {}}
    draws = []
    for r in range(N_DRAWS):
        rng = np.random.default_rng(10000 + r)  # same draw seeds as v3
        chosen = set(rng.choice(parents, size=K, replace=False).tolist())
        cal = np.array([i for p in chosen for i in by_parent[p] if y[i] == 0])
        ev = ~np.isin(pid, list(chosen))
        draws.append((cal, ev))
    for run, seeds in sorted(score_sets().items()):
        zs, an = [], []
        for s, tau0 in seeds:
            zs.append(metrics(s, tau0, np.ones(len(s), bool)))
            per = []
            for cal, ev in draws:
                tau = conformal(s[cal], list(pid[cal]))
                per.append(metrics(s, tau, ev))
            an.append(np.mean(per, axis=0))
        zs, an = np.mean(zs, axis=0) * 100, np.mean(an, axis=0) * 100
        out["runs"][run] = {"n_seeds": len(seeds),
                            "zero_shot": {"wg": round(zs[0], 2), "cluster_human_fpr": round(zs[1], 2), "machine_recall": round(zs[2], 2)},
                            "anchor_k30": {"wg": round(an[0], 2), "cluster_human_fpr": round(an[1], 2), "machine_recall": round(an[2], 2)}}
        print(f"{run:28s} zero-shot WG {zs[0]:5.1f} FPR {zs[1]:4.1f} | anchored WG {an[0]:5.1f} FPR {an[1]:4.1f} rec {an[2]:5.1f}", flush=True)
    json.dump(out, open(os.path.join(ROOT, "results_e2", "e2_anchor.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

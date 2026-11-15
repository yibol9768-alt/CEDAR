#!/usr/bin/env python3
"""E2 phase B: pick one candidate by the pre-registered score (docs/E2_plan_2026-11-15.md).

Score = worst-group accuracy over the 16 cells (origin x E0..E3) x {val_select, val_shift} at the locked
val_calib threshold (rule: score > tau => machine), averaged over seeds; ties (<0.5 pp) -> mean balanced acc.
Reads only val_select / val_calib / val_shift predictions. Never opens test_confirm2 predictions.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "results_e2", "model_preds_phase_b")
OUT = os.path.join(ROOT, "results_e2", "phase_b_selection.json")
SEEDS = [42, 1042]
CANDS = {"R0": "e2_mb_spot", "R1": "e2_mb_supcon", "C1": "e2_mb_gappd_split", "C2": "e2_mb_gappd_ema",
         "C3": "e2_mb_supcon_pairxm", "C4": ("e2_mb_spot", "e2_mb_aug")}

val_manifest = json.load(open(os.path.join(ROOT, "data_v2", "val_manifest.json")))
val_rows = json.load(open(os.path.join(ROOT, "data", "split_val.json")))
shift_rows = json.load(open(os.path.join(ROOT, "data_e2", "split_val_shift.json")))
META = {}
for r in val_rows + shift_rows:
    META[r["id"]] = (int(r["origin_label"]), r["edit_level"], r["parent_id"])


def cluster_conformal(scores, pids, alpha=0.05):  # identical rule to the training suite
    scores = np.asarray(scores, dtype=np.float64)
    cnt = Counter(pids)
    K = len(cnt)
    w = np.array([1.0 / cnt[p] for p in pids])
    for c in np.unique(np.sort(scores)):
        if (1.0 + w[scores > c].sum()) / (K + 1) <= alpha + 1e-12:
            return float(c)
    return float("inf")


def load(run, seed):
    d = json.load(open(os.path.join(PRED, f"{run}_seed{seed}.json")))
    return {sp: dict(zip(d["row_ids"][sp], d["preds"][sp])) for sp in ("val_select", "val_calib", "val_shift")}, d


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def cells(scores, tau):
    acc = {}
    for sp in ("val_select", "val_shift"):
        g = defaultdict(list)
        for rid, s in scores[sp].items():
            o, lv, _ = META[rid]
            g[(o, lv)].append((s > tau) == (o == 1))
        for k, v in g.items():
            acc[(sp,) + k] = float(np.mean(v))
    bal = []
    for sp in ("val_select", "val_shift"):
        for lv in ("E0", "E1", "E2", "E3"):
            bal.append(0.5 * (acc[(sp, 0, lv)] + acc[(sp, 1, lv)]))
    return min(acc.values()), float(np.mean(bal)), acc


def score_candidate(spec, seed):
    if isinstance(spec, tuple):
        a, _ = load(spec[0], seed)
        b, _ = load(spec[1], seed)
        sc = {sp: {rid: float(logit(a[sp][rid]) + logit(b[sp][rid])) / 2.0 for rid in a[sp]} for sp in a}
        hum = [(s, META[r][2]) for r, s in sc["val_calib"].items() if META[r][0] == 0]
        tau = cluster_conformal([h[0] for h in hum], [h[1] for h in hum])
    else:
        sc, d = load(spec, seed)
        tau = d["locked_thresholds"]["tau_cal_all5"]
    return cells(sc, tau)


def main():
    res = {}
    for name, spec in CANDS.items():
        try:
            per = [score_candidate(spec, sd) for sd in SEEDS]
        except FileNotFoundError as e:
            print(f"{name}: missing {e.filename}")
            continue
        res[name] = {"wg16": round(float(np.mean([p[0] for p in per])) * 100, 2),
                     "mean_bal16": round(float(np.mean([p[1] for p in per])) * 100, 2),
                     "per_seed_wg16": [round(p[0] * 100, 2) for p in per],
                     "val_select_wg": [round(min(v for k, v in p[2].items() if k[0] == "val_select") * 100, 1) for p in per],
                     "val_shift_wg": [round(min(v for k, v in p[2].items() if k[0] == "val_shift") * 100, 1) for p in per]}
        print(name, res[name])
    if len(res) < len(CANDS):
        sys.exit("not all candidates available; no selection made")
    best = max(res.values(), key=lambda r: r["wg16"])["wg16"]
    tied = [n for n, r in res.items() if best - r["wg16"] < 0.5]
    chosen = max(tied, key=lambda n: res[n]["mean_bal16"])
    out = {"rule": "docs/E2_plan_2026-11-15.md", "seeds": SEEDS, "scores": res, "tied_within_0.5pp": tied, "chosen": chosen}
    json.dump(out, open(OUT, "w"), indent=1)
    print("CHOSEN:", chosen, "tied:", tied)


if __name__ == "__main__":
    main()

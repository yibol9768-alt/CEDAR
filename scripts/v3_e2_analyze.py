#!/usr/bin/env python3
"""E2 phase C analysis on test_confirm2 / val_shift (rules in docs/E2_plan_2026-11-15.md).

Per configuration: metrics are computed per seed at that seed's locked val_calib threshold and averaged over
seeds (as in v3). Paired differences use a parent-cluster bootstrap (5,000 resamples, 95% percentile CI).
Writes results_e2/e2_analysis.json.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata


def roc_auc_score(y, s):  # Mann-Whitney AUROC with average ranks for ties (= sklearn's value)
    y = np.asarray(y)
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "results_e2", "model_preds")
OUT = os.path.join(ROOT, "results_e2", "e2_analysis.json")
LEVELS = ["E0", "E1", "E2", "E3"]
SEL = json.load(open(os.path.join(ROOT, "results_e2", "phase_b_selection.json")))
TAG = {"R0": "spot", "R1": "supcon", "C1": "gappd_split", "C2": "gappd_ema", "C3": "supcon_pairxm", "C4": "ens_spot_aug"}
CHOSEN = TAG[SEL["chosen"]]
MB = ["unedited", "aug", "joint8", "multitask", "supcon", "gdro", "pair", "spot", "spot_nodte", "spot_nodte600"]
DB = ["unedited", "aug", "supcon", "pair", "spot", "spot_nodte", "spot_nodte600"]
PRETRAINED = ["desklib_raid_v1", "fast_detectgpt_qwen3b", "binoculars_qwen3b", "radar_vicuna7b", "hc3_roberta_detector"]

rows = {sp: json.load(open(os.path.join(ROOT, "data_e2", f"split_{sp}.json"))) for sp in ("val_shift", "test_confirm2")}
val_rows = json.load(open(os.path.join(ROOT, "data", "split_val.json")))
val_manifest = json.load(open(os.path.join(ROOT, "data_v2", "val_manifest.json")))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def cluster_conformal(scores, pids, alpha=0.05):  # identical rule to the training suite
    from collections import Counter
    scores = np.asarray(scores, dtype=np.float64)
    cnt = Counter(pids)
    K = len(cnt)
    w = np.array([1.0 / cnt[p] for p in pids])
    for c in np.unique(np.sort(scores)):
        if (1.0 + w[scores > c].sum()) / (K + 1) <= alpha + 1e-12:
            return float(c)
    return float("inf")


def load_seeds(run):
    """list of (scores_by_split {split: np.array aligned to rows[split] / val_calib ids}, tau)"""
    out = []
    for fp in sorted(glob.glob(os.path.join(PRED, f"{run}_seed*.json"))):
        d = json.load(open(fp))
        sc = {}
        for sp in list(rows) + ["val_calib"]:
            if sp in d["preds"]:
                m = dict(zip(d["row_ids"][sp], d["preds"][sp]))
                ids = [r["id"] for r in rows[sp]] if sp in rows else d["row_ids"][sp]
                sc[sp] = np.asarray([m[i] for i in ids], dtype=np.float64)
        sc["_val_calib_ids"] = d["row_ids"]["val_calib"]
        out.append((sc, d["locked_thresholds"]["tau_cal_all5"]))
    return out


VAL_META = {r["id"]: r for r in val_rows}


def ensemble(a_seeds, b_seeds):
    """C4: logit average of two runs, same seed order; threshold re-locked on val_calib humans."""
    out = []
    for (a, _), (b, _) in zip(a_seeds, b_seeds):
        sc = {sp: (logit(a[sp]) + logit(b[sp])) / 2 for sp in a if not sp.startswith("_")}
        ids = a["_val_calib_ids"]
        hum = [(s, VAL_META[i]["parent_id"]) for i, s in zip(ids, sc["val_calib"]) if VAL_META[i]["origin_label"] == 0]
        out.append((sc, cluster_conformal([h[0] for h in hum], [h[1] for h in hum])))
    return out


def get_config(backbone, tag):
    if tag == "ens_spot_aug":
        return ensemble(load_seeds(f"e2_{backbone}_spot"), load_seeds(f"e2_{backbone}_aug"))
    return load_seeds(f"e2_{backbone}_{tag}")


def metric_fns(sp, seeds):
    R = rows[sp]
    y = np.array([r["origin_label"] for r in R])
    lv = np.array([r["edit_level"] for r in R])
    gen = np.array([r["draft_generator"] for r in R])
    par_gen = {r["parent_id"]: r["draft_generator"] for r in R if r["origin_label"] == 1}
    pgen = np.array([par_gen[r["parent_id"]] for r in R])

    def wg(idx):
        vals = []
        for sc, tau in seeds:
            s, yy, ll = sc[sp][idx], y[idx], lv[idx]
            cells = [np.mean((s[(yy == o) & (ll == e)] > tau) == (o == 1)) for o in (0, 1) for e in LEVELS
                     if ((yy == o) & (ll == e)).any()]
            vals.append(min(cells))
        return float(np.mean(vals))

    def auc_sub(mask_fn):
        def f(idx):
            vals = []
            for sc, _ in seeds:
                m = mask_fn(idx)
                yy = y[idx][m]
                if len(np.unique(yy)) < 2:
                    return float("nan")
                vals.append(roc_auc_score(yy, sc[sp][idx][m]))
            return float(np.mean(vals))
        return f

    def hfpr(e):
        def f(idx):
            return float(np.mean([np.mean(sc[sp][idx][(y[idx] == 0) & (lv[idx] == e)] > tau) for sc, tau in seeds]))
        return f

    fns = {"wg": wg, "auroc": auc_sub(lambda idx: np.ones(len(idx), bool))}
    for g in sorted(set(gen) - {"human"}):
        short = g.split("/")[-1]
        fns[f"auroc_{short}"] = auc_sub(lambda idx, g=g: pgen[idx] == g)
    for e in LEVELS:
        fns[f"human_fpr_{e}"] = hfpr(e)
    return fns


def boot_diff(sp, fa, fb, n_boot=5000, seed=42):
    R = rows[sp]
    groups = defaultdict(list)
    for i, r in enumerate(R):
        groups[r["parent_id"]].append(i)
    pids = list(groups)
    rng = np.random.default_rng(seed)
    full = np.arange(len(R))
    pt = fa(full) - fb(full)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[pids[c]] for c in rng.integers(0, len(pids), len(pids))])
        a, b = fa(idx), fb(idx)
        if not (np.isnan(a) or np.isnan(b)):
            d.append(a - b)
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)))
    return {"diff_pp": round(pt * 100, 2), "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)], "p_boot": round(float(p), 5)}


def main():
    res = {"chosen": SEL["chosen"], "chosen_tag": CHOSEN, "configs": {}, "contrasts": {}, "pretrained": {}}
    cfg = {}
    for bb, tags in (("mb", MB), ("db", DB)):
        for t in tags + ([CHOSEN] if CHOSEN not in tags else []):
            seeds = get_config(bb, t)
            if not seeds:
                print(f"missing {bb} {t}")
                continue
            cfg[(bb, t)] = {sp: metric_fns(sp, seeds) for sp in rows}
            full = {sp: np.arange(len(rows[sp])) for sp in rows}
            res["configs"][f"{bb}_{t}"] = {"n_seeds": len(seeds), **{
                f"{sp}/{k}": round(f(full[sp]) * 100, 2) for sp in rows for k, f in cfg[(bb, t)][sp].items()}}
            print(bb, t, len(seeds), res["configs"][f"{bb}_{t}"]["test_confirm2/wg"])
    # pre-registered decision contrasts (rule 1 and 2) + replication of the v3 headline contrasts
    pairs = [(CHOSEN, "spot"), (CHOSEN, "supcon"), (CHOSEN, "spot_nodte600"), (CHOSEN, "spot_nodte"), ("spot_nodte", "spot"), ("spot", "aug"), ("spot", "unedited"), ("spot", "supcon"),
             ("spot", "pair"), ("spot", "gdro"), ("supcon", "aug"), ("pair", "aug")]
    for bb in ("mb", "db"):
        for a, b in pairs:
            if a == b or (bb, a) not in cfg or (bb, b) not in cfg:
                continue
            key = f"{bb}_{a}_vs_{b}"
            res["contrasts"][key] = {}
            for metric in ["wg", "auroc"] + [k for k in cfg[(bb, a)]["test_confirm2"] if k.startswith("auroc_")]:
                res["contrasts"][key][f"test_confirm2/{metric}"] = boot_diff(
                    "test_confirm2", cfg[(bb, a)]["test_confirm2"][metric], cfg[(bb, b)]["test_confirm2"][metric])
            res["contrasts"][key]["val_shift/wg"] = boot_diff("val_shift", cfg[(bb, a)]["val_shift"]["wg"], cfg[(bb, b)]["val_shift"]["wg"])
            print(key, res["contrasts"][key]["test_confirm2/wg"])
    # decision rules
    dream = [k for k in res["configs"].get(f"mb_{CHOSEN}", {}) if k.startswith("test_confirm2/auroc_Dream")]
    if CHOSEN != "spot":
        r1 = all(res["contrasts"][f"mb_{CHOSEN}_vs_{b}"]["test_confirm2/wg"]["ci95_pp"][0] > 0
                 and res["contrasts"].get(f"db_{CHOSEN}_vs_{b}", {}).get("test_confirm2/wg", {}).get("diff_pp", -1) > 0
                 for b in ("spot", "supcon") if f"mb_{CHOSEN}_vs_{b}" in res["contrasts"])
        r2 = bool(dream) and all(res["contrasts"].get(f"{bb}_{CHOSEN}_vs_spot", {}).get(dream[0], {}).get("ci95_pp", [-1])[0] > 0
                                 for bb in ("mb", "db"))
        res["rules"] = {"rule1_headline_improvement": bool(r1), "rule2_recovers_transfer": bool(r2)}
    # pretrained detectors: threshold locked on v3 val_calib predictions, as in v3
    pp = os.path.join(ROOT, "results_e2", "preds_pretrained_e2.json")
    if os.path.exists(pp):
        pe = json.load(open(pp))
        pv3 = json.load(open(os.path.join(ROOT, "results_v3", "preds_pretrained_and_probes_v3.json")))
        cal_set = set(val_manifest["val_calib_parent_ids"])
        cal_idx = [i for i, r in enumerate(val_rows) if r["parent_id"] in cal_set]  # same as v3 step 5
        cal_rows = [val_rows[i] for i in cal_idx]
        for det in PRETRAINED:
            if det not in pe:
                continue
            cal = np.asarray(pv3[det]["val"], dtype=np.float64)[cal_idx]
            hum = [i for i, r in enumerate(cal_rows) if r["origin_label"] == 0]
            tau = cluster_conformal(cal[hum], [cal_rows[i]["parent_id"] for i in hum])
            seeds = [({sp: np.asarray(pe[det][sp], dtype=np.float64) for sp in rows}, tau)]
            for sp in rows:
                assert pe["row_ids"][sp] == [r["id"] for r in rows[sp]]
            fns = {sp: metric_fns(sp, seeds) for sp in rows}
            res["pretrained"][det] = {f"{sp}/{k}": round(f(np.arange(len(rows[sp]))) * 100, 2)
                                      for sp in rows for k, f in fns[sp].items()}
            print(det, res["pretrained"][det]["test_confirm2/auroc"], res["pretrained"][det]["test_confirm2/wg"])
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()

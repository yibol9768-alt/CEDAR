#!/usr/bin/env python3
"""Post-review analyses on existing predictions (no training). Writes results_e2/review_analyses.json.

1. Seed+parent uncertainty for headline WG contrasts on Confirm (v3 runs) and Confirm-2 (E2 runs):
   - per-seed WG of each config (each at its own locked val_calib threshold) and paired-by-seed differences;
   - two-level bootstrap: each resample draws parent clusters with replacement AND, independently for A and B,
     seeds with replacement; statistic = mean over drawn seeds of WG on the drawn parents (A minus B).
   The existing CIs (parents only, per-seed metrics averaged over the 5 fixed seeds) are reproduced for reference.
2. Surface-feature shortcut controls: length-only and a 5-feature logistic regression (length, mean word length,
   type-token ratio, punctuation rate, sentences) trained on train_v2, threshold locked on val_calib humans with the
   same cluster-conformal rule; AUROC and WG on Confirm and Confirm-2.
3. Confirm-2 by genre: AUROC/WG of headline configs excluding the Wikipedia parents (genre also in Confirm), and per genre.
"""
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LV = ["E0", "E1", "E2", "E3"]
NB = 5000


def load_rows(path):
    R = json.load(open(path))
    for r in R:
        r["origin_label"] = int(r["origin_label"])
    return R


SPLITS = {"test_confirm": (load_rows(os.path.join(ROOT, "data_v2", "split_test_confirm.json")),
                           os.path.join(ROOT, "results_v3", "model_preds_v3")),
          "test_confirm2": (load_rows(os.path.join(ROOT, "data_e2", "split_test_confirm2.json")),
                            os.path.join(ROOT, "results_e2", "model_preds"))}


def auroc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class Split:
    def __init__(self, sp):
        R, self.pred_dir = SPLITS[sp]
        self.sp = sp
        self.R = [r for r in R if r["edit_level"] in LV]
        self.ids = [r["id"] for r in self.R]
        self.y = np.array([r["origin_label"] for r in self.R])
        self.lv = np.array([r["edit_level"] for r in self.R])
        self.pid = np.array([r["parent_id"] for r in self.R])
        self.dom = np.array([r["domain"] for r in self.R])
        self.parents = sorted(set(self.pid))
        pix = {p: i for i, p in enumerate(self.parents)}
        self.pidx = np.array([pix[p] for p in self.pid])
        self.cell = np.array([o * 4 + LV.index(e) for o, e in zip(self.y, self.lv)])

    def seeds(self, run):
        out = []
        for f in sorted(glob.glob(os.path.join(self.pred_dir, f"{run}_seed*.json")), key=lambda f: int(f.rsplit("seed", 1)[1][:-5])):
            d = json.load(open(f))
            m = dict(zip(d["row_ids"][self.sp], d["preds"][self.sp]))
            out.append(((np.array([m[i] for i in self.ids], dtype=np.float64) > d["locked_thresholds"]["tau_cal_all5"]) == (self.y == 1)).astype(float))
        return np.array(out)  # seeds x rows correctness

    def wg_from_counts(self, correct, w):
        """correct: (k, rows) ; w: (rows,) weights = multiplicity of each row's parent. Returns (k,) WG."""
        res = np.full(correct.shape[0], np.inf)
        for c in range(8):
            mk = self.cell == c
            acc = (correct[:, mk] * w[mk]).sum(1) / w[mk].sum()
            res = np.minimum(res, acc)
        return res


def contrasts():
    C = {"test_confirm": [("mb", "mb_v3_strong_combined_no_nsd", "mb_v3_augmented_erm", "Spot vs Aug"),
                          ("db", "db_v3_strong_combined_no_nsd", "db_v3_augmented_erm", "Spot vs Aug"),
                          ("mb", "mb_v3_erm_paired_consist", "mb_v3_augmented_erm", "Pair vs Aug"),
                          ("db", "db_v3_erm_paired_consist", "db_v3_augmented_erm", "Pair vs Aug"),
                          ("mb", "mb_v3_strong_combined_no_nsd", "mb_v3_erm_paired_consist", "Spot vs Pair"),
                          ("db", "db_v3_strong_combined_no_nsd", "db_v3_erm_paired_consist", "Spot vs Pair"),
                          ("mb", "mb_v3_strong_combined_no_nsd", "mb_v3_erm_group_dro", "Spot vs GDRO"),
                          ("mb", "mb_v3_erm_group_dro", "mb_v3_augmented_erm", "GDRO vs Aug")],
         "test_confirm2": [("mb", "e2_mb_spot", "e2_mb_aug", "Spot vs Aug"), ("db", "e2_db_spot", "e2_db_aug", "Spot vs Aug"),
                           ("mb", "e2_mb_pair", "e2_mb_aug", "Pair vs Aug"), ("db", "e2_db_pair", "e2_db_aug", "Pair vs Aug"),
                           ("mb", "e2_mb_gappd_split", "e2_mb_spot", "gPD vs Spot"), ("db", "e2_db_gappd_split", "e2_db_spot", "gPD vs Spot"),
                           ("mb", "e2_mb_spot_nodte", "e2_mb_spot", "Spot w/o DTE vs Spot"),
                           ("db", "e2_db_spot_nodte", "e2_db_spot", "Spot w/o DTE vs Spot")]}
    out = {}
    for sp, lst in C.items():
        S = Split(sp)
        P = len(S.parents)
        rng = np.random.default_rng(42)
        Wp = np.stack([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(NB)]).astype(float)  # NB x P
        for bb, a, b, name in lst:
            ca, cb = S.seeds(a), S.seeds(b)
            wa = S.wg_from_counts(ca, np.ones(len(S.y))) * 100
            wb = S.wg_from_counts(cb, np.ones(len(S.y))) * 100
            k = min(len(wa), len(wb))
            per_seed = (wa[:k] - wb[:k]).tolist()
            pa = np.stack([S.wg_from_counts(ca, Wp[i][S.pidx]) for i in range(NB)]) * 100  # NB x seeds
            pb = np.stack([S.wg_from_counts(cb, Wp[i][S.pidx]) for i in range(NB)]) * 100
            parent_only = pa.mean(1) - pb.mean(1)
            sa = rng.integers(0, pa.shape[1], (NB, pa.shape[1]))
            sb = rng.integers(0, pb.shape[1], (NB, pb.shape[1]))
            two = np.take_along_axis(pa, sa, 1).mean(1) - np.take_along_axis(pb, sb, 1).mean(1)
            rec = {"diff": round(float(wa.mean() - wb.mean()), 2),
                   "ci_parents": [round(float(x), 2) for x in np.percentile(parent_only, [2.5, 97.5])],
                   "ci_seeds_and_parents": [round(float(x), 2) for x in np.percentile(two, [2.5, 97.5])],
                   "per_seed_diff_paired_by_seed": [round(x, 1) for x in per_seed],
                   "n_seed_diffs_positive": int(sum(x > 0 for x in per_seed)), "n_seeds": k,
                   "wg_per_seed_a": [round(x, 1) for x in wa], "wg_per_seed_b": [round(x, 1) for x in wb]}
            out[f"{sp}/{bb}/{name}"] = rec
            print(sp, bb, name, rec["diff"], rec["ci_parents"], rec["ci_seeds_and_parents"], rec["per_seed_diff_paired_by_seed"], flush=True)
    return out


def feats(t):
    w = t.split()
    n = max(len(w), 1)
    sents = max(len(re.findall(r"[.!?]+(\s|$)", t)), 1)
    return [len(w), np.mean([len(x) for x in w]) if w else 0, len(set(x.lower() for x in w)) / n,
            len(re.findall(r"[,;:()\-\"']", t)) / n, n / sents]


def conformal(scores, pids, alpha=0.05):
    cnt = Counter(pids)
    w = np.array([1.0 / cnt[p] for p in pids])
    for c in np.unique(scores):
        if (1.0 + w[scores > c].sum()) / (len(cnt) + 1) <= alpha + 1e-12:
            return float(c)
    return float("inf")


def shortcut_controls():
    tr = load_rows(os.path.join(ROOT, "data_v2", "split_train_v2.json"))
    tr = [r for r in tr if r["edit_level"] in LV]
    val = load_rows(os.path.join(ROOT, "data", "split_val.json"))
    cal_set = set(json.load(open(os.path.join(ROOT, "data_v2", "val_manifest.json")))["val_calib_parent_ids"])
    cal = [r for r in val if r["parent_id"] in cal_set and r["origin_label"] == 0 and r["edit_level"] in LV]
    X = np.array([feats(r["text"]) for r in tr]); y = np.array([r["origin_label"] for r in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    wv = np.zeros(Z.shape[1])
    for _ in range(3000):  # plain gradient descent logistic regression
        p = 1 / (1 + np.exp(-Z @ wv))
        wv -= 0.5 * (Z.T @ (p - y) / len(y) + 1e-3 * np.r_[wv[:-1], 0])
    sign_len = 1.0 if auroc(y, X[:, 0]) >= 0.5 else -1.0
    models = {"length_only": lambda F: sign_len * F[:, 0],
              "surface_logreg5": lambda F: np.c_[(F - mu) / sd, np.ones(len(F))] @ wv}
    out = {"train_auroc": {"length_only": round(100 * auroc(y, sign_len * X[:, 0]), 1),
                           "surface_logreg5": round(100 * auroc(y, Z @ wv), 1)}}
    Fc = np.array([feats(r["text"]) for r in cal])
    for name, f in models.items():
        tau = conformal(f(Fc), [r["parent_id"] for r in cal])
        for sp in SPLITS:
            S = Split(sp)
            F = np.array([feats(r["text"]) for r in S.R])
            s = f(F)
            corr = ((s > tau) == (S.y == 1)).astype(float)[None]
            wg = S.wg_from_counts(corr, np.ones(len(S.y)))[0]
            hfpr = np.mean(s[S.y == 0] > tau)
            out[f"{sp}/{name}"] = {"auroc": round(100 * auroc(S.y, s), 1), "wg": round(100 * wg, 1),
                                   "human_fpr": round(100 * hfpr, 1), "machine_recall": round(100 * np.mean(s[S.y == 1] > tau), 1)}
            print(sp, name, out[f"{sp}/{name}"], flush=True)
    for sp in SPLITS:
        S = Split(sp)
        wc = np.array([len(r["text"].split()) for r in S.R])
        out[f"{sp}/median_wc"] = {"human": float(np.median(wc[S.y == 0])), "machine": float(np.median(wc[S.y == 1]))}
    return out


def by_genre():
    S = Split("test_confirm2")
    runs = ["e2_mb_aug", "e2_mb_pair", "e2_mb_spot", "e2_mb_spot_nodte", "e2_mb_gappd_split",
            "e2_db_aug", "e2_db_pair", "e2_db_spot", "e2_db_spot_nodte", "e2_db_gappd_split"]
    out = {}
    doms = sorted(set(S.dom))
    for run in runs:
        c = S.seeds(run)
        rec = {}
        for name, mask in [("all", np.ones(len(S.y), bool)), ("no_wiki", np.array(["wiki" not in d for d in S.dom]))] + \
                          [(d, S.dom == d) for d in doms]:
            rec[name] = round(float(S.wg_from_counts(c, mask.astype(float)).mean() * 100), 1)
        out[run] = rec
        print(run, rec, flush=True)
    for bb in ("mb", "db"):
        for a, b in [("spot", "aug"), ("pair", "aug")]:
            out[f"{bb}_{a}_vs_{b}"] = {k: round(out[f"e2_{bb}_{a}"][k] - out[f"e2_{bb}_{b}"][k], 1) for k in out[f"e2_{bb}_{a}"]}
            print(bb, a, b, out[f"{bb}_{a}_vs_{b}"])
    out["domains"] = {d: int((S.dom == d).sum() // 8) for d in doms}
    return out


if __name__ == "__main__":
    res = {"shortcut_controls": shortcut_controls(), "confirm2_by_genre": by_genre(), "contrasts_seed_parent": contrasts()}
    json.dump(res, open(os.path.join(ROOT, "results_e2", "review_analyses.json"), "w"), indent=1)

#!/usr/bin/env python3
"""ICML final analysis (pre-registration: docs/ICML_PLAN_2026-11-15.md section 6, docs/CONFIRM3_SPEC_2026-11-15.md).

Per method and split: WG / human FPR / human-E3 FPR / machine recall at the locked val_calib threshold, AUROC,
WG and recall at an in-split 5% cluster-FPR threshold (oracle diagnostic, set per seed on the full split), and
K=30 human-anchor re-locking. Pre-registered contrasts use the seed+parent two-level bootstrap (NB draws).
Confirm-3 is read only with --unseal (sealed until every final run exists).
Usage: python scripts/icml_final_analysis.py [--dev] [--unseal]
  --dev   use the P3 pilot runs (results_icml/p1) as stand-ins, to test the script on Confirm/Confirm-2 only."""
import glob, json, os, sys
from collections import Counter
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LV = ["E0", "E1", "E2", "E3"]
NB = int(os.environ.get("NB", "2000"))
DEV, UNSEAL = "--dev" in sys.argv, "--unseal" in sys.argv
if DEV:
    METHODS = {"8B pair": ("results_icml/p1", "qwen3_8b_pairx"), "8B aug": ("results_icml/p1", "qwen3_8b_augx"),
               "1.7B pair": ("results_icml/p1", "qwen3_1p7b_pair"), "1.7B aug": ("results_icml/p1", "qwen3_1p7b_aug"),
               "MB pair": ("results_e2/model_preds", "e2_mb_pair"), "MB aug": ("results_e2/model_preds", "e2_mb_aug")}
else:
    METHODS = {f"{n} {l}": ("results_icml/final", f"f_{t}_{l}") for n, t in (("8B", "8b"), ("4B", "4b"), ("1.7B", "1p7b"))
               for l in ("pair", "aug")}
    METHODS.update({"8B SupCon": ("results_icml/final", "f_8b_supcon"), "8B GroupDRO": ("results_icml/final", "f_8b_gdro"),
                    "8B logit-pair": ("results_icml/final", "f_8b_logitpair"),
                    "MB pair": ("results_icml/final_enc", "e2_mb_pair"), "MB aug": ("results_icml/final_enc", "e2_mb_aug")})
PRETRAINED = ["fast_detectgpt_qwen3b", "binoculars_qwen3b", "radar_vicuna7b", "desklib_raid_v1", "hc3_roberta_detector"]
SPLIT_FILES = {"test_confirm": "data_v2/split_test_confirm.json", "test_confirm2": "data_e2/split_test_confirm2.json",
               "test_confirm3": "data_c3/split_test_confirm3.json"}
SPLITS = ["test_confirm", "test_confirm2"] + (["test_confirm3", "test_confirm3_lenmatch"] if UNSEAL else [])
LEN_BAND = (0.75, 1.33)  # spec amendment 2026-11-15 10:0x SGT: octets with E0 machine/human word ratio in this band


def base_split(sp):
    return sp.replace("_lenmatch", "")


def wc(t):
    return len(t.split())


def conformal(s, pids, a=0.05, w=None):
    """Parent-cluster conformal threshold (Eq. 2); optional per-row bootstrap multiplicity w."""
    s = np.asarray(s, float); pids = list(pids)
    c = Counter(pids); w = np.ones(len(s)) if w is None else np.asarray(w, float)
    per = w / np.array([c[p] for p in pids]); k = per.sum()  # number of clusters, counted with bootstrap multiplicity
    order = np.argsort(s); ss = s[order]; pp = per[order]
    tail = np.cumsum(pp[::-1])[::-1]  # weight of rows with score >= ss[j]
    for j, t in enumerate(ss):
        above = tail[j + 1] if j + 1 < len(ss) else 0.0  # rows strictly above t (ties broken by order; scores are continuous)
        if (1 + above) / (k + 1) <= a + 1e-12:
            return float(t)
    return float("inf")


def auroc(y, s, w=None):
    if w is None:
        r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
        return 100 * (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    o = np.argsort(s); y, s, w = y[o], s[o], w[o]
    cw0 = np.cumsum(w * (y == 0)); tot0, tot1 = cw0[-1], (w * (y == 1)).sum()
    return 100 * float((w * (y == 1) * cw0).sum() / (tot0 * tot1))


class Split:
    def __init__(self, sp):
        self.sp = sp
        self.R = [r for r in json.load(open(os.path.join(ROOT, SPLIT_FILES[base_split(sp)]))) if r["edit_level"] in LV]
        if sp.endswith("_lenmatch"):
            e0 = {(r["parent_id"], int(r["origin_label"])): wc(r["text"]) for r in self.R if r["edit_level"] == "E0"}
            keep = {p for (p, o) in e0 if o == 0 and (p, 1) in e0 and LEN_BAND[0] <= e0[(p, 1)] / e0[(p, 0)] <= LEN_BAND[1]}
            self.R = [r for r in self.R if r["parent_id"] in keep]
        self.ids = [r["id"] for r in self.R]
        self.y = np.array([int(r["origin_label"]) for r in self.R]); self.lv = np.array([r["edit_level"] for r in self.R])
        self.pid = np.array([r["parent_id"] for r in self.R]); self.parents = sorted(set(self.pid))
        pix = {p: i for i, p in enumerate(self.parents)}; self.pidx = np.array([pix[p] for p in self.pid])
        self.cell = self.y * 4 + np.array([LV.index(e) for e in self.lv])
        self.period = np.array([r.get("period", "") for r in self.R])


def load_seeds(src, tag, sp):
    """[(scores aligned to Split rows, tau, seed)] for one method."""
    sp = base_split(sp); out = []
    for f in sorted(glob.glob(os.path.join(ROOT, src, f"{tag}_seed*.json")), key=lambda f: int(f.rsplit("seed", 1)[1][:-5])):
        d = json.load(open(f))
        if "preds" not in d:  # e.g. skip placeholders for dropped runs
            continue
        P = d["preds"] if sp in d["preds"] else d.get("sealed", {}).get("preds", {})
        I = d["row_ids"] if sp in d["row_ids"] else d.get("sealed", {}).get("row_ids", {})
        if sp not in P:
            continue
        out.append((dict(zip(I[sp], P[sp])), d["locked_thresholds"]["tau_cal_all5"], f.rsplit("seed", 1)[1][:-5]))
    return out


def load_pretrained(sp):
    sp = base_split(sp)
    val = json.load(open(os.path.join(ROOT, "data/split_val.json")))
    cal_set = set(json.load(open(os.path.join(ROOT, "data_v2/val_manifest.json")))["val_calib_parent_ids"])
    cal_idx = [i for i, r in enumerate(val) if r["parent_id"] in cal_set]
    # word-count probe (spec amendment): score = -word count, threshold locked on val_calib humans like every detector
    cr = [val[i] for i in cal_idx if int(val[i]["origin_label"]) == 0 and val[i]["edit_level"] in LV]
    wc_tau = conformal([-wc(r["text"]) for r in cr], [r["parent_id"] for r in cr])
    rows = json.load(open(os.path.join(ROOT, SPLIT_FILES[sp])))
    probe = {"word_count_probe": [({r["id"]: -float(wc(r["text"])) for r in rows}, wc_tau, "0")]}
    pv3 = json.load(open(os.path.join(ROOT, "results_v3/preds_pretrained_and_probes_v3.json")))
    srcs = {"test_confirm": pv3, "test_confirm2": json.load(open(os.path.join(ROOT, "results_e2/preds_pretrained_e2.json")))}
    fp3 = os.path.join(ROOT, "results_icml/final/preds_pretrained_c3_seed0.json")
    if sp == "test_confirm3" and os.path.exists(fp3):
        srcs["test_confirm3"] = json.load(open(fp3))
    if sp not in srcs:
        return probe
    S = srcs[sp]; ids = S["row_ids"][sp] if "row_ids" in S and sp in S["row_ids"] else [r["id"] for r in json.load(open(os.path.join(ROOT, SPLIT_FILES[sp])))]
    out = {}
    for det in PRETRAINED:
        if det not in S or sp not in S[det]:
            continue
        cal = np.asarray(pv3[det]["val"], float)[cal_idx]; cr = [val[i] for i in cal_idx]
        hum = [i for i, r in enumerate(cr) if int(r["origin_label"]) == 0]
        tau = conformal(cal[hum], [cr[i]["parent_id"] for i in hum])
        out[det] = [(dict(zip(ids, S[det][sp])), tau, "0")]
    out.update(probe)
    return out


def metrics(S, sc, tau, w=None, t5=None):
    w = np.ones(len(S.y)) if w is None else w
    pred = sc > tau; corr = pred == (S.y == 1)
    acc = [float((corr * w)[S.cell == c].sum() / w[S.cell == c].sum()) for c in range(8)]
    hum, h3, mac = S.y == 0, (S.y == 0) & (S.lv == "E3"), S.y == 1
    m = {"wg": 100 * min(acc), "fpr_h": 100 * (pred * w)[hum].sum() / w[hum].sum(), "fpr_hE3": 100 * (pred * w)[h3].sum() / w[h3].sum(),
         "tpr_m": 100 * (pred * w)[mac].sum() / w[mac].sum(), "human_acc": 100 * np.mean(acc[:4]), "auroc": auroc(S.y, sc, None if w is None else w)}
    if t5 is not None:
        p5 = sc > t5; c5 = p5 == (S.y == 1)
        m["wg_t5"] = 100 * min(float((c5 * w)[S.cell == c].sum() / w[S.cell == c].sum()) for c in range(8))
        m["rec_t5"] = 100 * (p5 * w)[mac].sum() / w[mac].sum()
    return m


def seed_arrays(S, seeds):
    arr = []
    for m, tau, sd in seeds:
        sc = np.array([m[i] for i in S.ids], float)
        t5 = conformal(sc[S.y == 0], S.pid[S.y == 0])
        arr.append((sc, tau, t5, sd))
    return arr


def anchor30(S, sc, ndraw=100):
    res = []
    for r in range(ndraw):
        ch = np.random.default_rng(10000 + r).choice(S.parents, 30, replace=False); a = np.isin(S.pid, ch)
        t = conformal(sc[a & (S.y == 0)], S.pid[a & (S.y == 0)])
        keep = ~a; corr = (sc[keep] > t) == (S.y[keep] == 1)
        res.append(100 * min(np.mean(corr[S.cell[keep] == c]) for c in range(8)))
    return float(np.mean(res))


def summarize(S, arr):
    per = [metrics(S, sc, tau, t5=t5) for sc, tau, t5, _ in arr]
    out = {k: [float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per]))] for k in per[0]}
    out["wg_anchor30"] = [float(np.mean([anchor30(S, sc) for sc, *_ in arr])), 0.0]
    out["n_seeds"] = len(arr); out["per_seed_wg"] = [round(p["wg"], 1) for p in per]
    return out


def boot_contrast(S, A, B, keys, mask=None, rng_seed=42):
    """Two-level bootstrap of mean(A) - mean(B) over parents (shared draw) and seeds (independent)."""
    rng = np.random.default_rng(rng_seed); P = len(S.parents)
    mask = np.ones(len(S.y), bool) if mask is None else mask
    out = {k: [] for k in keys}
    point = {k: float(np.mean([metrics(S, sc, tau, w=mask.astype(float), t5=t5)[k] for sc, tau, t5, _ in A])
                     - np.mean([metrics(S, sc, tau, w=mask.astype(float), t5=t5)[k] for sc, tau, t5, _ in B])) for k in keys}
    for b in range(NB):
        w = np.bincount(rng.integers(0, P, P), minlength=P)[S.pidx].astype(float) * mask
        ia, ib = rng.integers(0, len(A), len(A)), rng.integers(0, len(B), len(B))
        ma = [metrics(S, *A[i][:2], w=w, t5=A[i][2]) for i in ia]; mb = [metrics(S, *B[i][:2], w=w, t5=B[i][2]) for i in ib]
        for k in keys:
            out[k].append(np.mean([m[k] for m in ma]) - np.mean([m[k] for m in mb]))
    return {k: {"diff": round(point[k], 2), "ci95": [round(float(x), 2) for x in np.percentile(out[k], [2.5, 97.5])]} for k in keys}


def boot_period(S, A, keys, rng_seed=7):
    """C3-H1: post - pre within one method; parents resampled within each period, seeds resampled."""
    rng = np.random.default_rng(rng_seed)
    post, pre = S.period == "post", S.period == "pre"
    pp = {g: sorted(set(S.pidx[m])) for g, m in (("post", post), ("pre", pre))}
    def stat(w):
        r = {}
        for k in keys:
            a = [metrics(S, sc, tau, w=w * post, t5=t5)[k] for sc, tau, t5, _ in A]
            b = [metrics(S, sc, tau, w=w * pre, t5=t5)[k] for sc, tau, t5, _ in A]
            r[k] = (a, b)
        return r
    base = stat(np.ones(len(S.y)))
    point = {k: round(float(np.mean(v[0]) - np.mean(v[1])), 2) for k, v in base.items()}
    draws = {k: [] for k in keys}
    for b in range(NB // 2):
        cnt = np.zeros(len(S.parents))
        for g in ("post", "pre"):
            ids = np.array(pp[g]); cnt[ids] += np.bincount(rng.choice(len(ids), len(ids)), minlength=len(ids))
        w = cnt[S.pidx]; si = rng.integers(0, len(A), len(A))
        Ab = [A[i] for i in si]
        for k in keys:
            a = [metrics(S, sc, tau, w=w * post, t5=t5)[k] for sc, tau, t5, _ in Ab]
            c = [metrics(S, sc, tau, w=w * pre, t5=t5)[k] for sc, tau, t5, _ in Ab]
            draws[k].append(np.mean(a) - np.mean(c))
    return {k: {"post_minus_pre": point[k], "ci95": [round(float(x), 2) for x in np.percentile(draws[k], [2.5, 97.5])],
                "post": round(float(np.mean(base[k][0])), 2), "pre": round(float(np.mean(base[k][1])), 2)} for k in keys}


def main():
    res = {"splits": {}, "contrasts": {}, "contamination": {}}
    for sp in SPLITS:
        S = Split(sp); arrs = {}
        for name, (src, tag) in METHODS.items():
            seeds = load_seeds(src, tag, sp)
            if seeds:
                arrs[name] = seed_arrays(S, seeds)
        for det, seeds in load_pretrained(sp).items():
            arrs[det] = seed_arrays(S, seeds)
        res["splits"][sp] = {n: summarize(S, a) for n, a in arrs.items()}
        print(f"\n== {sp} ({len(S.parents)} parents)")
        for n, m in res["splits"][sp].items():
            print(f"  {n:22s} WG {m['wg'][0]:5.1f}±{m['wg'][1]:4.1f}  FPRh {m['fpr_h'][0]:4.1f}  FPRhE3 {m['fpr_hE3'][0]:5.1f}  TPR {m['tpr_m'][0]:5.1f}"
                  f"  AUROC {m['auroc'][0]:5.1f}  WG@5% {m['wg_t5'][0]:5.1f}  rec@5% {m['rec_t5'][0]:5.1f}  anch30 {m['wg_anchor30'][0]:5.1f}  (n={m['n_seeds']})", flush=True)
        keys = ["wg", "fpr_hE3", "wg_t5", "auroc"]
        pairs = [("8B pair", "8B aug"), ("8B pair", "MB pair"), ("8B pair", "8B SupCon"), ("8B pair", "8B GroupDRO"),
                 ("8B pair", "8B logit-pair"), ("1.7B pair", "1.7B aug"), ("4B pair", "4B aug")]
        pre = [d for d in PRETRAINED if d in arrs]
        if pre:  # best public/zero-shot detector by locked-threshold WG on this split (pre-registered as "the best")
            best = max(pre, key=lambda d: res["splits"][sp][d]["wg"][0])
            if sp.endswith("_lenmatch") and res.get("best_pretrained", {}).get(base_split(sp)) in pre:
                best = res["best_pretrained"][base_split(sp)]  # same detector as on the full split (spec amendment)
            res.setdefault("best_pretrained", {})[sp] = best; pairs.append(("8B pair", best))
        for a, b in pairs:
            if a in arrs and b in arrs:
                res["contrasts"][f"{sp}/{a} vs {b}"] = c = boot_contrast(S, arrs[a], arrs[b], keys)
                print(f"  [{a} - {b}] " + "  ".join(f"{k} {v['diff']:+.1f} [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}]" for k, v in c.items()), flush=True)
        if sp == "test_confirm3":
            for n in [m for m in ("8B pair", "8B aug", "1.7B pair", "MB pair") if m in arrs] + pre:
                res["contamination"][n] = c = boot_period(S, arrs[n], ["wg", "human_acc", "auroc"])
                print(f"  [post-pre {n}] " + "  ".join(f"{k} {v['post_minus_pre']:+.1f} [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}]" for k, v in c.items()), flush=True)
    out = os.path.join(ROOT, "results_icml", f"final_analysis{'_dev' if DEV else ''}{'_unsealed' if UNSEAL else ''}.json")
    json.dump(res, open(out, "w"), indent=1); print("saved", out)


if __name__ == "__main__":
    main()

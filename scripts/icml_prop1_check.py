#!/usr/bin/env python3
"""Proposition 1 check: decompose the false-accusation rate on restructured human text (E3) at the locked threshold.
For each run: s0 = human E0 scores on the split, Delta = s(E3) - s(E0) per human parent, tau = locked threshold.
  bound(delta) = Pr[s0 > tau - delta] + Pr[Delta > delta];  report min over delta, the minimiser's two terms, and the actual E3 FPR.
Also Delta on val_calib (training editors) vs the test split (unseen editor): does edit invariance transfer?
Usage: python scripts/icml_prop1_check.py [--dev] [--unseal]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import icml_final_analysis as A  # noqa: E402

METH = [m for m in ("8B pair", "8B aug", "8B SupCon", "8B GroupDRO", "8B logit-pair", "1.7B pair", "1.7B aug", "MB pair", "MB aug") if m in A.METHODS]


def to_logit(p):
    """Scores are sigmoid probabilities; Prop. 1's linear-head statement is about the logit s = w'z + b."""
    p = np.asarray(p, float)
    if p.min() < 0 or p.max() > 1:
        return p
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def deltas(S, sc):
    idx = {(p, o, e): i for i, (p, o, e) in enumerate(zip(S.pid, S.y, S.lv))}
    ps = [p for p in S.parents if (p, 0, "E3") in idx and (p, 0, "E0") in idx]
    return np.array([sc[idx[(p, 0, "E3")]] - sc[idx[(p, 0, "E0")]] for p in ps]), np.array([sc[idx[(p, 0, "E0")]] for p in ps]), \
        np.array([sc[idx[(p, 0, "E3")]] for p in ps])


def val_deltas(src, tag, seed):
    d = json.load(open(os.path.join(A.ROOT, src, f"{tag}_seed{seed}.json")))
    if "val_calib" not in d["preds"]:
        return None
    val = {r["id"]: r for r in json.load(open(os.path.join(A.ROOT, "data/split_val.json")))}
    sc = dict(zip(d["row_ids"]["val_calib"], to_logit(d["preds"]["val_calib"])))
    by = {}
    for i, s in sc.items():
        r = val.get(i)
        if r and int(r["origin_label"]) == 0 and r["edit_level"] in ("E0", "E3"):
            by.setdefault(r["parent_id"], {})[r["edit_level"]] = s
    return np.array([v["E3"] - v["E0"] for v in by.values() if len(v) == 2])


def main():
    out = {}
    for sp in A.SPLITS:
        if sp == "test_confirm":
            continue
        S = A.Split(sp); out[sp] = {}
        print(f"\n== {sp}")
        for m in METH:
            src, tag = A.METHODS[m]; seeds = A.load_seeds(src, tag, sp)
            if not seeds:
                continue
            rec = []
            for (mp, tau, sd), (sc, *_ ) in zip(seeds, A.seed_arrays(S, seeds)):
                sc, tau = to_logit(sc), float(to_logit([tau])[0]) if 0 <= tau <= 1 and sc.min() >= 0 and sc.max() <= 1 else tau
                D, s0, s3 = deltas(S, sc); scale = np.median(sc[S.y == 1]) - np.median(sc[S.y == 0])
                grid = np.linspace(0, max(1e-6, D.max()), 400)
                b = np.array([np.mean(s0 > tau - g) + np.mean(D > g) for g in grid]); j = int(b.argmin())
                vd = val_deltas(src, tag, sd)
                rec.append({"actual_E3_fpr": 100 * np.mean(s3 > tau), "bound": 100 * b[j], "term_unedited": 100 * np.mean(s0 > tau - grid[j]),
                            "term_drift": 100 * np.mean(D > grid[j]), "E0_fpr": 100 * np.mean(s0 > tau),
                            "drift_test_p90_rel": float(np.percentile(D, 90) / scale),
                            "drift_val_p90_rel": None if vd is None or len(vd) == 0 else float(np.percentile(vd, 90) / scale)})
            agg = {k: float(np.mean([r[k] for r in rec if r[k] is not None])) if any(r[k] is not None for r in rec) else None for k in rec[0]}
            out[sp][m] = agg
            print(f"  {m:14s} E3 FPR {agg['actual_E3_fpr']:5.1f} <= bound {agg['bound']:5.1f} (unedited {agg['term_unedited']:4.1f} + drift {agg['term_drift']:4.1f});"
                  f" E0 FPR {agg['E0_fpr']:4.1f}; p90 drift/sep: val {agg['drift_val_p90_rel'] if agg['drift_val_p90_rel'] is None else round(agg['drift_val_p90_rel'], 3)}"
                  f" test {agg['drift_test_p90_rel']:.3f}  (n={len(rec)})")
    fn = os.path.join(A.ROOT, "results_icml", f"prop1_check{'_dev' if A.DEV else ''}{'_unsealed' if A.UNSEAL else ''}.json")
    json.dump(out, open(fn, "w"), indent=1); print("saved", fn)


if __name__ == "__main__":
    main()

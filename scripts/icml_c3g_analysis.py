#!/usr/bin/env python3
"""Confirm-3G analysis (docs/CONFIRM3G_SPEC_2026-11-15.md). Reads results_icml/c3g/c3g_rescore_seed0.json.
1. Re-scoring check: decision flips of re-scored (bf16) Confirm-3 vs the sealed predictions at each run's locked tau.
   If any run flips > 1% of Confirm-3 texts, every run's tau is re-locked on re-scored val_calib (pre-registered rule).
2. Metrics per method on Confirm-3G (and on re-scored Confirm-3 for the paired editor comparison), pre-registered
   contrasts G-H1..H3 with the paper's seed x parent bootstrap.
Output: results_icml/c3g/c3g_analysis.json"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import icml_final_analysis as FA  # noqa: E402
ROOT = FA.ROOT
FA.SPLIT_FILES["test_confirm3g"] = "data_c3g/split_test_confirm3g.json"
RES = json.load(open(os.path.join(ROOT, "results_icml/c3g/c3g_rescore_seed0.json")))
NAMES = {"8B pair": "f_8b_pair", "8B aug": "f_8b_aug", "8B SupCon": "f_8b_supcon", "8B GroupDRO": "f_8b_gdro",
         "8B logit-pair": "f_8b_logitpair", "1.7B pair": "f_1p7b_pair", "1.7B aug": "f_1p7b_aug"}


def taus():
    """Locked tau per run from the original result files, and the sealed Confirm-3 predictions."""
    out = {}
    for run in RES["runs"]:
        d = json.load(open(os.path.join(ROOT, "results_icml/final", f"{run}.json")))
        out[run] = (d["locked_thresholds"]["tau_cal_all5"], dict(zip(d["sealed"]["row_ids"]["test_confirm3"], d["sealed"]["preds"]["test_confirm3"])))
    return out


def main():
    T = taus(); ids = RES["row_ids"]; report = {"rescore_check": {}}
    worst = 0.0
    for run, (tau, sealed) in T.items():
        new = dict(zip(ids["test_confirm3"], RES["runs"][run]["test_confirm3"]))
        flips = np.mean([(new[i] > tau) != (sealed[i] > tau) for i in new]) * 100
        mad = float(np.max(np.abs([new[i] - sealed[i] for i in new])))
        report["rescore_check"][run] = {"flip_pct": round(float(flips), 3), "max_abs_diff": mad}; worst = max(worst, flips)
    relock = worst > 1.0
    report["relocked"] = relock
    print(f"re-scoring check: worst decision flips {worst:.3f}% -> {'RE-LOCK on re-scored val_calib' if relock else 'keep locked thresholds'}")
    val = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "data/split_val.json")))}
    report["splits"], report["contrasts"] = {}, {}
    for sp in ("test_confirm3g", "test_confirm3"):
        S = FA.Split(sp); arrs = {}
        for name, tag in NAMES.items():
            seeds = []
            for run in sorted(r for r in RES["runs"] if r.startswith(tag + "_seed")):
                tau = T[run][0]
                if relock:
                    cal = [(val[i], s) for i, s in zip(ids["val_calib"], RES["runs"][run]["val_calib"])
                           if int(val[i]["origin_label"]) == 0 and val[i]["edit_level"] in FA.LV]
                    tau = FA.conformal([s for _, s in cal], [r["parent_id"] for r, _ in cal])
                seeds.append((dict(zip(ids[sp], RES["runs"][run][sp])), tau, run.rsplit("seed", 1)[1]))
            if seeds:
                arrs[name] = FA.seed_arrays(S, seeds)
        if sp == "test_confirm3g" and RES.get("pretrained"):
            pv3 = json.load(open(os.path.join(ROOT, "results_v3/preds_pretrained_and_probes_v3.json")))
            vrows = json.load(open(os.path.join(ROOT, "data/split_val.json")))
            cset = set(json.load(open(os.path.join(ROOT, "data_v2/val_manifest.json")))["val_calib_parent_ids"])
            ci = [i for i, r in enumerate(vrows) if r["parent_id"] in cset and int(r["origin_label"]) == 0]
            for det in FA.PRETRAINED:
                if det in RES["pretrained"]:
                    tau = FA.conformal(np.asarray(pv3[det]["val"], float)[ci], [vrows[i]["parent_id"] for i in ci])
                    arrs[det] = FA.seed_arrays(S, [(dict(zip(ids[sp], RES["pretrained"][det][sp])), tau, "0")])
        report["splits"][sp] = {n: FA.summarize(S, a) for n, a in arrs.items()}
        print(f"\n== {sp} ({len(S.parents)} parents)")
        for n, m in report["splits"][sp].items():
            print(f"  {n:22s} WG {m['wg'][0]:5.1f}±{m['wg'][1]:4.1f}  FPRh {m['fpr_h'][0]:4.1f}  FPRhE3 {m['fpr_hE3'][0]:5.1f}  TPR {m['tpr_m'][0]:5.1f}"
                  f"  AUROC {m['auroc'][0]:5.1f}  WG@5% {m['wg_t5'][0]:5.1f}  anch30 {m['wg_anchor30'][0]:5.1f}  (n={m['n_seeds']})", flush=True)
        if sp != "test_confirm3g":
            continue
        pairs = [("8B pair", "8B aug"), ("8B pair", "8B SupCon"), ("8B pair", "8B GroupDRO"), ("8B pair", "8B logit-pair"), ("1.7B pair", "1.7B aug")]
        pre = [d for d in FA.PRETRAINED if d in arrs]
        if pre:
            best = max(pre, key=lambda d: report["splits"][sp][d]["wg"][0]); report["best_pretrained"] = best; pairs.append(("8B pair", best))
        for a, b in pairs:
            if a in arrs and b in arrs:
                report["contrasts"][f"{a} vs {b}"] = c = FA.boot_contrast(S, arrs[a], arrs[b], ["wg", "fpr_hE3", "wg_t5", "auroc"])
                print(f"  [{a} - {b}] " + "  ".join(f"{k} {v['diff']:+.1f} [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}]" for k, v in c.items()), flush=True)
    c = report["contrasts"].get("8B pair vs 8B aug")
    if c:
        p = report["splits"]["test_confirm3g"]["8B pair"]["fpr_hE3"][0]
        report["claims"] = {"G-H1": c["fpr_hE3"]["ci95"][1] < 0, "G-H2": c["wg"]["ci95"][0] > 0, "G-H3": p <= 10.0}
        print("\nclaims:", report["claims"])
    json.dump(report, open(os.path.join(ROOT, "results_icml/c3g/c3g_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

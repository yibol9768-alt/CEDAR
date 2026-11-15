#!/usr/bin/env python3
"""Free probe: 5-seed logit ensemble per E2 config, threshold re-locked on val_calib humans (cluster conformal).
Reports Confirm / Confirm-2 WG and AUROC for single seeds (mean, min) vs the ensemble. No training."""
import glob, json, os
from collections import Counter
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LV = ["E0", "E1", "E2", "E3"]
def rows(p):
    R = json.load(open(p))
    return {r["id"]: r for r in R}
VAL = rows(os.path.join(ROOT, "data", "split_val.json"))
CAL = set(json.load(open(os.path.join(ROOT, "data_v2", "val_manifest.json")))["val_calib_parent_ids"])
SPL = {"test_confirm": rows(os.path.join(ROOT, "data_v2", "split_test_confirm.json")),
       "test_confirm2": rows(os.path.join(ROOT, "data_e2", "split_test_confirm2.json"))}
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def conformal(s, pids, a=0.05):
    c = Counter(pids); w = np.array([1 / c[p] for p in pids])
    for t in np.unique(s):
        if (1 + w[s > t].sum()) / (len(c) + 1) <= a + 1e-12: return float(t)
    return float("inf")
def auroc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def evalsplit(sp, ids, s, tau):
    R = SPL[sp]; keep = [i for i, k in enumerate(ids) if R[k]["edit_level"] in LV]
    y = np.array([int(R[ids[i]]["origin_label"]) for i in keep]); lv = np.array([R[ids[i]]["edit_level"] for i in keep]); s = s[keep]
    wg = min(np.mean((s[(y == o) & (lv == e)] > tau) == (o == 1)) for o in (0, 1) for e in LV)
    return 100 * wg, 100 * auroc(y, s)
out = {}
for run in sorted({os.path.basename(f).rsplit("_seed", 1)[0] for f in glob.glob(os.path.join(ROOT, "results_e2/model_preds/e2_*_seed*.json"))}):
    D = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, f"results_e2/model_preds/{run}_seed*.json")))]
    vid = D[0]["row_ids"]["val_calib"]
    hum = [i for i, k in enumerate(vid) if int(VAL[k]["origin_label"]) == 0 and VAL[k]["edit_level"] in LV]
    pids = [VAL[vid[i]]["parent_id"] for i in hum]
    res = {}
    for sp in SPL:
        ids = D[0]["row_ids"][sp]
        single = [evalsplit(sp, ids, np.asarray(d["preds"][sp], float), d["locked_thresholds"]["tau_cal_all5"]) for d in D]
        ens_v = np.mean([logit(d["preds"]["val_calib"]) for d in D], 0)
        ens_s = np.mean([logit(d["preds"][sp]) for d in D], 0)
        tau = conformal(ens_v[hum], pids)
        e = evalsplit(sp, ids, ens_s, tau)
        res[sp] = {"single_wg_mean": round(np.mean([x[0] for x in single]), 1), "single_wg_min": round(min(x[0] for x in single), 1),
                   "ens_wg": round(e[0], 1), "ens_auroc": round(e[1], 1)}
    out[run] = res
    print(f"{run:24s}", " | ".join(f"{sp[5:]}: seeds {r['single_wg_mean']:5.1f} (min {r['single_wg_min']:5.1f}) ens {r['ens_wg']:5.1f} auc {r['ens_auroc']:5.1f}" for sp, r in res.items()), flush=True)
json.dump(out, open(os.path.join(ROOT, "results_e2", "seed_ensemble_probe.json"), "w"), indent=1)

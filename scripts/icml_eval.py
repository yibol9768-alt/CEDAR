#!/usr/bin/env python3
"""ICML evaluation over all splits for groups of runs (seeds pooled per tag).
Every run uses its own threshold locked on val_calib humans (5% cluster FPR, `tau_cal_all5`).
Pretrained detectors from results_e2/preds_pretrained_e2.json (Confirm-2 only) get thresholds from
results_v3/preds_pretrained_and_probes_v3.json val_calib if present, else are reported threshold-free (AUROC only).
Usage: python scripts/icml_eval.py TAG_GLOB [TAG_GLOB ...]    e.g. 'results_icml/p1/qwen3_8b_pairx_seed*.json'
Writes results_icml/eval_<name>.json when --out NAME is given. test_confirm3 is never read (sealed until the final run)."""
import glob, json, os, sys
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LV4 = ["E0", "E1", "E2", "E3"]


def rows(rel):
    return {r["id"]: r for r in json.load(open(os.path.join(ROOT, rel)))}


SPLITS = {"test_id": "data/split_test_id.json", "test_confirm": "data_v2/split_test_confirm.json",
          "test_confirm2": "data_e2/split_test_confirm2.json", "test_ood_ar": "data/split_test_ood_ar.json",
          "test_ood_llada": "data/split_test_ood_llada.json", "external_esl_full": "data/external_esl_full.json",
          "external_mixset": "data/external_mixset.json", "external_beemo": "data/external_beemo.json",
          "external_mage_wild": "data/external_mage_wild.json"}
R = {k: rows(v) for k, v in SPLITS.items()}


def auroc(y, s):
    y = np.asarray(y); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float("nan") if n1 == 0 or n0 == 0 else 100 * (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rate(m, s, t):
    return 100 * float(np.mean(s[m] > t)) if m.sum() else float("nan")


def split_metrics(sp, ids, p, tau):
    rr = [R[sp][k] for k in ids]; s = np.asarray(p, float)
    y = np.array([int(r["origin_label"]) for r in rr])
    if sp in ("test_id", "test_confirm", "test_confirm2", "test_ood_ar", "test_ood_llada"):
        lv = np.array([r["edit_level"] for r in rr]); keep = np.isin(lv, LV4)
        cells = [(o, e) for o in (0, 1) for e in LV4 if ((y == o) & (lv == e) & keep).any()]
        acc = {f"O{o}{e}": 100 * float(np.mean((s[(y == o) & (lv == e)] > tau) == (o == 1))) for o, e in cells}
        return {"auroc": auroc(y[keep], s[keep]), "wg": min(acc.values()), "fpr_h": rate(keep & (y == 0), s, tau),
                "tpr_m": rate(keep & (y == 1), s, tau), **acc}
    if sp == "external_esl_full":
        lv = np.array([r["edit_level"] for r in rr])
        return {"fpr_E0": rate(lv == "E0", s, tau), "fpr_E1pol": rate(lv == "E1", s, tau)}
    if sp == "external_mixset":
        sub = [r["subset"] for r in rr]; pure = np.array([not r.get("is_completion", False) for r in rr])
        hp = np.array(["polish" in u and o == 0 for u, o in zip(sub, y)]); hz = np.array(["humanize" in u and o == 1 for u, o in zip(sub, y)])
        mp = np.array(["MGT_polish" in u and o == 1 for u, o in zip(sub, y)])
        return {"auroc_pure": auroc(y[pure], s[pure]), "fpr_pure": rate(pure & (y == 0), s, tau), "tpr_pure": rate(pure & (y == 1), s, tau),
                "fpr_llm_polished_h": rate(hp, s, tau), "tpr_humanized_m": rate(hz, s, tau), "tpr_human_polished_m": rate(mp, s, tau)}
    if sp == "external_beemo":
        sub = np.array([r["subset"] for r in rr]); h = sub == "human_raw"
        out = {"auroc": auroc(y, s), "fpr_h": rate(h, s, tau)}
        for k in ("mgt_raw", "mgt_expert_human_edited", "mgt_gpt4o_edited"):
            out[f"tpr_{k[4:]}"] = rate(sub == k, s, tau)
            m = h | (sub == k); out[f"auroc_{k[4:]}"] = auroc(y[m], s[m])
        return out
    if sp == "external_mage_wild":
        f, t = rate(y == 0, s, tau), rate(y == 1, s, tau)
        return {"auroc": auroc(y, s), "fpr_h": f, "tpr_m": t, "bal_acc": 0.5 * (100 - f + t)}
    raise KeyError(sp)


def eval_run(d):
    tau = d["locked_thresholds"]["tau_cal_all5"]; out = {}
    for sp in SPLITS:
        if sp in d["preds"]:
            out[sp] = split_metrics(sp, d["row_ids"][sp], d["preds"][sp], tau)
    return out


def pooled(files):
    per = [eval_run(json.load(open(f))) for f in files]
    agg = defaultdict(lambda: defaultdict(list))
    for m in per:
        for sp, kv in m.items():
            for k, v in kv.items():
                agg[sp][k].append(v)
    return {sp: {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in kv.items()} for sp, kv in agg.items()}


SHOW = {"test_confirm": ["wg", "auroc", "fpr_h", "tpr_m"], "test_confirm2": ["wg", "auroc", "fpr_h", "tpr_m"],
        "test_id": ["wg", "auroc"], "test_ood_ar": ["wg", "auroc"], "test_ood_llada": ["wg", "auroc"],
        "external_esl_full": ["fpr_E0", "fpr_E1pol"],
        "external_mixset": ["auroc_pure", "fpr_pure", "tpr_pure", "fpr_llm_polished_h", "tpr_humanized_m"],
        "external_beemo": ["auroc", "fpr_h", "tpr_raw", "tpr_expert_human_edited", "tpr_gpt4o_edited"],
        "external_mage_wild": ["auroc", "fpr_h", "tpr_m", "bal_acc"]}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if name:
        args = [a for a in args if a != name]
    res = {}
    for g in args:
        files = sorted(glob.glob(g if os.path.isabs(g) else os.path.join(ROOT, g)))
        if not files:
            continue
        tag = os.path.basename(files[0]).rsplit("_seed", 1)[0]
        res[tag] = pooled(files)
    for sp, keys in SHOW.items():
        print(f"\n== {sp}")
        for tag, m in res.items():
            if sp in m:
                print(f"  {tag:28s} " + "  ".join(f"{k} {m[sp][k][0]:5.1f}±{m[sp][k][1]:4.1f}" for k in keys if k in m[sp]) + f"  (n={m[sp][keys[0]][2]})")
    if name:
        json.dump(res, open(os.path.join(ROOT, "results_icml", f"eval_{name}.json"), "w"), indent=1)

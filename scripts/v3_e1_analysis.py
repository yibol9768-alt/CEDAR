"""E1 analysis per the pre-registered rules in docs/v3_results_2026-11-14.md.
Reuses v3_step5 metric + bootstrap code; prints per-variant summaries and paired tests (B=5000, fixed seed)."""
import os, sys, json, glob, multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_step5_statistical_analysis as S5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "results_v3", "model_preds_v3")
raw, _ = S5.load_all_splits()
y_tc = np.array([r["origin_label"] for r in raw["test_confirm"]]); y_tid = np.array([r["origin_label"] for r in raw["test_id"]])
lv = {"E0": 0, "E1": 2, "E2": 4, "E3": 6, "E4": 8}
g_tc = np.array([lv[r["edit_level"]] + int(r["origin_label"]) for r in raw["test_confirm"]])
g_tid = np.array([lv[r["edit_level"]] + int(r["origin_label"]) for r in raw["test_id"]])
ms_all = raw["external_mixset"]; ms_idx = np.array([i for i, r in enumerate(ms_all) if not r.get("is_completion", False)])
ms_items = [ms_all[i] for i in ms_idx]; y_ms = np.array([r["origin_label"] for r in ms_items])
ll = raw["test_ood_llada"]; ll_idx = np.array([i for i, r in enumerate(ll) if 120 <= r.get("wc", len(r["text"].split())) <= 220])
ll_items = [ll[i] for i in ll_idx]; y_ll = np.array([r["origin_label"] for r in ll_items])
y_bm = np.array([r["origin_label"] for r in raw["external_beemo"]])


def load(run):
    out = []
    for f in sorted(glob.glob(os.path.join(PRED, f"{run}_seed*.json")), key=lambda p: int(p.rsplit("seed", 1)[1][:-5])):
        out.append(json.load(open(f)))
    return out


def fns(pls):
    tau = [float(p["locked_thresholds"]["tau_cal_all5"]) for p in pls]
    tc = [((np.asarray(p["preds"]["test_confirm"]) > t).astype(int) == y_tc).astype(float) for p, t in zip(pls, tau)]
    ti = [((np.asarray(p["preds"]["test_id"]) > t).astype(int) == y_tid).astype(float) for p, t in zip(pls, tau)]
    s_tc = [np.asarray(p["preds"]["test_confirm"], float) for p in pls]
    s_ms = [np.asarray(p["preds"]["external_mixset"], float)[ms_idx] for p in pls]
    s_ll = [np.asarray(p["preds"]["test_ood_llada"], float)[ll_idx] for p in pls]
    s_bm = [np.asarray(p["preds"]["external_beemo"], float) for p in pls]

    def wg(g, corr, n):
        def f(b):
            gb = g[b]; cnt = np.bincount(gb, minlength=n)[:8]
            if np.any(cnt == 0):
                return float("nan")
            return float(np.mean([np.min(np.bincount(gb, weights=c[b], minlength=n)[:8] / cnt) for c in corr]))
        return f
    auc = lambda y, ss: (lambda b: float(np.mean([S5.safe_auroc(y[b], s[b]) for s in ss])))
    return {"tc_wg": (raw["test_confirm"], wg(g_tc, tc, 8)), "tid_wg": (raw["test_id"], wg(g_tid, ti, 10)),
            "tc_auc": (raw["test_confirm"], auc(y_tc, s_tc)), "ms_auc": (ms_items, auc(y_ms, s_ms)),
            "ll_lm_auc": (ll_items, auc(y_ll, s_ll)), "bm_auc": (raw["external_beemo"], auc(y_bm, s_bm))}


VARIANTS = ["mb_v3_ablation_no_dte", "mb_v3_nodte_fit600_no_nsd", "mb_v3_nodte_no_nsd", "mb_v3_strong_combined_no_nsd",
            "mb_v3_strong_combined_fit600_no_nsd", "mb_v3_cedar_split_op",
            "db_v3_cedar_no_dte", "db_v3_nodte_no_nsd", "db_v3_strong_combined_no_nsd", "db_v3_cedar_split_op"]
PAIRS = [("mb_v3_ablation_no_dte", "mb_v3_nodte_fit600_no_nsd"), ("mb_v3_ablation_no_dte", "mb_v3_nodte_no_nsd"),
         ("mb_v3_ablation_no_dte", "mb_v3_strong_combined_no_nsd"), ("mb_v3_nodte_fit600_no_nsd", "mb_v3_strong_combined_fit600_no_nsd"),
         ("db_v3_cedar_no_dte", "db_v3_nodte_no_nsd"), ("db_v3_cedar_no_dte", "db_v3_strong_combined_no_nsd")]
F = {}
TASKS = []
for v in VARIANTS:
    pls = load(v)
    if pls:
        F[v] = fns(pls)
for a, b in PAIRS:
    if a in F and b in F:
        for m in ["tc_wg", "tid_wg", "tc_auc", "ms_auc", "ll_lm_auc", "bm_auc"]:
            TASKS.append((a, b, m))


def run(i):
    a, b, m = TASKS[i]
    items, fa = F[a][m]; _, fb = F[b][m]
    return S5.paired_parent_bootstrap_estimand_diff(items, fa, fb, n_boot=5000)


if __name__ == "__main__":
    summ = {}
    for v in VARIANTS:
        if v not in F:
            continue
        n = len(load(v)); full = None
        row = {}
        for m in ["tc_wg", "tid_wg", "tc_auc", "ms_auc", "ll_lm_auc", "bm_auc"]:
            items, f = F[v][m]
            row[m] = round(f(np.arange(len(items))) * 100, 1)
        summ[v] = {"n_seeds": n, **row}
        print(f"{v:40s} n={n} " + " ".join(f"{k}={x}" for k, x in row.items()), flush=True)
    with mp.get_context("fork").Pool(min(12, len(TASKS))) as pool:
        res = pool.map(run, range(len(TASKS)))
    tests = {}
    for (a, b, m), r in zip(TASKS, res):
        tests.setdefault(f"{a}__vs__{b}", {})[m] = r
        print(f"{a} vs {b} {m:10s} diff={r['diff']*100:+.2f}pp CI=[{r['ci95'][0]*100:+.2f},{r['ci95'][1]*100:+.2f}] p={r['p_boot']:.4f}", flush=True)
    json.dump({"seed_mean_metrics_pct": summ, "paired_tests": tests}, open(os.path.join(ROOT, "results_v3", "e1_analysis.json"), "w"), indent=2)

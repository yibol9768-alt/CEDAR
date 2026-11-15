import os
import glob
import math
import json
import hashlib
from collections import defaultdict, Counter

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "/content/scripts/v3_step5.py"))))
if os.path.exists("/content/cedar_data") and os.path.exists("/content/cedar_data_v2"):
    V1_DIR = "/content/cedar_data"
    V2_DIR = "/content/cedar_data_v2"
    RES_DIR = "/content/cedar_results_v3"
else:
    V1_DIR = os.path.join(_ROOT, "data")
    V2_DIR = os.path.join(_ROOT, "data_v2")
    RES_DIR = os.path.join(_ROOT, "results_v3")

PRED_DIR = os.path.join(RES_DIR, "model_preds_v3")
os.makedirs(RES_DIR, exist_ok=True)


def exact_sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def det_rank_key(tag: str, pid: str) -> str:
    return hashlib.sha256(f"{tag}|{pid}".encode("utf-8")).hexdigest()


def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def tpr_at_target_fpr(y_true, y_score, target_fpr=0.05):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(tpr[valid[-1]])


def cluster_conformal_upper_threshold(hum_scores: np.ndarray, hum_pids: list, alpha: float = 0.05) -> float:
    """
    Parent-cluster split-conformal risk control threshold for strict rule `score > tau`.
    Guarantees E[L_{K+1}(tau)] <= alpha across exchangeable human parent clusters without
    requiring passage-level independence across sibling edits of the same parent.
    """
    scores = np.asarray(hum_scores, dtype=np.float64)
    assert len(scores) == len(hum_pids) and len(scores) > 0, "Invalid human calibration inputs"
    cnts = Counter(hum_pids)
    K = len(cnts)
    weights = np.array([1.0 / float(cnts[pid]) for pid in hum_pids], dtype=np.float64)
    for cand in np.unique(np.sort(scores)):
        risk_ub = (1.0 + float(weights[scores > cand].sum())) / float(K + 1)
        if risk_ub <= alpha + 1e-12:
            return float(cand)
    return float("inf")


def cluster_avg_fpr(hum_scores: np.ndarray, hum_pids: list, tau: float) -> float:
    scores = np.asarray(hum_scores, dtype=np.float64)
    by_p = defaultdict(list)
    for s, pid in zip(scores, hum_pids):
        by_p[pid].append(float(s > tau))
    return float(np.mean([np.mean(v) for v in by_p.values()]))


def compute_full_split_fingerprint(raw_splits: dict, val_manifest: dict) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(val_manifest, sort_keys=True).encode("utf-8"))
    for k in sorted(raw_splits.keys()):
        h.update(f"SPLIT:{k}|".encode("utf-8"))
        for r in raw_splits[k]:
            th = exact_sha256(r["text"])
            p_key = r.get("parent_id", r.get("prompt_id", r["id"]))
            reg = r.get("edit_register", 0)
            h.update(f"{r['id']}|{p_key}|{r['origin_label']}|{r.get('edit_level','E0')}|{reg}|{th}|".encode("utf-8"))
    return h.hexdigest()[:24]


def resolve_cluster_id(it: dict, allow_record_id: bool = False) -> str:
    if it.get("parent_id"):
        return str(it["parent_id"])
    if it.get("prompt_id"):
        return str(it["prompt_id"])
    rid = str(it.get("id", ""))
    if rid.startswith("esl_"):
        return rid[:-3] if rid.endswith("_E1") else rid
    if allow_record_id and rid:
        return rid
    raise ValueError(f"Missing cluster identifier (parent_id/prompt_id) on record {it.get('id')}!")


def compute_ece(y_true, probs, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.float64)
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(y_true))
    for b in range(n_bins):
        if b == n_bins - 1:
            m = (probs >= bins[b]) & (probs <= bins[b + 1])
        else:
            m = (probs >= bins[b]) & (probs < bins[b + 1])
        if m.any():
            ece += (m.sum() / n) * abs(y_true[m].mean() - probs[m].mean())
    return float(ece)


def parent_bootstrap_ci(items, scores, metric_fn, n_boot=2000, seed=42, allow_record_id=False):
    rng = np.random.default_rng(seed)
    p_groups = defaultdict(list)
    for idx, it in enumerate(items):
        pid = resolve_cluster_id(it, allow_record_id=allow_record_id)
        p_groups[pid].append(idx)
    pids = list(p_groups.keys())
    K = len(pids)
    y_all = np.array([it["origin_label"] for it in items])
    s_all = np.asarray(scores, dtype=np.float64)
    point = metric_fn(y_all, s_all)

    vals = []
    for _ in range(n_boot):
        chosen = rng.integers(0, K, size=K)
        b_idx = []
        for c in chosen:
            b_idx.extend(p_groups[pids[c]])
        b_idx = np.asarray(b_idx, dtype=np.int64)
        v = metric_fn(y_all[b_idx], s_all[b_idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), (float(lo), float(hi))


def paired_parent_bootstrap_estimand_diff(
    items, eval_fn_a, eval_fn_b, n_boot=5000, seed=42, allow_record_id=False, zero_tol=1e-12
):
    """
    Paired cluster bootstrap by parent_id/prompt_id for the exact table estimand (EvalA(b_idx) - EvalB(b_idx)),
    averaging per-seed metrics in ascending numeric seed order. Uses B=5000 cluster resamples, 95% percentile CI,
    and two-sided sign-tail proportion p_boot (+1 smoothing, zero tolerance 1e-12).
    """
    rng = np.random.default_rng(seed)
    p_groups = defaultdict(list)
    for idx, it in enumerate(items):
        pid = resolve_cluster_id(it, allow_record_id=allow_record_id)
        p_groups[pid].append(idx)
    pids = list(p_groups.keys())
    K = len(pids)
    full_idx = np.arange(len(items), dtype=np.int64)
    pt_a = eval_fn_a(full_idx)
    pt_b = eval_fn_b(full_idx)
    pt_diff = pt_a - pt_b

    diffs = []
    for _ in range(n_boot):
        chosen = rng.integers(0, K, size=K)
        b_idx = []
        for c in chosen:
            b_idx.extend(p_groups[pids[c]])
        b_idx = np.asarray(b_idx, dtype=np.int64)
        va = eval_fn_a(b_idx)
        vb = eval_fn_b(b_idx)
        if not (np.isnan(va) or np.isnan(vb)):
            diffs.append(va - vb)
    diffs = np.asarray(diffs, dtype=np.float64)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_le = (np.sum(diffs <= zero_tol) + 1.0) / (len(diffs) + 1.0)
    p_ge = (np.sum(diffs >= -zero_tol) + 1.0) / (len(diffs) + 1.0)
    p_two = min(1.0, 2.0 * min(p_le, p_ge))
    return {
        "n_clusters": int(K),
        "n_boot": int(len(diffs)),
        "metric_a": round(float(pt_a), 8),
        "metric_b": round(float(pt_b), 8),
        "diff": round(float(pt_diff), 8),
        "ci95": [round(float(lo), 8), round(float(hi), 8)],
        "p_boot": round(float(p_two), 8),
    }


# --- Parallel execution of the paired bootstrap tests (results identical to serial: every test
# uses its own fixed seed). Tasks hold closures, so workers are forked and look tasks up by index.
_DEFERRED_BOOTS = []


class _DeferredBoot:
    def __init__(self, idx):
        self.idx = idx


def _defer_boot(*args, **kwargs):
    _DEFERRED_BOOTS.append((args, kwargs))
    return _DeferredBoot(len(_DEFERRED_BOOTS) - 1)


def _run_deferred_boot(i):
    a, k = _DEFERRED_BOOTS[i]
    return paired_parent_bootstrap_estimand_diff(*a, **k)


_BOOT_RESULTS = {}


def _resolve_deferred_boots(tree):
    """Run every not-yet-computed deferred test (cached across calls), then substitute results into `tree`."""
    import multiprocessing as mp
    import time as _t
    n_proc = int(os.environ.get("CEDAR_STAT_PROCS", str(max(1, (os.cpu_count() or 2) - 2))))
    t0 = _t.time()
    todo = [i for i in range(len(_DEFERRED_BOOTS)) if i not in _BOOT_RESULTS]
    print(f"  [BOOT] running {len(todo)} paired bootstrap tests on {n_proc} processes ...", flush=True)
    if todo:
        with mp.get_context("fork").Pool(n_proc) as pool:
            for j, (i, r) in enumerate(zip(todo, pool.imap(_run_deferred_boot, todo, chunksize=1))):
                _BOOT_RESULTS[i] = r
                if (j + 1) % 20 == 0:
                    print(f"  [BOOT] {j + 1}/{len(todo)} done ({_t.time() - t0:.0f}s)", flush=True)

    def _sub(node):
        for kk, vv in node.items():
            if isinstance(vv, _DeferredBoot):
                node[kk] = _BOOT_RESULTS[vv.idx]
            elif isinstance(vv, dict):
                _sub(vv)
    _sub(tree)
    print(f"  [BOOT] {len(todo)} tests done in {_t.time() - t0:.0f}s", flush=True)


def holm_bonferroni(p_map: dict) -> dict:
    """Holm-Bonferroni step-down FWER adjustment across a pre-specified family of tests."""
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running_max = 0.0
    for idx, (k, p) in enumerate(items):
        val = min(1.0, float(p) * float(m - idx))
        running_max = max(running_max, val)
        adj[k] = round(running_max, 8)
    return adj


def compute_risk_coverage(y_true: np.ndarray, scores: np.ndarray, tau: float, conf: np.ndarray, edit_levels=None):
    order = np.argsort(-conf)
    pred_bin = (scores > tau).astype(np.int64)
    errs = (pred_bin != y_true).astype(np.float64)
    rc_curve = {}
    cov_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for cov in cov_grid:
        k = max(1, int(round(cov * len(y_true))))
        rc_curve[f"cov_{int(cov * 100)}"] = round(float(errs[order[:k]].mean()), 8)
        if edit_levels is not None:
            rc_curve[f"e4_share_cov_{int(cov * 100)}"] = round(float((edit_levels[order[:k]] == "E4").mean()), 8)
    dense_covs = np.linspace(0.2, 1.0, 81)
    dense_risks = [float(errs[order[: max(1, int(round(c * len(y_true))))]].mean()) for c in dense_covs]
    rc_curve["aurc_20_100"] = round(float(np.trapezoid(dense_risks, dense_covs) / 0.8), 8)
    return rc_curve


def evaluate_detector_dict(
    name: str,
    preds_by_split: dict,
    raw_splits: dict,
    llada_lm_idx: np.ndarray,
    llada_ipw_weights: np.ndarray,
    locked_thresholds: dict = None,
    edit_intens_by_split: dict = None,
):
    """
    Evaluates a detector using STRICT conformal thresholds (`score > tau`) calibrated ONLY on
    the untouched source-stratified 60-parent `val_calib` partition (30 MAGE + 30 No Robots).
    """
    cal_rows = raw_splits["val_calib"]
    cal_scores = np.asarray(preds_by_split["val_calib"], dtype=np.float64)

    cal_y = np.array([r["origin_label"] for r in cal_rows])
    cal_e = np.array([r.get("edit_level", "E0") for r in cal_rows])
    cal_pids = [r.get("parent_id", r["id"]) for r in cal_rows]
    hum_all_idx = np.where(cal_y == 0)[0]
    hum_e0_idx = np.where((cal_y == 0) & (cal_e == "E0"))[0]

    tau_all5 = cluster_conformal_upper_threshold(cal_scores[hum_all_idx], [cal_pids[i] for i in hum_all_idx], alpha=0.05)
    tau_e0_5 = cluster_conformal_upper_threshold(cal_scores[hum_e0_idx], [cal_pids[i] for i in hum_e0_idx], alpha=0.05)

    if locked_thresholds is not None:
        # The locked tau was computed from a separate GPU inference pass than the stored preds;
        # non-deterministic GPU kernels give ~1e-5 drift. Locked tau is used either way.
        for _lab, _re, _lk in [("all5", tau_all5, "tau_cal_all5"), ("e0_5", tau_e0_5, "tau_cal_e0_5")]:
            _d = abs(_re - float(locked_thresholds[_lk]))
            assert _d < 1e-3, f"Threshold mismatch ({_lab}) for {name}: {_re} vs {locked_thresholds[_lk]}"
            if _d >= 1e-5:
                print(f"  [THRESH-DRIFT] {name} {_lab}: recomputed={_re:.7f} locked={float(locked_thresholds[_lk]):.7f} |d|={_d:.2e}", flush=True)
        tau_all5 = float(locked_thresholds["tau_cal_all5"])
        tau_e0_5 = float(locked_thresholds["tau_cal_e0_5"])

    res = {
        "name": name,
        "tau_cal_all5": round(tau_all5, 6),
        "tau_cal_e0_5": round(tau_e0_5, 6),
        "realized_val_calib_all_fpr": round(float((cal_scores[cal_y == 0] > tau_all5).mean()), 8),
        "realized_val_calib_e0_fpr": round(float((cal_scores[(cal_y == 0) & (cal_e == "E0")] > tau_e0_5).mean()), 8),
    }

    # 1. Controlled Test-ID: 5-Level Operational Recoverability Curve E0..E4
    tid = raw_splits["test_id"]
    s_tid = np.asarray(preds_by_split["test_id"], dtype=np.float64)
    y_tid = np.array([r["origin_label"] for r in tid])
    e_tid = np.array([r["edit_level"] for r in tid])
    wc_tid = np.array([r.get("wc", len(r["text"].split())) for r in tid])

    recov = {}
    wg_e02, wg_e03, wg_e04 = [], [], []
    ebal_e02, ebal_e03, ebal_e04 = [], [], []
    for lv in ["E0", "E1", "E2", "E3", "E4"]:
        m_lv = (e_tid == lv)
        m_h = m_lv & (y_tid == 0)
        m_m = m_lv & (y_tid == 1)
        auc_lv = safe_auroc(y_tid[m_lv], s_tid[m_lv])
        fpr_all5 = float((s_tid[m_h] > tau_all5).mean())
        fpr_e0_5 = float((s_tid[m_h] > tau_e0_5).mean())
        tpr_all5 = float((s_tid[m_m] > tau_all5).mean())
        tpr_e0_5 = float((s_tid[m_m] > tau_e0_5).mean())
        tpr_orc5 = tpr_at_target_fpr(y_tid[m_lv], s_tid[m_lv], target_fpr=0.05)
        bal_acc = 0.5 * ((1.0 - fpr_all5) + tpr_all5)
        recov[lv] = {
            "auroc": round(auc_lv, 8),
            "fpr_at_tau_all5": round(fpr_all5, 8),
            "tpr_at_tau_all5": round(tpr_all5, 8),
            "fpr_at_tau_e0_5": round(fpr_e0_5, 8),
            "tpr_at_tau_e0_5": round(tpr_e0_5, 8),
            "oracle_tpr_at_5fpr": round(tpr_orc5, 8),
            "bal_acc": round(bal_acc, 8),
        }
        if lv in {"E0", "E1", "E2"}:
            wg_e02.extend([1.0 - fpr_all5, tpr_all5])
            ebal_e02.append(bal_acc)
        if lv in {"E0", "E1", "E2", "E3"}:
            wg_e03.extend([1.0 - fpr_all5, tpr_all5])
            ebal_e03.append(bal_acc)
        wg_e04.extend([1.0 - fpr_all5, tpr_all5])
        ebal_e04.append(bal_acc)

    m_e03 = np.isin(e_tid, ["E0", "E1", "E2", "E3"])
    res["test_id_recoverability"] = recov
    res["test_id_auroc_E0_E3"] = round(safe_auroc(y_tid[m_e03], s_tid[m_e03]), 8)
    res["test_id_worst_group_acc_E0_E2"] = round(min(wg_e02), 8)
    res["test_id_worst_group_acc_E0_E3"] = round(min(wg_e03), 8)
    res["test_id_worst_group_acc_E0_E4"] = round(min(wg_e04), 8)
    res["test_id_worst_edit_bal_acc_E0_E3"] = round(min(ebal_e03), 8)
    res["test_id_mean_bal_acc_E0_E3"] = round(float(np.mean(ebal_e03)), 8)
    res["test_id_mean_bal_acc_E0_E4"] = round(float(np.mean(ebal_e04)), 8)
    res["test_id_human_fpr_E0_E3"] = round(float((s_tid[m_e03 & (y_tid == 0)] > tau_all5).mean()), 8)
    res["test_id_machine_tpr_E0_E3"] = round(float((s_tid[m_e03 & (y_tid == 1)] > tau_all5).mean()), 8)

    if np.min(s_tid) >= 0.0 and np.max(s_tid) <= 1.0:
        res["test_id_ece_E0_E3"] = round(compute_ece(y_tid[m_e03], s_tid[m_e03]), 8)
    else:
        res["test_id_ece_E0_E3"] = None

    # 2. Pristine Untouched Confirmation Benchmark (`test_confirm`: 160 parents, 1,280 passages, Bidirectional E0..E3)
    tc = raw_splits["test_confirm"]
    s_tc = np.asarray(preds_by_split["test_confirm"], dtype=np.float64)
    y_tc = np.array([r["origin_label"] for r in tc])
    e_tc = np.array([r["edit_level"] for r in tc])
    reg_tc = np.array([int(r.get("edit_register", 0)) for r in tc])
    ug_mask = np.array([str(r["parent_id"]).startswith("ugcf_") for r in tc])

    tc_by_lv = {}
    wg_tc, ebal_tc = [], []
    for lv in ["E0", "E1", "E2", "E3"]:
        ml = (e_tc == lv)
        mh = ml & (y_tc == 0)
        mm = ml & (y_tc == 1)
        fpr = float((s_tc[mh] > tau_all5).mean())
        tpr = float((s_tc[mm] > tau_all5).mean())
        b_acc = 0.5 * ((1.0 - fpr) + tpr)
        tc_by_lv[lv] = {
            "auroc": round(safe_auroc(y_tc[ml], s_tc[ml]), 8),
            "fpr_at_tau_all5": round(fpr, 8),
            "tpr_at_tau_all5": round(tpr, 8),
            "bal_acc": round(b_acc, 8),
        }
        wg_tc.extend([1.0 - fpr, tpr])
        ebal_tc.append(b_acc)

    tc_by_reg = {}
    for reg_id, reg_name in [(0, "unedited_E0"), (1, "formal_edited"), (2, "colloquial_humanized")]:
        mr = (reg_tc == reg_id)
        mh = mr & (y_tc == 0)
        mm = mr & (y_tc == 1)
        fpr_r = float((s_tc[mh] > tau_all5).mean())
        tpr_r = float((s_tc[mm] > tau_all5).mean())
        tc_by_reg[reg_name] = {
            "auroc": round(safe_auroc(y_tc[mr], s_tc[mr]), 8),
            "human_fpr_tau_all5": round(fpr_r, 8),
            "machine_tpr_tau_all5": round(tpr_r, 8),
            "bal_acc_tau_all5": round(0.5 * ((1.0 - fpr_r) + tpr_r), 8),
        }

    # Unseen-genre subset (80 ugcf_* parents) vs seen-domain subset (80 nrcf_* + mgcf_* parents)
    wg_ug = []
    for lv in ["E0", "E1", "E2", "E3"]:
        mh = ug_mask & (e_tc == lv) & (y_tc == 0)
        mm = ug_mask & (e_tc == lv) & (y_tc == 1)
        wg_ug.extend([1.0 - float((s_tc[mh] > tau_all5).mean()), float((s_tc[mm] > tau_all5).mean())])

    res["test_confirm"] = {
        "overall_auroc": round(safe_auroc(y_tc, s_tc), 8),
        "worst_group_acc_E0_E3": round(min(wg_tc), 8),
        "worst_edit_bal_acc_E0_E3": round(min(ebal_tc), 8),
        "mean_bal_acc_E0_E3": round(float(np.mean(ebal_tc)), 8),
        "human_fpr_tau_all5": round(float((s_tc[y_tc == 0] > tau_all5).mean()), 8),
        "machine_tpr_tau_all5": round(float((s_tc[y_tc == 1] > tau_all5).mean()), 8),
        "unseen_genres_auroc": round(safe_auroc(y_tc[ug_mask], s_tc[ug_mask]), 8),
        "unseen_genres_wg_E0_E3": round(min(wg_ug), 8),
        "seen_domains_auroc": round(safe_auroc(y_tc[~ug_mask], s_tc[~ug_mask]), 8),
        "by_level": tc_by_lv,
        "by_register": tc_by_reg,
    }

    # 3. Crossed-OOD AR & Crossed-OOD LLaDA Masked Diffusion
    for ood_key in ["test_ood_ar", "test_ood_llada"]:
        rows_o = raw_splits[ood_key]
        s_o = np.asarray(preds_by_split[ood_key], dtype=np.float64)
        y_o = np.array([r["origin_label"] for r in rows_o])
        e_o = np.array([r["edit_level"] for r in rows_o])

        by_lv = {}
        wg_o, ebal_o = [], []
        for lv in ["E0", "E1", "E2"]:
            ml = (e_o == lv)
            mh = ml & (y_o == 0)
            mm = ml & (y_o == 1)
            fpr = float((s_o[mh] > tau_all5).mean())
            tpr = float((s_o[mm] > tau_all5).mean())
            b_acc = 0.5 * ((1.0 - fpr) + tpr)
            by_lv[lv] = {
                "auroc": round(safe_auroc(y_o[ml], s_o[ml]), 8),
                "fpr_at_tau_all5": round(fpr, 8),
                "tpr_at_tau_all5": round(tpr, 8),
                "bal_acc": round(b_acc, 8),
            }
            wg_o.extend([1.0 - fpr, tpr])
            ebal_o.append(b_acc)

        ood_entry = {
            "overall_auroc": round(safe_auroc(y_o, s_o), 8),
            "worst_group_acc": round(min(wg_o), 8),
            "worst_edit_bal_acc": round(min(ebal_o), 8),
            "mean_bal_acc": round(float(np.mean(ebal_o)), 8),
            "human_fpr_tau_all5": round(float((s_o[y_o == 0] > tau_all5).mean()), 8),
            "machine_tpr_tau_all5": round(float((s_o[y_o == 1] > tau_all5).mean()), 8),
            "by_level": by_lv,
        }
        if ood_key == "test_ood_llada":
            y_lm = y_o[llada_lm_idx]
            s_lm = s_o[llada_lm_idx]
            ood_entry["len120_220_auroc"] = round(safe_auroc(y_lm, s_lm), 8)
            ood_entry["len120_220_ipw_auroc"] = round(float(roc_auc_score(y_lm, s_lm, sample_weight=llada_ipw_weights)), 8)
            fpr_lm = float((s_lm[y_lm == 0] > tau_all5).mean())
            tpr_lm = float((s_lm[y_lm == 1] > tau_all5).mean())
            ood_entry["len120_220_bal_acc"] = round(0.5 * ((1.0 - fpr_lm) + tpr_lm), 8)
        res[ood_key] = ood_entry

    # 4. External ESL Student Essays (300 Unedited E0 + 300 Grammar-Polished E1 — ALL Human O=0)
    esl_rows = raw_splits["external_esl_full"]
    s_esl = np.asarray(preds_by_split["external_esl_full"], dtype=np.float64)
    e_esl = np.array([r["edit_level"] for r in esl_rows])
    res["external_esl_fairness"] = {
        "esl_E0_fpr_at_tau_all5": round(float((s_esl[e_esl == "E0"] > tau_all5).mean()), 8),
        "esl_E1_polished_fpr_at_tau_all5": round(float((s_esl[e_esl == "E1"] > tau_all5).mean()), 8),
        "esl_E0_fpr_at_tau_e0_5": round(float((s_esl[e_esl == "E0"] > tau_e0_5).mean()), 8),
        "esl_E1_polished_fpr_at_tau_e0_5": round(float((s_esl[e_esl == "E1"] > tau_e0_5).mean()), 8),
    }

    # 5. External MixSet Test Split (Pure 620 & All 720)
    ms_rows = raw_splits["external_mixset"]
    s_ms = np.asarray(preds_by_split["external_mixset"], dtype=np.float64)
    pure_mask = np.array([not r.get("is_completion", False) for r in ms_rows])
    y_ms = np.array([r["origin_label"] for r in ms_rows])

    hwt_pol_mask = np.array([("polish" in r["subset"]) and (r["origin_label"] == 0) for r in ms_rows])
    mgt_hum_mask = np.array([("humanize" in r["subset"]) and (r["origin_label"] == 1) for r in ms_rows])
    mgt_pol_mask = np.array([("MGT_polish" in r["subset"]) and (r["origin_label"] == 1) for r in ms_rows])
    fpr_ms_pure = float((s_ms[pure_mask & (y_ms == 0)] > tau_all5).mean())
    tpr_ms_pure = float((s_ms[pure_mask & (y_ms == 1)] > tau_all5).mean())

    res["external_mixset"] = {
        "pure_origin_auroc": round(safe_auroc(y_ms[pure_mask], s_ms[pure_mask]), 8),
        "all_720_auroc": round(safe_auroc(y_ms, s_ms), 8),
        "pure_human_fpr_tau_all5": round(fpr_ms_pure, 8),
        "pure_machine_tpr_tau_all5": round(tpr_ms_pure, 8),
        "pure_bal_acc_tau_all5": round(0.5 * ((1.0 - fpr_ms_pure) + tpr_ms_pure), 8),
        "hwt_llm_polished_fpr_tau_all5": round(float((s_ms[hwt_pol_mask] > tau_all5).mean()), 8),
        "mgt_humanized_tpr_tau_all5": round(float((s_ms[mgt_hum_mask] > tau_all5).mean()), 8),
        "mgt_human_polished_tpr_tau_all5": round(float((s_ms[mgt_pol_mask] > tau_all5).mean()), 8),
    }

    # 6. External Toloka Beemo (1,200 rows across 300 prompts)
    bm_rows = raw_splits["external_beemo"]
    s_bm = np.asarray(preds_by_split["external_beemo"], dtype=np.float64)
    y_bm = np.array([r["origin_label"] for r in bm_rows])
    sub_bm = np.array([r["subset"] for r in bm_rows])
    m_hum_bm = (sub_bm == "human_raw")
    m_raw_bm = (sub_bm == "mgt_raw")
    m_exp_bm = (sub_bm == "mgt_expert_human_edited")
    m_gpt_bm = (sub_bm == "mgt_gpt4o_edited")

    res["external_beemo"] = {
        "overall_auroc": round(safe_auroc(y_bm, s_bm), 8),
        "auroc_expert_edited_vs_human": round(safe_auroc(y_bm[m_hum_bm | m_exp_bm], s_bm[m_hum_bm | m_exp_bm]), 8),
        "auroc_gpt4o_edited_vs_human": round(safe_auroc(y_bm[m_hum_bm | m_gpt_bm], s_bm[m_hum_bm | m_gpt_bm]), 8),
        "human_fpr_tau_all5": round(float((s_bm[m_hum_bm] > tau_all5).mean()), 8),
        "mgt_raw_tpr_tau_all5": round(float((s_bm[m_raw_bm] > tau_all5).mean()), 8),
        "mgt_expert_edited_tpr_tau_all5": round(float((s_bm[m_exp_bm] > tau_all5).mean()), 8),
        "mgt_gpt4o_edited_tpr_tau_all5": round(float((s_bm[m_gpt_bm] > tau_all5).mean()), 8),
    }

    # 7. External MAGE Wild Unseen Domains
    mg_rows = raw_splits["external_mage_wild"]
    s_mg = np.asarray(preds_by_split["external_mage_wild"], dtype=np.float64)
    y_mg = np.array([r["origin_label"] for r in mg_rows])
    fpr_mg = float((s_mg[y_mg == 0] > tau_all5).mean())
    tpr_mg = float((s_mg[y_mg == 1] > tau_all5).mean())
    res["external_mage_wild"] = {
        "auroc": round(safe_auroc(y_mg, s_mg), 8),
        "human_fpr_tau_all5": round(fpr_mg, 8),
        "machine_tpr_tau_all5": round(tpr_mg, 8),
        "bal_acc_tau_all5": round(0.5 * ((1.0 - fpr_mg) + tpr_mg), 8),
    }

    # 8. Selective Risk-Coverage Curve across E0..E4 on test_id
    if np.min(s_tid) >= -1e-6 and np.max(s_tid) <= 1.0 + 1e-6 and 0.0 < tau_all5 < 1.0:
        denom_tid = np.where(s_tid > tau_all5, max(1.0 - tau_all5, 1e-6), max(tau_all5, 1e-6))
    else:
        pos_span = max(float(np.max(s_tid) - tau_all5), 1e-6)
        neg_span = max(float(tau_all5 - np.min(s_tid)), 1e-6)
        denom_tid = np.where(s_tid > tau_all5, pos_span, neg_span)
    margin_conf = np.abs(s_tid - tau_all5) / denom_tid
    res["selective_rc_margin_only_E0_E4"] = compute_risk_coverage(y_tid, s_tid, tau_all5, margin_conf, edit_levels=e_tid)

    if edit_intens_by_split and "test_id" in edit_intens_by_split and len(edit_intens_by_split["test_id"]) == len(s_tid):
        a_hat = np.asarray(edit_intens_by_split["test_id"], dtype=np.float64)
        intens_conf = margin_conf * (1.0 - 0.5 * np.clip(a_hat, 0.0, 1.0))
        res["selective_rc_intensity_aware_E0_E4"] = compute_risk_coverage(y_tid, s_tid, tau_all5, intens_conf, edit_levels=e_tid)
    else:
        res["selective_rc_intensity_aware_E0_E4"] = res["selective_rc_margin_only_E0_E4"]

    return res


def evaluate_few_shot_human_anchor_crc(
    payload_list: list,
    raw_splits: dict,
    K_anchors: int = 30,
    alpha: float = 0.05,
    n_draws: int = 200,
):
    """
    Evaluates Few-Shot Human-Only Domain-Anchor Split-CRC (`K=30` verified human families from target platform,
    ZERO machine examples!). Strictly holds out ALL human and machine siblings of the sampled `K` anchor families
    from the evaluation pool on every draw `r = 0..n_draws-1`, verifying marginal expected cluster-averaged human
    FPR <= alpha = 0.05 while reporting recovered TPR / balanced accuracy alongside Zero-Shot locked `tau_cal_all5`.
    """
    benchmarks = [
        ("test_confirm", raw_splits["test_confirm"], False),
        ("external_esl_full", raw_splits["external_esl_full"], False),
        ("external_mixset_pure", [r for r in raw_splits["external_mixset"] if not r.get("is_completion", False)], False),
        ("external_beemo", raw_splits["external_beemo"], False),
        ("external_mage_wild", raw_splits["external_mage_wild"], True),
    ]

    out = {"K_human_families": K_anchors, "alpha": alpha, "n_draws": n_draws, "benchmarks": {}}

    for b_name, rows, allow_rec in benchmarks:
        split_key = "external_mixset" if b_name == "external_mixset_pure" else b_name
        if b_name == "external_mixset_pure":
            pure_idx = np.array([i for i, r in enumerate(raw_splits["external_mixset"]) if not r.get("is_completion", False)], dtype=np.int64)
            seed_scores = [np.asarray(pl["preds"][split_key], dtype=np.float64)[pure_idx] for pl in payload_list]
        else:
            seed_scores = [np.asarray(pl["preds"][split_key], dtype=np.float64) for pl in payload_list]
        seed_taus_zs = [float(pl["locked_thresholds"]["tau_cal_all5"]) for pl in payload_list]

        pids = [resolve_cluster_id(r, allow_record_id=allow_rec) for r in rows]
        y = np.array([int(r["origin_label"]) for r in rows], dtype=np.int64)

        # Eligible calibration clusters = unique family IDs possessing at least one human (O=0) row
        hum_family_ids = sorted({pids[i] for i in range(len(rows)) if y[i] == 0})
        assert len(hum_family_ids) > K_anchors + 10, f"Too few human families ({len(hum_family_ids)}) in {b_name}"

        fam_to_hum_idx = defaultdict(list)
        fam_to_all_idx = defaultdict(list)
        for idx, (pid, yi) in enumerate(zip(pids, y)):
            fam_to_all_idx[pid].append(idx)
            if yi == 0:
                fam_to_hum_idx[pid].append(idx)

        # Zero-shot metrics across seeds
        zs_clust_fpr, zs_pass_fpr, zs_mach_tpr, zs_bal = [], [], [], []
        for sc, t_zs in zip(seed_scores, seed_taus_zs):
            h_m = (y == 0)
            m_m = (y == 1)
            zs_clust_fpr.append(cluster_avg_fpr(sc[h_m], [pids[i] for i in np.where(h_m)[0]], t_zs))
            zs_pass_fpr.append(float((sc[h_m] > t_zs).mean()))
            if m_m.any():
                tpr_v = float((sc[m_m] > t_zs).mean())
                zs_mach_tpr.append(tpr_v)
                zs_bal.append(0.5 * ((1.0 - float((sc[h_m] > t_zs).mean())) + tpr_v))

        # Few-shot K=30 human-only anchor draws (strictly cluster-disjoint evaluation!)
        seed_draw_cfpr, seed_draw_pfpr, seed_draw_tpr, seed_draw_bal, seed_draw_q90 = [], [], [], [], []
        # Specific ESL E1 polished FPR & Beemo expert-edited TPR tracking
        seed_draw_esl_e1_fpr, seed_draw_beemo_exp_tpr = [], []
        e_arr = np.array([r.get("edit_level", "E0") for r in rows])
        sub_arr = np.array([r.get("subset", "") for r in rows])

        for sc in seed_scores:
            d_cfpr, d_pfpr, d_tpr, d_bal = [], [], [], []
            d_esl_e1, d_bm_exp = [], []
            for r_idx in range(n_draws):
                rng = np.random.default_rng(10000 + r_idx)
                chosen_fams = set(rng.choice(hum_family_ids, size=K_anchors, replace=False).tolist())

                cal_h_idx = []
                for cf in chosen_fams:
                    cal_h_idx.extend(fam_to_hum_idx[cf])
                cal_h_idx = np.asarray(cal_h_idx, dtype=np.int64)
                tau_dom = cluster_conformal_upper_threshold(
                    sc[cal_h_idx], [pids[i] for i in cal_h_idx], alpha=alpha
                )

                # Exclude ALL human AND machine siblings belonging to chosen_fams!
                eval_mask = np.array([pids[i] not in chosen_fams for i in range(len(rows))], dtype=bool)
                ev_h = eval_mask & (y == 0)
                ev_m = eval_mask & (y == 1)

                c_fpr = cluster_avg_fpr(sc[ev_h], [pids[i] for i in np.where(ev_h)[0]], tau_dom)
                p_fpr = float((sc[ev_h] > tau_dom).mean())
                d_cfpr.append(c_fpr)
                d_pfpr.append(p_fpr)
                if ev_m.any():
                    tpr_d = float((sc[ev_m] > tau_dom).mean())
                    d_tpr.append(tpr_d)
                    d_bal.append(0.5 * ((1.0 - p_fpr) + tpr_d))
                if b_name == "external_esl_full":
                    ev_e1 = ev_h & (e_arr == "E1")
                    d_esl_e1.append(float((sc[ev_e1] > tau_dom).mean()))
                if b_name == "external_beemo":
                    ev_exp = ev_m & (sub_arr == "mgt_expert_human_edited")
                    d_bm_exp.append(float((sc[ev_exp] > tau_dom).mean()))

            seed_draw_cfpr.append(float(np.mean(d_cfpr)))
            seed_draw_pfpr.append(float(np.mean(d_pfpr)))
            seed_draw_q90.append(float(np.percentile(d_cfpr, 90)))
            if d_tpr:
                seed_draw_tpr.append(float(np.mean(d_tpr)))
                seed_draw_bal.append(float(np.mean(d_bal)))
            if d_esl_e1:
                seed_draw_esl_e1_fpr.append(float(np.mean(d_esl_e1)))
            if d_bm_exp:
                seed_draw_beemo_exp_tpr.append(float(np.mean(d_bm_exp)))

        def _ms(v):
            if not v:
                return None
            return {"mean": round(float(np.mean(v)), 8), "std": round(float(np.std(v, ddof=0)), 8)}

        b_entry = {
            "total_human_families": len(hum_family_ids),
            "zero_shot_locked_val_calib": {
                "cluster_human_fpr": _ms(zs_clust_fpr),
                "passage_human_fpr": _ms(zs_pass_fpr),
                "machine_tpr": _ms(zs_mach_tpr),
                "bal_acc": _ms(zs_bal),
            },
            "few_shot_K30_human_anchor_crc": {
                "marginal_cluster_human_fpr": _ms(seed_draw_cfpr),
                "marginal_passage_human_fpr": _ms(seed_draw_pfpr),
                "p90_cluster_human_fpr_across_draws": _ms(seed_draw_q90),
                "machine_tpr": _ms(seed_draw_tpr),
                "bal_acc": _ms(seed_draw_bal),
                "crc_bound_satisfied": bool(float(np.mean(seed_draw_cfpr)) <= alpha + 1e-6),
            },
        }
        if seed_draw_esl_e1_fpr:
            b_entry["few_shot_K30_human_anchor_crc"]["esl_E1_polished_fpr"] = _ms(seed_draw_esl_e1_fpr)
        if seed_draw_beemo_exp_tpr:
            b_entry["few_shot_K30_human_anchor_crc"]["beemo_expert_edited_tpr"] = _ms(seed_draw_beemo_exp_tpr)
        out["benchmarks"][b_name] = b_entry

    return out


def load_all_splits():
    val_manifest = json.load(open(os.path.join(V2_DIR, "val_manifest.json")))
    sel_set = set(val_manifest["val_select_parent_ids"])
    cal_set = set(val_manifest["val_calib_parent_ids"])
    assert len(sel_set) == 60 and len(cal_set) == 60 and len(sel_set & cal_set) == 0

    raw = {
        "train_fit_v2": json.load(open(os.path.join(V2_DIR, "split_train_fit_v2.json"))),
        "train_op_v2": json.load(open(os.path.join(V2_DIR, "split_train_op_v2.json"))),
        "train_all_v2": json.load(open(os.path.join(V2_DIR, "split_train_v2.json"))),
        "train_v1": json.load(open(os.path.join(V1_DIR, "split_train.json"))),
        "test_confirm": json.load(open(os.path.join(V2_DIR, "split_test_confirm.json"))),
    }
    val_rows = json.load(open(os.path.join(V1_DIR, "split_val.json")))
    raw["val_select"] = [r for r in val_rows if r["parent_id"] in sel_set]
    raw["val_calib"] = [r for r in val_rows if r["parent_id"] in cal_set]

    for sp in ["test_id", "test_ood_ar", "test_ood_llada"]:
        raw[sp] = json.load(open(os.path.join(V1_DIR, f"split_{sp}.json")))
    for ext in ["external_esl_full", "external_mixset", "external_beemo", "external_mage_wild"]:
        raw[ext] = json.load(open(os.path.join(V1_DIR, f"{ext}.json")))

    v1_p_dom = {r["parent_id"]: r["domain"] for r in raw["train_v1"]}
    v1_nr = sorted([p for p, d in v1_p_dom.items() if d == "instruct_prose"], key=lambda p: det_rank_key("bridge_v1_op", p))
    v1_mg = sorted([p for p, d in v1_p_dom.items() if d == "public_expository"], key=lambda p: det_rank_key("bridge_v1_op", p))
    v1_fit_set = set(v1_nr[:188] + v1_mg[:187])
    v1_op_set = set(v1_nr[188:] + v1_mg[187:])
    raw["train_v1_fit"] = [r for r in raw["train_v1"] if r["parent_id"] in v1_fit_set]
    raw["train_v1_op"] = [r for r in raw["train_v1"] if r["parent_id"] in v1_op_set]
    return raw, val_manifest


def main():
    raw_splits, val_manifest = load_all_splits()
    expected_fp = compute_full_split_fingerprint(raw_splits, val_manifest)

    eval_split_keys = [
        "val_calib", "test_id", "test_ood_ar", "test_ood_llada", "test_confirm",
        "external_esl_full", "external_mixset", "external_beemo", "external_mage_wild",
    ]
    expected_row_ids = {k: [r["id"] for r in raw_splits[k]] for k in eval_split_keys + ["val_select"]}

    # Precompute LLaDA [120, 220] common-support indices & 4-bin IPW weights
    ll_rows = raw_splits["test_ood_llada"]
    ll_lm_idx = np.array([i for i, r in enumerate(ll_rows) if 120 <= r.get("wc", len(r["text"].split())) <= 220], dtype=np.int64)
    ll_lm_items = [ll_rows[i] for i in ll_lm_idx]
    y_ll_lm = np.array([r["origin_label"] for r in ll_lm_items])
    wc_ll_lm = np.array([r.get("wc", len(r["text"].split())) for r in ll_lm_items], dtype=np.float64)
    bins_ll = [120, 145, 170, 195, 221]
    bin_ids_ll = np.digitize(wc_ll_lm, bins_ll) - 1
    ipw_w_ll = np.ones(len(ll_lm_items), dtype=np.float64)
    for b in range(4):
        mb0 = (bin_ids_ll == b) & (y_ll_lm == 0)
        mb1 = (bin_ids_ll == b) & (y_ll_lm == 1)
        if mb0.any() and mb1.any():
            ipw_w_ll[mb0] = 0.25 / float(mb0.sum())
            ipw_w_ll[mb1] = 0.25 / float(mb1.sum())

    summary = {
        "code_version": "cedar_v3_spotlight",
        "calibration_protocol": "parent-cluster split-conformal risk control (`score > tau`, alpha=0.05) on source-stratified 60-parent val_calib (30 MAGE + 30 No Robots), plus few-shot K=30 human-only domain-anchor split-CRC with strictly cluster-disjoint holdout",
        "bootstrap_protocol": "5000-resample paired cluster bootstrap by parent_id/prompt_id (record id only for independent MAGE Wild), conditional on saved 5-seed core runs (42, 1042, 2042, 3042, 4042) or 3-seed ablations (42, 1042, 2042); two-sided sign-tail p_boot (+1 smoothing, zero_tol=1e-12) plus Holm-Bonferroni adjustment across primary confirmatory family",
        "data_fingerprint": expected_fp,
        "llada_common_support_120_220_counts": {
            "n_passages": int(len(ll_lm_items)),
            "n_parents": int(len({r["parent_id"] for r in ll_lm_items})),
            "n_human": int((y_ll_lm == 0).sum()),
            "n_machine": int((y_ll_lm == 1).sum()),
        },
        "pretrained_and_probes": {},
        "trained_per_seed": {},
        "trained_multi_seed_agg": {},
        "nsd_subspace_diagnostics": {},
        "checkpoint_registry": [],
        "attribution_bridge_2x2": {},
        "few_shot_human_anchor_crc": {},
        "paired_significance_tests": {},
        "holm_bonferroni_primary_family": {},
    }

    # 1. Evaluate Pretrained Detectors & Shortcut Probes (calibrated on stratified val_calib!)
    pre_preds_normalized = {}
    pre_path = os.path.join(RES_DIR, "preds_pretrained_and_probes_v3.json")
    assert os.path.exists(pre_path), f"Missing {pre_path}!"
    val_full_rows = json.load(open(os.path.join(V1_DIR, "split_val.json")))
    cal_set = set(val_manifest["val_calib_parent_ids"])
    val_calib_idx_in_val = np.array([i for i, r in enumerate(val_full_rows) if r["parent_id"] in cal_set], dtype=np.int64)

    raw_pre = json.load(open(pre_path))
    for m_name, p_dict in raw_pre.items():
        norm_p = dict(p_dict)
        if "val_calib" not in norm_p:
            norm_p["val_calib"] = np.asarray(p_dict["val"], dtype=np.float64)[val_calib_idx_in_val].tolist()
        pre_preds_normalized[m_name] = norm_p
        summary["pretrained_and_probes"][m_name] = evaluate_detector_dict(
            m_name, norm_p, raw_splits, ll_lm_idx, ipw_w_ll
        )

    # 2. Load & Verify All 116 Trained Model Runs
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.json")))
    assert len(pred_files) >= 116, f"Expected >=116 trained prediction files in {PRED_DIR}, found {len(pred_files)}"

    loaded_payloads = []
    for fp in pred_files:
        payload = json.load(open(fp))
        assert payload.get("data_fingerprint") == expected_fp, (
            f"STALE PREDICTION DETECTED in {fp}: fp={payload.get('data_fingerprint')} != {expected_fp}"
        )
        for sp_k, rids in payload.get("row_ids", {}).items():
            assert rids == expected_row_ids[sp_k], f"Row-ID misalignment in {fp} split {sp_k}!"
        loaded_payloads.append(payload)

    loaded_payloads.sort(key=lambda pl: (pl["run_name"], int(pl["seed"])))

    by_variant = defaultdict(list)
    by_variant_payloads = defaultdict(list)
    by_variant_nsd = defaultdict(list)

    for payload in loaded_payloads:
        r_name = payload["run_name"]
        sd = int(payload["seed"])
        key = f"{r_name}_seed{sd}"
        ev = evaluate_detector_dict(
            key,
            payload["preds"],
            raw_splits,
            ll_lm_idx,
            ipw_w_ll,
            locked_thresholds=payload.get("locked_thresholds"),
            edit_intens_by_split=payload.get("edit_intensity"),
        )
        ev["seed"] = sd
        ev["stage_timings"] = payload["stage_timings"]
        ev["trainable_params"] = payload.get("trainable_params")
        ev["best_val_select"] = payload.get("best_val_select")
        summary["trained_per_seed"][key] = ev
        by_variant[r_name].append(ev)
        by_variant_payloads[r_name].append(payload)
        if payload.get("nsd_diagnostics", {}).get("active"):
            by_variant_nsd[r_name].append(payload["nsd_diagnostics"])
        if payload.get("checkpoint_archive"):
            summary["checkpoint_registry"].append({
                "run_name": r_name,
                "seed": sd,
                **payload["checkpoint_archive"],
            })

    def ms(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        if not vals:
            return {"mean": None, "std": None}
        return {"mean": round(float(np.mean(vals)), 8), "std": round(float(np.std(vals, ddof=0)), 8)}

    # 3. Aggregate Multi-Seed Statistics
    for r_name, ev_list in by_variant.items():
        recov_agg = {}
        for lv in ["E0", "E1", "E2", "E3", "E4"]:
            recov_agg[lv] = {
                "auroc": ms([x["test_id_recoverability"][lv]["auroc"] for x in ev_list]),
                "bal_acc": ms([x["test_id_recoverability"][lv]["bal_acc"] for x in ev_list]),
                "fpr_allcal": ms([x["test_id_recoverability"][lv]["fpr_at_tau_all5"] for x in ev_list]),
                "tpr_allcal": ms([x["test_id_recoverability"][lv]["tpr_at_tau_all5"] for x in ev_list]),
                "fpr_e0cal": ms([x["test_id_recoverability"][lv]["fpr_at_tau_e0_5"] for x in ev_list]),
            }

        tc_lv_agg = {}
        for lv in ["E0", "E1", "E2", "E3"]:
            tc_lv_agg[lv] = {
                "auroc": ms([x["test_confirm"]["by_level"][lv]["auroc"] for x in ev_list]),
                "bal_acc": ms([x["test_confirm"]["by_level"][lv]["bal_acc"] for x in ev_list]),
                "fpr_allcal": ms([x["test_confirm"]["by_level"][lv]["fpr_at_tau_all5"] for x in ev_list]),
                "tpr_allcal": ms([x["test_confirm"]["by_level"][lv]["tpr_at_tau_all5"] for x in ev_list]),
            }

        tc_reg_agg = {}
        for reg_name in ["unedited_E0", "formal_edited", "colloquial_humanized"]:
            tc_reg_agg[reg_name] = {
                "auroc": ms([x["test_confirm"]["by_register"][reg_name]["auroc"] for x in ev_list]),
                "human_fpr": ms([x["test_confirm"]["by_register"][reg_name]["human_fpr_tau_all5"] for x in ev_list]),
                "machine_tpr": ms([x["test_confirm"]["by_register"][reg_name]["machine_tpr_tau_all5"] for x in ev_list]),
                "bal_acc": ms([x["test_confirm"]["by_register"][reg_name]["bal_acc_tau_all5"] for x in ev_list]),
            }

        summary["trained_multi_seed_agg"][r_name] = {
            "n_seeds": len(ev_list),
            "seeds": [x["seed"] for x in ev_list],
            "trainable_params": ev_list[0].get("trainable_params"),
            "fit_and_select_seconds": ms([x["stage_timings"]["fit_and_select_seconds"] for x in ev_list]),
            "calib_seconds": ms([x["stage_timings"]["calib_seconds"] for x in ev_list]),
            "eval_inference_seconds": ms([x["stage_timings"]["eval_inference_seconds"] for x in ev_list]),
            "total_seconds": ms([x["stage_timings"]["total_seconds"] for x in ev_list]),
            # Test-ID
            "id_auroc_E0_E3": ms([x["test_id_auroc_E0_E3"] for x in ev_list]),
            "id_wg_E0_E3": ms([x["test_id_worst_group_acc_E0_E3"] for x in ev_list]),
            "id_wg_E0_E4": ms([x["test_id_worst_group_acc_E0_E4"] for x in ev_list]),
            "id_worst_ebal_E0_E3": ms([x["test_id_worst_edit_bal_acc_E0_E3"] for x in ev_list]),
            "id_mean_bal_E0_E3": ms([x["test_id_mean_bal_acc_E0_E3"] for x in ev_list]),
            "id_ece_E0_E3": ms([x["test_id_ece_E0_E3"] for x in ev_list]),
            "id_recoverability": recov_agg,
            # Pristine Confirmation Benchmark (`test_confirm`)
            "confirm_auroc": ms([x["test_confirm"]["overall_auroc"] for x in ev_list]),
            "confirm_wg_E0_E3": ms([x["test_confirm"]["worst_group_acc_E0_E3"] for x in ev_list]),
            "confirm_mean_bal_E0_E3": ms([x["test_confirm"]["mean_bal_acc_E0_E3"] for x in ev_list]),
            "confirm_hum_fpr": ms([x["test_confirm"]["human_fpr_tau_all5"] for x in ev_list]),
            "confirm_mach_tpr": ms([x["test_confirm"]["machine_tpr_tau_all5"] for x in ev_list]),
            "confirm_unseen_genres_auroc": ms([x["test_confirm"]["unseen_genres_auroc"] for x in ev_list]),
            "confirm_unseen_genres_wg": ms([x["test_confirm"]["unseen_genres_wg_E0_E3"] for x in ev_list]),
            "confirm_seen_domains_auroc": ms([x["test_confirm"]["seen_domains_auroc"] for x in ev_list]),
            "confirm_by_level": tc_lv_agg,
            "confirm_by_register": tc_reg_agg,
            # Crossed OOD AR & LLaDA
            "ood_ar_auroc": ms([x["test_ood_ar"]["overall_auroc"] for x in ev_list]),
            "ood_ar_wg_acc": ms([x["test_ood_ar"]["worst_group_acc"] for x in ev_list]),
            "ood_ar_mean_bal": ms([x["test_ood_ar"]["mean_bal_acc"] for x in ev_list]),
            "ood_ar_E2_auroc": ms([x["test_ood_ar"]["by_level"]["E2"]["auroc"] for x in ev_list]),
            "ood_llada_auroc": ms([x["test_ood_llada"]["overall_auroc"] for x in ev_list]),
            "ood_llada_len120_220_auroc": ms([x["test_ood_llada"]["len120_220_auroc"] for x in ev_list]),
            "ood_llada_len120_220_ipw_auroc": ms([x["test_ood_llada"]["len120_220_ipw_auroc"] for x in ev_list]),
            "ood_llada_wg_acc": ms([x["test_ood_llada"]["worst_group_acc"] for x in ev_list]),
            "ood_llada_mean_bal": ms([x["test_ood_llada"]["mean_bal_acc"] for x in ev_list]),
            "ood_llada_E2_auroc": ms([x["test_ood_llada"]["by_level"]["E2"]["auroc"] for x in ev_list]),
            # External ESL, MixSet, Beemo, MAGE Wild
            "esl_E0_fpr": ms([x["external_esl_fairness"]["esl_E0_fpr_at_tau_all5"] for x in ev_list]),
            "esl_E1_fpr": ms([x["external_esl_fairness"]["esl_E1_polished_fpr_at_tau_all5"] for x in ev_list]),
            "esl_E1_fpr_e0cal": ms([x["external_esl_fairness"]["esl_E1_polished_fpr_at_tau_e0_5"] for x in ev_list]),
            "mixset_pure_auroc": ms([x["external_mixset"]["pure_origin_auroc"] for x in ev_list]),
            "mixset_all_auroc": ms([x["external_mixset"]["all_720_auroc"] for x in ev_list]),
            "mixset_pure_bal_acc": ms([x["external_mixset"]["pure_bal_acc_tau_all5"] for x in ev_list]),
            "mixset_hwt_pol_fpr": ms([x["external_mixset"]["hwt_llm_polished_fpr_tau_all5"] for x in ev_list]),
            "mixset_mgt_hum_tpr": ms([x["external_mixset"]["mgt_humanized_tpr_tau_all5"] for x in ev_list]),
            "mixset_mgt_pol_tpr": ms([x["external_mixset"]["mgt_human_polished_tpr_tau_all5"] for x in ev_list]),
            "beemo_auroc": ms([x["external_beemo"]["overall_auroc"] for x in ev_list]),
            "beemo_exp_auroc": ms([x["external_beemo"]["auroc_expert_edited_vs_human"] for x in ev_list]),
            "beemo_gpt4o_auroc": ms([x["external_beemo"]["auroc_gpt4o_edited_vs_human"] for x in ev_list]),
            "beemo_hum_fpr": ms([x["external_beemo"]["human_fpr_tau_all5"] for x in ev_list]),
            "beemo_raw_tpr": ms([x["external_beemo"]["mgt_raw_tpr_tau_all5"] for x in ev_list]),
            "beemo_exp_tpr": ms([x["external_beemo"]["mgt_expert_edited_tpr_tau_all5"] for x in ev_list]),
            "beemo_gpt4o_tpr": ms([x["external_beemo"]["mgt_gpt4o_edited_tpr_tau_all5"] for x in ev_list]),
            "mage_wild_auroc": ms([x["external_mage_wild"]["auroc"] for x in ev_list]),
            "mage_wild_hum_fpr": ms([x["external_mage_wild"]["human_fpr_tau_all5"] for x in ev_list]),
            "mage_wild_mach_tpr": ms([x["external_mage_wild"]["machine_tpr_tau_all5"] for x in ev_list]),
            "mage_wild_bal_acc": ms([x["external_mage_wild"]["bal_acc_tau_all5"] for x in ev_list]),
            # Selective Risk-Coverage
            "rc_margin_aurc": ms([x["selective_rc_margin_only_E0_E4"]["aurc_20_100"] for x in ev_list]),
            "rc_intens_aurc": ms([x["selective_rc_intensity_aware_E0_E4"]["aurc_20_100"] for x in ev_list]),
            "rc_margin_cov80_err": ms([x["selective_rc_margin_only_E0_E4"]["cov_80"] for x in ev_list]),
            "rc_intens_cov80_err": ms([x["selective_rc_intensity_aware_E0_E4"]["cov_80"] for x in ev_list]),
            "rc_margin_curve": {f"cov_{c}": ms([x["selective_rc_margin_only_E0_E4"][f"cov_{c}"] for x in ev_list]) for c in [50, 60, 70, 80, 90, 100]},
            "rc_intens_curve": {f"cov_{c}": ms([x["selective_rc_intensity_aware_E0_E4"][f"cov_{c}"] for x in ev_list]) for c in [50, 60, 70, 80, 90, 100]},
            "rc_margin_e4_share": {f"cov_{c}": ms([x["selective_rc_margin_only_E0_E4"][f"e4_share_cov_{c}"] for x in ev_list]) for c in [50, 60, 70, 80, 90, 100]},
            "rc_intens_e4_share": {f"cov_{c}": ms([x["selective_rc_intensity_aware_E0_E4"][f"e4_share_cov_{c}"] for x in ev_list]) for c in [50, 60, 70, 80, 90, 100]},
        }

    for r_name, diag_list in by_variant_nsd.items():
        summary["nsd_subspace_diagnostics"][r_name] = {
            "frac_edit_energy_removed": ms([d.get("frac_edit_energy_removed") for d in diag_list]),
            "frac_orig_energy_removed": ms([d.get("frac_orig_energy_removed") for d in diag_list]),
            "eigengap_r": ms([d.get("eigengap_r") for d in diag_list]),
            "mu_r": ms([d.get("mu_r") for d in diag_list]),
            "min_residual_norm": ms([d.get("min_residual_norm") for d in diag_list]),
            "p05_residual_norm": ms([d.get("p05_residual_norm") for d in diag_list]),
            "mean_residual_norm": ms([d.get("mean_residual_norm") for d in diag_list]),
        }

    # 4. 2x2 Old-vs-New Data x Old-vs-New Recipe Attribution Bridge Summary
    bridge_cells = [
        "bridge_v1data_aug_erm",
        "bridge_v1data_v1cedar",
        "bridge_v1data_v2cedar",
        "mb_v3_augmented_erm",
        "mb_v3_strong_combined_no_nsd",
        "mb_v3_cedar_split_op",
        "mb_v3_cedar_full800",
    ]
    for bc in bridge_cells:
        if bc in summary["trained_multi_seed_agg"]:
            agg = summary["trained_multi_seed_agg"][bc]
            summary["attribution_bridge_2x2"][bc] = {
                "n_seeds": agg["n_seeds"],
                "confirm_wg_E0_E3": agg["confirm_wg_E0_E3"],
                "confirm_auroc": agg["confirm_auroc"],
                "confirm_colloquial_auroc": agg["confirm_by_register"]["colloquial_humanized"]["auroc"],
                "id_wg_E0_E3": agg["id_wg_E0_E3"],
                "ood_ar_auroc": agg["ood_ar_auroc"],
                "ood_llada_len120_220_auroc": agg["ood_llada_len120_220_auroc"],
                "mixset_pure_auroc": agg["mixset_pure_auroc"],
                "mixset_mgt_hum_tpr": agg["mixset_mgt_hum_tpr"],
                "beemo_auroc": agg["beemo_auroc"],
                "esl_E1_fpr": agg["esl_E1_fpr"],
            }

    # 5. Few-Shot K=30 Human-Only Domain-Anchor Split-CRC Evaluation
    for crc_model in ["mb_v3_cedar_split_op", "mb_v3_augmented_erm", "mb_v3_strong_combined_no_nsd", "db_v3_cedar_split_op",
                      "db_v3_strong_combined_no_nsd", "mb_v3_ablation_no_dte", "mb_v3_unedited_erm"]:
        if crc_model in by_variant_payloads:
            summary["few_shot_human_anchor_crc"][crc_model] = evaluate_few_shot_human_anchor_crc(
                by_variant_payloads[crc_model], raw_splits, K_anchors=30, alpha=0.05, n_draws=200
            )
    if "desklib_raid_v1" in pre_preds_normalized:
        cal_rows = raw_splits["val_calib"]
        cal_y = np.array([r["origin_label"] for r in cal_rows])
        cal_pids = [r["parent_id"] for r in cal_rows]
        hum_idx = np.where(cal_y == 0)[0]
        p_cal = np.asarray(pre_preds_normalized["desklib_raid_v1"]["val_calib"], dtype=np.float64)
        tau_dl = cluster_conformal_upper_threshold(p_cal[hum_idx], [cal_pids[i] for i in hum_idx], alpha=0.05)
        summary["few_shot_human_anchor_crc"]["desklib_raid_v1"] = evaluate_few_shot_human_anchor_crc(
            [{"preds": pre_preds_normalized["desklib_raid_v1"], "locked_thresholds": {"tau_cal_all5": tau_dl}}],
            raw_splits,
            K_anchors=30,
            alpha=0.05,
            n_draws=200,
        )

    # 6. High-Power 5,000-Resample Paired Parent-Cluster Bootstrap Tests
    y_ar = np.array([r["origin_label"] for r in raw_splits["test_ood_ar"]])
    y_ll = np.array([r["origin_label"] for r in raw_splits["test_ood_llada"]])
    y_bm = np.array([r["origin_label"] for r in raw_splits["external_beemo"]])
    ms_all_items = raw_splits["external_mixset"]
    y_ms_all = np.array([r["origin_label"] for r in ms_all_items])
    ms_pure_items = [r for r in ms_all_items if not r.get("is_completion", False)]
    ms_pure_idx = np.array([i for i, r in enumerate(ms_all_items) if not r.get("is_completion", False)], dtype=np.int64)
    y_ms_pure = np.array([r["origin_label"] for r in ms_pure_items])
    y_mg = np.array([r["origin_label"] for r in raw_splits["external_mage_wild"]])
    y_tid = np.array([r["origin_label"] for r in raw_splits["test_id"]])
    y_tc = np.array([r["origin_label"] for r in raw_splits["test_confirm"]])

    lv_map = {"E0": 0, "E1": 2, "E2": 4, "E3": 6, "E4": 8}
    g_tid = np.array([lv_map[r["edit_level"]] + int(r["origin_label"]) for r in raw_splits["test_id"]], dtype=np.int64)
    g_tc = np.array([lv_map[r["edit_level"]] + int(r["origin_label"]) for r in raw_splits["test_confirm"]], dtype=np.int64)

    def make_seed_avg_fns(payload_list):
        tid_corrs = [
            ((np.asarray(pl["preds"]["test_id"], dtype=np.float64) > float(pl["locked_thresholds"]["tau_cal_all5"])).astype(np.int64) == y_tid).astype(np.float64)
            for pl in payload_list
        ]
        tc_corrs = [
            ((np.asarray(pl["preds"]["test_confirm"], dtype=np.float64) > float(pl["locked_thresholds"]["tau_cal_all5"])).astype(np.int64) == y_tc).astype(np.float64)
            for pl in payload_list
        ]
        s_tc = [np.asarray(pl["preds"]["test_confirm"], dtype=np.float64) for pl in payload_list]
        s_ar = [np.asarray(pl["preds"]["test_ood_ar"], dtype=np.float64) for pl in payload_list]
        s_ll = [np.asarray(pl["preds"]["test_ood_llada"], dtype=np.float64) for pl in payload_list]
        s_ll_lm = [np.asarray(pl["preds"]["test_ood_llada"], dtype=np.float64)[ll_lm_idx] for pl in payload_list]
        s_bm = [np.asarray(pl["preds"]["external_beemo"], dtype=np.float64) for pl in payload_list]
        s_ms = [np.asarray(pl["preds"]["external_mixset"], dtype=np.float64)[ms_pure_idx] for pl in payload_list]
        s_ms_all = [np.asarray(pl["preds"]["external_mixset"], dtype=np.float64) for pl in payload_list]
        s_mg = [np.asarray(pl["preds"]["external_mage_wild"], dtype=np.float64) for pl in payload_list]

        def tid_wg(b_idx):
            gb = g_tid[b_idx]
            cnt = np.bincount(gb, minlength=10)[:8]
            if np.any(cnt == 0):
                return float("nan")
            return float(np.mean([float(np.min(np.bincount(gb, weights=cr[b_idx], minlength=10)[:8] / cnt)) for cr in tid_corrs]))

        def tc_wg(b_idx):
            gb = g_tc[b_idx]
            cnt = np.bincount(gb, minlength=8)[:8]
            if np.any(cnt == 0):
                return float("nan")
            return float(np.mean([float(np.min(np.bincount(gb, weights=cr[b_idx], minlength=8)[:8] / cnt)) for cr in tc_corrs]))

        return {
            "tid_wg": tid_wg,
            "tc_wg": tc_wg,
            "tc_auc": lambda b_idx: float(np.mean([safe_auroc(y_tc[b_idx], sv[b_idx]) for sv in s_tc])),
            "ar_auc": lambda b_idx: float(np.mean([safe_auroc(y_ar[b_idx], sv[b_idx]) for sv in s_ar])),
            "ll_auc": lambda b_idx: float(np.mean([safe_auroc(y_ll[b_idx], sv[b_idx]) for sv in s_ll])),
            "ll_lm_auc": lambda b_idx: float(np.mean([safe_auroc(y_ll_lm[b_idx], sv[b_idx]) for sv in s_ll_lm])),
            "bm_auc": lambda b_idx: float(np.mean([safe_auroc(y_bm[b_idx], sv[b_idx]) for sv in s_bm])),
            "ms_auc": lambda b_idx: float(np.mean([safe_auroc(y_ms_pure[b_idx], sv[b_idx]) for sv in s_ms])),
            "ms_all_auc": lambda b_idx: float(np.mean([safe_auroc(y_ms_all[b_idx], sv[b_idx]) for sv in s_ms_all])),
            "mg_auc": lambda b_idx: float(np.mean([safe_auroc(y_mg[b_idx], sv[b_idx]) for sv in s_mg])),
        }

    ANCHOR_MB = os.environ.get("CEDAR_ANCHOR_MB", "mb_v3_strong_combined_no_nsd")
    ANCHOR_DB = os.environ.get("CEDAR_ANCHOR_DB", "db_v3_strong_combined_no_nsd")
    summary["anchors"] = {"modernbert": ANCHOR_MB, "deberta": ANCHOR_DB}
    cedar_fns = make_seed_avg_fns(by_variant_payloads[ANCHOR_MB])
    competitor_fns = {}
    for comp_name, pl_list in by_variant_payloads.items():
        if comp_name != ANCHOR_MB and not comp_name.startswith("db_"):
            competitor_fns[comp_name] = make_seed_avg_fns(pl_list)

    cal_rows = raw_splits["val_calib"]
    cal_y = np.array([r["origin_label"] for r in cal_rows])
    cal_pids = [r["parent_id"] for r in cal_rows]
    hum_cal_idx = np.where(cal_y == 0)[0]
    for pre_name in ["desklib_raid_v1", "binoculars_qwen3b", "fast_detectgpt_qwen3b", "radar_vicuna7b", "hc3_roberta_detector"]:
        if pre_name in pre_preds_normalized:
            p_cal = np.asarray(pre_preds_normalized[pre_name]["val_calib"], dtype=np.float64)
            p_tau = cluster_conformal_upper_threshold(p_cal[hum_cal_idx], [cal_pids[i] for i in hum_cal_idx], alpha=0.05)
            competitor_fns[pre_name] = make_seed_avg_fns([{"preds": pre_preds_normalized[pre_name], "locked_thresholds": {"tau_cal_all5": p_tau}}])

    for c_name, c_fns in competitor_fns.items():
        summary["paired_significance_tests"][f"{ANCHOR_MB}_vs_{c_name}"] = {
            "test_confirm_worst_group_E0_E3": _defer_boot(
                raw_splits["test_confirm"], cedar_fns["tc_wg"], c_fns["tc_wg"], n_boot=5000
            ),
            "test_confirm_auroc": _defer_boot(
                raw_splits["test_confirm"], cedar_fns["tc_auc"], c_fns["tc_auc"], n_boot=5000
            ),
            "test_id_worst_group_E0_E3": _defer_boot(
                raw_splits["test_id"], cedar_fns["tid_wg"], c_fns["tid_wg"], n_boot=5000
            ),
            "test_ood_ar_auroc": _defer_boot(
                raw_splits["test_ood_ar"], cedar_fns["ar_auc"], c_fns["ar_auc"], n_boot=5000
            ),
            "test_ood_llada_auroc": _defer_boot(
                raw_splits["test_ood_llada"], cedar_fns["ll_auc"], c_fns["ll_auc"], n_boot=5000
            ),
            "test_ood_llada_len120_220_auroc": _defer_boot(
                ll_lm_items, cedar_fns["ll_lm_auc"], c_fns["ll_lm_auc"], n_boot=5000
            ),
            "external_beemo_auroc": _defer_boot(
                raw_splits["external_beemo"], cedar_fns["bm_auc"], c_fns["bm_auc"], n_boot=5000
            ),
            "external_mixset_pure_auroc": _defer_boot(
                ms_pure_items, cedar_fns["ms_auc"], c_fns["ms_auc"], n_boot=5000
            ),
            "external_mage_wild_auroc": _defer_boot(
                raw_splits["external_mage_wild"], cedar_fns["mg_auc"], c_fns["mg_auc"], n_boot=5000, allow_record_id=True
            ),
        }

    # DeBERTa-v3-large paired tests
    db_cedar_fns = make_seed_avg_fns(by_variant_payloads[ANCHOR_DB])
    for db_comp in sorted(k for k in by_variant_payloads if k.startswith("db_") and k != ANCHOR_DB):
        dc_fns = make_seed_avg_fns(by_variant_payloads[db_comp])
        summary["paired_significance_tests"][f"{ANCHOR_DB}_vs_{db_comp}"] = {
            "test_confirm_worst_group_E0_E3": _defer_boot(
                raw_splits["test_confirm"], db_cedar_fns["tc_wg"], dc_fns["tc_wg"], n_boot=5000
            ),
            "test_confirm_auroc": _defer_boot(
                raw_splits["test_confirm"], db_cedar_fns["tc_auc"], dc_fns["tc_auc"], n_boot=5000
            ),
            "test_id_worst_group_E0_E3": _defer_boot(
                raw_splits["test_id"], db_cedar_fns["tid_wg"], dc_fns["tid_wg"], n_boot=5000
            ),
            "test_ood_llada_len120_220_auroc": _defer_boot(
                ll_lm_items, db_cedar_fns["ll_lm_auc"], dc_fns["ll_lm_auc"], n_boot=5000
            ),
            "external_mixset_pure_auroc": _defer_boot(
                ms_pure_items, db_cedar_fns["ms_auc"], dc_fns["ms_auc"], n_boot=5000
            ),
            "external_beemo_auroc": _defer_boot(
                raw_splits["external_beemo"], db_cedar_fns["bm_auc"], dc_fns["bm_auc"], n_boot=5000
            ),
        }

    # Bridge isolated data effect (`mb_v3_augmented_erm` vs `bridge_v1data_aug_erm`)
    aug_v2_fns = make_seed_avg_fns(by_variant_payloads["mb_v3_augmented_erm"])
    aug_v1_fns = make_seed_avg_fns(by_variant_payloads["bridge_v1data_aug_erm"])
    summary["paired_significance_tests"]["bridge_data_effect_aug_erm_v2_vs_v1"] = {
        "test_confirm_worst_group_E0_E3": _defer_boot(
            raw_splits["test_confirm"], aug_v2_fns["tc_wg"], aug_v1_fns["tc_wg"], n_boot=5000
        ),
        "external_mixset_pure_auroc": _defer_boot(
            ms_pure_items, aug_v2_fns["ms_auc"], aug_v1_fns["ms_auc"], n_boot=5000
        ),
    }

    # Secondary family: isolated effect of signal-protected deflation (NSD) at matched trunk data, and of DTE.
    nsd_pairs = [
        ("mb_v3_cedar_split_op", "mb_v3_strong_combined_fit600_no_nsd"),   # NSD effect, with DTE, N_fit=600
        ("mb_v3_ablation_no_dte", "mb_v3_nodte_fit600_no_nsd"),             # NSD effect, no DTE, N_fit=600
        ("db_v3_cedar_no_dte", "db_v3_nodte_no_nsd"),                       # NSD effect, no DTE, DeBERTa (800 vs 600+200)
        ("mb_v3_ablation_unprotected_nsd", "mb_v3_cedar_split_op"),         # signal protection (lambda=0 vs 0.5)
        ("mb_v3_strong_combined_no_nsd", "mb_v3_nodte_no_nsd"),             # DTE effect, no NSD, 800
        ("mb_v3_strong_combined_fit600_no_nsd", "mb_v3_nodte_fit600_no_nsd"),  # DTE effect, no NSD, 600
    ]
    summary["nsd_and_dte_effect_tests"] = {}
    for a_name, b_name in nsd_pairs:
        if a_name in by_variant_payloads and b_name in by_variant_payloads:
            fa, fb = make_seed_avg_fns(by_variant_payloads[a_name]), make_seed_avg_fns(by_variant_payloads[b_name])
            summary["nsd_and_dte_effect_tests"][f"{a_name}_vs_{b_name}"] = {
                "test_confirm_worst_group_E0_E3": _defer_boot(raw_splits["test_confirm"], fa["tc_wg"], fb["tc_wg"], n_boot=5000),
                "test_id_worst_group_E0_E3": _defer_boot(raw_splits["test_id"], fa["tid_wg"], fb["tid_wg"], n_boot=5000),
                "test_confirm_auroc": _defer_boot(raw_splits["test_confirm"], fa["tc_auc"], fb["tc_auc"], n_boot=5000),
                "test_ood_llada_len120_220_auroc": _defer_boot(ll_lm_items, fa["ll_lm_auc"], fb["ll_lm_auc"], n_boot=5000),
                "test_ood_ar_auroc": _defer_boot(raw_splits["test_ood_ar"], fa["ar_auc"], fb["ar_auc"], n_boot=5000),
                "external_mixset_pure_auroc": _defer_boot(ms_pure_items, fa["ms_auc"], fb["ms_auc"], n_boot=5000),
            }

    _resolve_deferred_boots(summary["paired_significance_tests"])
    _resolve_deferred_boots(summary["nsd_and_dte_effect_tests"])

    # Holm-Bonferroni across the primary family: recommended recipe vs standard/external baselines.
    # Sign-aware: each entry keeps the paired difference so a significant result in the wrong direction is visible.
    pst = summary["paired_significance_tests"]
    primary = {
        "confirm_wg_vs_unedited_erm": (f"{ANCHOR_MB}_vs_mb_v3_unedited_erm", "test_confirm_worst_group_E0_E3"),
        "confirm_wg_vs_aug_erm": (f"{ANCHOR_MB}_vs_mb_v3_augmented_erm", "test_confirm_worst_group_E0_E3"),
        "confirm_wg_vs_desklib": (f"{ANCHOR_MB}_vs_desklib_raid_v1", "test_confirm_worst_group_E0_E3"),
        "confirm_wg_vs_binoculars": (f"{ANCHOR_MB}_vs_binoculars_qwen3b", "test_confirm_worst_group_E0_E3"),
        "confirm_wg_vs_fast_detectgpt": (f"{ANCHOR_MB}_vs_fast_detectgpt_qwen3b", "test_confirm_worst_group_E0_E3"),
        "id_wg_vs_aug_erm": (f"{ANCHOR_MB}_vs_mb_v3_augmented_erm", "test_id_worst_group_E0_E3"),
        "mixset_pure_vs_aug_erm": (f"{ANCHOR_MB}_vs_mb_v3_augmented_erm", "external_mixset_pure_auroc"),
        "deberta_confirm_wg_vs_aug_erm": (f"{ANCHOR_DB}_vs_db_v3_augmented_erm", "test_confirm_worst_group_E0_E3"),
    }
    primary_p_map = {k: pst[a][m]["p_boot"] for k, (a, m) in primary.items() if a in pst}
    summary["holm_bonferroni_primary_family"] = {
        "raw_p_boot": primary_p_map,
        "holm_adjusted_p": holm_bonferroni(primary_p_map),
        "diff": {k: pst[a][m]["diff"] for k, (a, m) in primary.items() if a in pst},
    }

    out_summary = os.path.join(RES_DIR, os.environ.get("CEDAR_SUMMARY_NAME", "master_evaluation_summary_v3_final.json"))
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved master evaluation summary v3 -> {out_summary}")


if __name__ == "__main__":
    main()

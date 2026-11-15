import os
import glob
import math
import json
import hashlib
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, roc_curve

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "/content/scripts/step5.py"))))
if os.path.exists("/content/cedar_data"):
    DATA_DIR = "/content/cedar_data"
    RES_DIR = "/content/cedar_data"
else:
    DATA_DIR = os.path.join(_ROOT, "data")
    RES_DIR = os.path.join(_ROOT, "results")
PRED_DIR = os.path.join(RES_DIR, "model_preds_v2")


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
    requiring passage-level independence across edits of the same parent.
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


def compute_dataset_fingerprint(raw_splits: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(raw_splits.keys()):
        h.update(k.encode())
        for r in raw_splits[k]:
            h.update(f"{r['id']}|{r['origin_label']}|{r.get('edit_level','E0')}|{len(r['text'])}|".encode())
    return h.hexdigest()[:16]


def compute_full_content_and_cluster_digest(raw_splits: dict) -> str:
    """Full SHA-256 digest over split names, IDs, clustering keys (parent_id/prompt_id), labels, and full text."""
    h = hashlib.sha256()
    for k in sorted(raw_splits.keys()):
        if k in {"val_select", "val_calib"}:
            continue
        h.update(k.encode("utf-8"))
        for r in raw_splits[k]:
            h.update(
                f"{r['id']}|{r.get('parent_id','')}|{r.get('prompt_id','')}|{r['origin_label']}|{r.get('edit_level','E0')}|".encode("utf-8")
            )
            h.update(r["text"].encode("utf-8"))
    return h.hexdigest()


def resolve_cluster_id(it: dict, allow_record_id: bool = False) -> str:
    if it.get("parent_id"):
        return str(it["parent_id"])
    if it.get("prompt_id"):
        return str(it["prompt_id"])
    if allow_record_id and it.get("id"):
        return str(it["id"])
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


def parent_bootstrap_ci(items, scores, metric_fn, n_boot=1000, seed=42, allow_record_id=False):
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


def paired_parent_bootstrap_estimand_diff(items, eval_fn_a, eval_fn_b, n_boot=1000, seed=42, allow_record_id=False, zero_tol=1e-12):
    """
    Paired cluster bootstrap by parent_id/prompt_id for the exact table estimand (EvalA(b_idx) - EvalB(b_idx)),
    where EvalA / EvalB average per-seed metrics evaluated at each seed's own locked threshold in ascending
    numeric seed order (42, 1042, 2042). Conditional on saved runs; reports 8-decimal precision, 95% percentile
    bootstrap CI, and two-sided bootstrap sign-tail proportion p_boot (+1 smoothing, zero tolerance 1e-12).
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
        "metric_a": round(float(pt_a), 8),
        "metric_b": round(float(pt_b), 8),
        "diff": round(float(pt_diff), 8),
        "ci95": [round(float(lo), 8), round(float(hi), 8)],
        "p_boot": round(float(p_two), 8),
        "p_value": round(float(p_two), 8),
    }


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
    # Dense AURC across 20%..100% coverage
    dense_covs = np.linspace(0.2, 1.0, 81)
    dense_risks = [float(errs[order[: max(1, int(round(c * len(y_true))))]].mean()) for c in dense_covs]
    rc_curve["aurc_20_100"] = round(float(np.trapezoid(dense_risks, dense_covs) / 0.8), 8)
    return rc_curve


def evaluate_detector_dict(
    name: str,
    preds_by_split: dict,
    raw_splits: dict,
    val_calib_indices_in_val: list,
    locked_thresholds: dict = None,
    edit_intens_by_split: dict = None,
):
    """
    Evaluates a detector using STRICT conformal thresholds (`score > tau`) calibrated ONLY on
    the untouched 60-parent `val_calib` partition (never touching `val_select` or any test set):
      1) tau_cal_all5: 5%-FPR conformal upper threshold over ALL `val_calib` human texts (E0..E3)
      2) tau_cal_e0_5: 5%-FPR conformal upper threshold over UNEDITED `val_calib` human texts (E0)
    """
    if "val_calib" in preds_by_split:
        cal_rows = raw_splits["val_calib"]
        cal_scores = np.asarray(preds_by_split["val_calib"], dtype=np.float64)
    else:
        val_rows = raw_splits["val"]
        val_scores = np.asarray(preds_by_split["val"], dtype=np.float64)
        cal_rows = [val_rows[i] for i in val_calib_indices_in_val]
        cal_scores = val_scores[val_calib_indices_in_val]

    cal_y = np.array([r["origin_label"] for r in cal_rows])
    cal_e = np.array([r.get("edit_level", "E0") for r in cal_rows])
    cal_pids = [r.get("parent_id", r["id"]) for r in cal_rows]
    hum_all_idx = np.where(cal_y == 0)[0]
    hum_e0_idx = np.where((cal_y == 0) & (cal_e == "E0"))[0]

    tau_all5 = cluster_conformal_upper_threshold(cal_scores[hum_all_idx], [cal_pids[i] for i in hum_all_idx], alpha=0.05)
    tau_e0_5 = cluster_conformal_upper_threshold(cal_scores[hum_e0_idx], [cal_pids[i] for i in hum_e0_idx], alpha=0.05)

    if locked_thresholds is not None:
        assert abs(tau_all5 - float(locked_thresholds["tau_cal_all5"])) < 1e-5, (
            f"Threshold mismatch for {name}: {tau_all5} vs {locked_thresholds['tau_cal_all5']}"
        )
        assert abs(tau_e0_5 - float(locked_thresholds["tau_cal_e0_5"])) < 1e-5, (
            f"E0 threshold mismatch for {name}: {tau_e0_5} vs {locked_thresholds['tau_cal_e0_5']}"
        )
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
    wg_e02, wg_e03 = [], []
    ebal_e02, ebal_e03 = [], []
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
            "mean_lex_dist": round(float(np.mean([tid[i]["measured_lex_dist"] for i in np.where(m_lv)[0]])), 8),
        }
        if lv in {"E0", "E1", "E2"}:
            wg_e02.extend([1.0 - fpr_all5, tpr_all5])
            ebal_e02.append(bal_acc)
        if lv in {"E0", "E1", "E2", "E3"}:
            wg_e03.extend([1.0 - fpr_all5, tpr_all5])
            ebal_e03.append(bal_acc)

    res["test_id_recoverability"] = recov
    res["test_id_worst_group_acc_E0_E2"] = round(min(wg_e02), 8)
    res["test_id_worst_group_acc_E0_E3"] = round(min(wg_e03), 8)
    res["test_id_worst_edit_bal_acc_E0_E2"] = round(min(ebal_e02), 8)
    res["test_id_worst_edit_bal_acc_E0_E3"] = round(min(ebal_e03), 8)
    res["test_id_mean_bal_acc_E0_E3"] = round(float(np.mean(ebal_e03)), 8)

    m_e03 = np.isin(e_tid, ["E0", "E1", "E2", "E3"])
    if np.min(s_tid) >= 0.0 and np.max(s_tid) <= 1.0:
        res["test_id_ece_E0_E3"] = round(compute_ece(y_tid[m_e03], s_tid[m_e03]), 8)
    else:
        res["test_id_ece_E0_E3"] = None

    # Length-stratified audit on test_id (E0..E2)
    m_e02 = np.isin(e_tid, ["E0", "E1", "E2"])
    short_m = m_e02 & (wc_tid <= 180)
    long_m = m_e02 & (wc_tid > 180)
    res["test_id_length_stratified_E0_E2"] = {
        "short_le180_auroc": round(safe_auroc(y_tid[short_m], s_tid[short_m]), 8),
        "long_gt180_auroc": round(safe_auroc(y_tid[long_m], s_tid[long_m]), 8),
    }

    # 2. Crossed-OOD AR & Crossed-OOD LLaDA Masked Diffusion
    for ood_key in ["test_ood_ar", "test_ood_llada"]:
        rows_o = raw_splits[ood_key]
        s_o = np.asarray(preds_by_split[ood_key], dtype=np.float64)
        y_o = np.array([r["origin_label"] for r in rows_o])
        e_o = np.array([r["edit_level"] for r in rows_o])
        wc_o = np.array([r.get("wc", len(r["text"].split())) for r in rows_o])

        by_lv = {}
        wg_o, ebal_o = [], []
        for lv in ["E0", "E1", "E2"]:
            ml = (e_o == lv)
            mh = ml & (y_o == 0)
            mm = ml & (y_o == 1)
            fpr = float((s_o[mh] > tau_all5).mean())
            tpr = float((s_o[mm] > tau_all5).mean())
            fpr_e0 = float((s_o[mh] > tau_e0_5).mean())
            b_acc = 0.5 * ((1.0 - fpr) + tpr)
            by_lv[lv] = {
                "auroc": round(safe_auroc(y_o[ml], s_o[ml]), 8),
                "fpr_at_tau_all5": round(fpr, 8),
                "tpr_at_tau_all5": round(tpr, 8),
                "fpr_at_tau_e0_5": round(fpr_e0, 8),
                "bal_acc": round(b_acc, 8),
            }
            wg_o.extend([1.0 - fpr, tpr])
            ebal_o.append(b_acc)

        pt_auc, ci_auc = parent_bootstrap_ci(rows_o, s_o, safe_auroc, n_boot=1000)
        res[ood_key] = {
            "overall_auroc": round(pt_auc, 8),
            "overall_auroc_ci95": [round(ci_auc[0], 8), round(ci_auc[1], 8)],
            "worst_group_acc": round(min(wg_o), 8),
            "worst_edit_bal_acc": round(min(ebal_o), 8),
            "mean_bal_acc": round(float(np.mean(ebal_o)), 8),
            "short_le180_auroc": round(safe_auroc(y_o[wc_o <= 180], s_o[wc_o <= 180]), 8),
            "long_gt180_auroc": round(safe_auroc(y_o[wc_o > 180], s_o[wc_o > 180]), 8),
            "by_level": by_lv,
        }

    # 3. External ESL Student Essays (300 Unedited E0 + 300 Grammar-Polished E1 — ALL Human O=0)
    esl_rows = raw_splits["external_esl_full"]
    s_esl = np.asarray(preds_by_split["external_esl_full"], dtype=np.float64)
    e_esl = np.array([r["edit_level"] for r in esl_rows])
    res["external_esl_fairness"] = {
        "esl_E0_fpr_at_tau_all5": round(float((s_esl[e_esl == "E0"] > tau_all5).mean()), 8),
        "esl_E1_polished_fpr_at_tau_all5": round(float((s_esl[e_esl == "E1"] > tau_all5).mean()), 8),
        "esl_E0_fpr_at_tau_e0_5": round(float((s_esl[e_esl == "E0"] > tau_e0_5).mean()), 8),
        "esl_E1_polished_fpr_at_tau_e0_5": round(float((s_esl[e_esl == "E1"] > tau_e0_5).mean()), 8),
    }

    # 4. External MixSet Test Split
    ms_rows = raw_splits["external_mixset"]
    s_ms = np.asarray(preds_by_split["external_mixset"], dtype=np.float64)
    pure_mask = np.array([not r.get("is_completion", False) for r in ms_rows])
    y_ms = np.array([r["origin_label"] for r in ms_rows])

    hwt_pol_mask = np.array([("polish" in r["subset"]) and (r["origin_label"] == 0) for r in ms_rows])
    mgt_hum_mask = np.array([("humanize" in r["subset"]) and (r["origin_label"] == 1) for r in ms_rows])
    mgt_pol_mask = np.array([("MGT_polish" in r["subset"]) and (r["origin_label"] == 1) for r in ms_rows])

    res["external_mixset"] = {
        "pure_origin_auroc": round(safe_auroc(y_ms[pure_mask], s_ms[pure_mask]), 8),
        "all_720_auroc": round(safe_auroc(y_ms, s_ms), 8),
        "hwt_llm_polished_fpr_tau_all5": round(float((s_ms[hwt_pol_mask] > tau_all5).mean()), 8),
        "hwt_llm_polished_fpr_tau_e0_5": round(float((s_ms[hwt_pol_mask] > tau_e0_5).mean()), 8),
        "mgt_humanized_tpr_tau_all5": round(float((s_ms[mgt_hum_mask] > tau_all5).mean()), 8),
        "mgt_human_polished_tpr_tau_all5": round(float((s_ms[mgt_pol_mask] > tau_all5).mean()), 8),
    }

    # 5. External Toloka Beemo
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

    # 6. External MAGE Wild Unseen Domains
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

    # 7. Selective Risk-Coverage Curve across E0..E4 on test_id
    #    Use boundary-normalized conformal margin so positive [tau, 1] and negative [0, tau]
    #    sides share unit full-scale [0, 1] (and sample-extremum scaling for unbounded scores),
    #    plus edit-intensity modulation when edit_intensity probe is present.
    if np.min(s_tid) >= -1e-6 and np.max(s_tid) <= 1.0 + 1e-6 and 0.0 < tau_all5 < 1.0:
        denom_tid = np.where(s_tid > tau_all5, max(1.0 - tau_all5, 1e-6), max(tau_all5, 1e-6))
    else:
        pos_span = max(float(np.max(s_tid) - tau_all5), 1e-6)
        neg_span = max(float(tau_all5 - np.min(s_tid)), 1e-6)
        denom_tid = np.where(s_tid > tau_all5, pos_span, neg_span)
    margin_conf = np.abs(s_tid - tau_all5) / denom_tid
    res["selective_rc_margin_only_E0_E4"] = compute_risk_coverage(y_tid, s_tid, tau_all5, margin_conf, edit_levels=e_tid)

    if edit_intens_by_split and "test_id" in edit_intens_by_split:
        a_hat = np.asarray(edit_intens_by_split["test_id"], dtype=np.float64)
        intens_conf = margin_conf * (1.0 - 0.5 * np.clip(a_hat, 0.0, 1.0))
        res["selective_rc_intensity_aware_E0_E4"] = compute_risk_coverage(y_tid, s_tid, tau_all5, intens_conf, edit_levels=e_tid)
    else:
        res["selective_rc_intensity_aware_E0_E4"] = res["selective_rc_margin_only_E0_E4"]

    return res


def main():
    raw_splits = {}
    for sp in ["train", "val", "test_id", "test_ood_ar", "test_ood_llada"]:
        with open(os.path.join(DATA_DIR, f"split_{sp}.json")) as f:
            raw_splits[sp] = json.load(f)
    for ext in ["external_esl_full", "external_mixset", "external_beemo", "external_mage_wild"]:
        with open(os.path.join(DATA_DIR, f"{ext}.json")) as f:
            raw_splits[ext] = json.load(f)

    # Strict validation of MixSet 120-source-family clustering (never silently fall back to record IDs!)
    ms_check = raw_splits["external_mixset"]
    assert len(ms_check) == 720, f"Expected 720 MixSet records, got {len(ms_check)}"
    assert all(str(r.get("parent_id", "")).startswith("ms_orig_") for r in ms_check), (
        "Missing ms_orig_* parent_id in external_mixset.json!"
    )
    assert len({r["parent_id"] for r in ms_check}) == 120, "Expected 120 unique MixSet source families (All 720)!"
    assert len({r["parent_id"] for r in ms_check if not r.get("is_completion", False)}) == 120, (
        "Expected 120 unique MixSet source families (Pure 620)!"
    )

    # Identical deterministic partition of split_val into val_select (first 60) and val_calib (last 60)
    val_pids = sorted({r["parent_id"] for r in raw_splits["val"]})
    assert len(val_pids) == 120, f"Expected 120 val parents, got {len(val_pids)}"
    sel_set = set(val_pids[:60])
    cal_set = set(val_pids[60:])
    val_calib_indices = [i for i, r in enumerate(raw_splits["val"]) if r["parent_id"] in cal_set]
    raw_splits["val_select"] = [r for r in raw_splits["val"] if r["parent_id"] in sel_set]
    raw_splits["val_calib"] = [raw_splits["val"][i] for i in val_calib_indices]
    expected_fp = compute_dataset_fingerprint(raw_splits)
    full_digest = compute_full_content_and_cluster_digest(raw_splits)
    expected_row_ids = {k: [r["id"] for r in v] for k, v in raw_splits.items() if k != "train"}

    summary = {
        "calibration_protocol": "parent-cluster split-conformal risk control (`score > tau`, alpha=0.05) on untouched 60-parent val_calib",
        "bootstrap_protocol": "1000-resample cluster bootstrap by parent_id/prompt_id (record id only for independent MAGE Wild), conditional on saved 3-seed runs in numeric seed order (42, 1042, 2042); two-sided sign-tail p_boot with +1 smoothing and 1e-12 zero tolerance, unadjusted for multiplicity",
        "data_fingerprint": expected_fp,
        "full_content_and_cluster_digest": full_digest,
        "pretrained_and_probes": {},
        "trained_per_seed": {},
        "trained_multi_seed_agg": {},
        "nsd_subspace_diagnostics": {},
        "paired_significance_tests": {},
    }

    pre_preds = {}
    pre_path = os.path.join(RES_DIR, "preds_pretrained_and_probes.json")
    if os.path.exists(pre_path):
        with open(pre_path) as f:
            pre_preds = json.load(f)
        for m_name, p_dict in pre_preds.items():
            summary["pretrained_and_probes"][m_name] = evaluate_detector_dict(
                m_name, p_dict, raw_splits, val_calib_indices
            )

    by_variant = defaultdict(list)
    by_variant_payloads = defaultdict(list)
    by_variant_nsd = defaultdict(list)

    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.json")))
    loaded_payloads = []
    for fp in pred_files:
        with open(fp) as f:
            payload = json.load(f)
        assert payload.get("data_fingerprint") == expected_fp, (
            f"STALE PREDICTION DETECTED in {fp}: fp={payload.get('data_fingerprint')} != {expected_fp}"
        )
        for sp_k, rids in payload.get("row_ids", {}).items():
            assert rids == expected_row_ids[sp_k], f"Row-ID misalignment in {fp} split {sp_k}!"
        loaded_payloads.append(payload)

    # Sort deterministically by (run_name, numeric seed)
    loaded_payloads.sort(key=lambda pl: (pl["run_name"], int(pl["seed"])))

    for payload in loaded_payloads:
        r_name = payload["run_name"]
        sd = int(payload["seed"])
        key = f"{r_name}_seed{sd}"
        ev = evaluate_detector_dict(
            key,
            payload["preds"],
            raw_splits,
            val_calib_indices,
            locked_thresholds=payload.get("locked_thresholds"),
            edit_intens_by_split=payload.get("edit_intensity"),
        )
        ev["seed"] = sd
        ev["train_seconds"] = payload["train_seconds"]
        ev["trainable_params"] = payload.get("trainable_params")
        ev["best_val_select"] = payload.get("best_val_select")
        summary["trained_per_seed"][key] = ev
        by_variant[r_name].append(ev)
        by_variant_payloads[r_name].append(payload)
        if payload.get("nsd_diagnostics", {}).get("active"):
            by_variant_nsd[r_name].append(payload["nsd_diagnostics"])

    def ms(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        if not vals:
            return {"mean": None, "std": None}
        return {"mean": round(float(np.mean(vals)), 8), "std": round(float(np.std(vals, ddof=0)), 8)}

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

        summary["trained_multi_seed_agg"][r_name] = {
            "n_seeds": len(ev_list),
            "trainable_params": ev_list[0].get("trainable_params"),
            "mean_train_seconds": ms([x["train_seconds"] for x in ev_list]),
            "id_wg_E0_E2": ms([x["test_id_worst_group_acc_E0_E2"] for x in ev_list]),
            "id_wg_E0_E3": ms([x["test_id_worst_group_acc_E0_E3"] for x in ev_list]),
            "id_worst_ebal_E0_E3": ms([x["test_id_worst_edit_bal_acc_E0_E3"] for x in ev_list]),
            "id_mean_bal_E0_E3": ms([x["test_id_mean_bal_acc_E0_E3"] for x in ev_list]),
            "id_ece_E0_E3": ms([x["test_id_ece_E0_E3"] for x in ev_list]),
            "id_short_le180_auroc": ms([x["test_id_length_stratified_E0_E2"]["short_le180_auroc"] for x in ev_list]),
            "id_long_gt180_auroc": ms([x["test_id_length_stratified_E0_E2"]["long_gt180_auroc"] for x in ev_list]),
            "id_recoverability": recov_agg,
            "ood_ar_auroc": ms([x["test_ood_ar"]["overall_auroc"] for x in ev_list]),
            "ood_ar_wg_acc": ms([x["test_ood_ar"]["worst_group_acc"] for x in ev_list]),
            "ood_ar_mean_bal": ms([x["test_ood_ar"]["mean_bal_acc"] for x in ev_list]),
            "ood_ar_E2_auroc": ms([x["test_ood_ar"]["by_level"]["E2"]["auroc"] for x in ev_list]),
            "ood_llada_auroc": ms([x["test_ood_llada"]["overall_auroc"] for x in ev_list]),
            "ood_llada_wg_acc": ms([x["test_ood_llada"]["worst_group_acc"] for x in ev_list]),
            "ood_llada_mean_bal": ms([x["test_ood_llada"]["mean_bal_acc"] for x in ev_list]),
            "ood_llada_E2_auroc": ms([x["test_ood_llada"]["by_level"]["E2"]["auroc"] for x in ev_list]),
            "esl_E0_fpr": ms([x["external_esl_fairness"]["esl_E0_fpr_at_tau_all5"] for x in ev_list]),
            "esl_E1_fpr": ms([x["external_esl_fairness"]["esl_E1_polished_fpr_at_tau_all5"] for x in ev_list]),
            "esl_E1_fpr_e0cal": ms([x["external_esl_fairness"]["esl_E1_polished_fpr_at_tau_e0_5"] for x in ev_list]),
            "mixset_pure_auroc": ms([x["external_mixset"]["pure_origin_auroc"] for x in ev_list]),
            "mixset_all_auroc": ms([x["external_mixset"]["all_720_auroc"] for x in ev_list]),
            "mixset_hwt_pol_fpr": ms([x["external_mixset"]["hwt_llm_polished_fpr_tau_all5"] for x in ev_list]),
            "mixset_hwt_pol_fpr_e0cal": ms([x["external_mixset"]["hwt_llm_polished_fpr_tau_e0_5"] for x in ev_list]),
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
            "frac_edit_energy_removed": ms([d["frac_edit_energy_removed"] for d in diag_list]),
            "frac_orig_energy_removed": ms([d["frac_orig_energy_removed"] for d in diag_list]),
        }

    # Paired Parent-Clustered Bootstrap Significance Tests on EXACT Per-Seed Mean Estimand!
    if "mb_cedar_full" in by_variant_payloads:
        y_ar = np.array([r["origin_label"] for r in raw_splits["test_ood_ar"]])
        y_ll = np.array([r["origin_label"] for r in raw_splits["test_ood_llada"]])
        y_bm = np.array([r["origin_label"] for r in raw_splits["external_beemo"]])
        ms_all_items = raw_splits["external_mixset"]
        y_ms_all = np.array([r["origin_label"] for r in ms_all_items])
        ms_pure_items = [r for r in ms_all_items if not r.get("is_completion", False)]
        ms_pure_idx = np.array([i for i, r in enumerate(ms_all_items) if not r.get("is_completion", False)], dtype=np.int64)
        y_ms_pure = np.array([r["origin_label"] for r in ms_pure_items])
        ll_lm_idx = np.array([i for i, r in enumerate(raw_splits["test_ood_llada"]) if 120 <= r.get("wc", len(r["text"].split())) <= 220], dtype=np.int64)
        ll_lm_items = [raw_splits["test_ood_llada"][i] for i in ll_lm_idx]
        y_ll_lm = np.array([r["origin_label"] for r in ll_lm_items])
        y_mg = np.array([r["origin_label"] for r in raw_splits["external_mage_wild"]])
        y_tid = np.array([r["origin_label"] for r in raw_splits["test_id"]])
        e_tid = np.array([r["edit_level"] for r in raw_splits["test_id"]])

        lv_map = {"E0": 0, "E1": 2, "E2": 4, "E3": 6, "E4": 8}
        g_tid = np.array([lv_map[r["edit_level"]] + int(r["origin_label"]) for r in raw_splits["test_id"]], dtype=np.int64)

        def make_seed_avg_fns(payload_list):
            tid_corrs = [
                ((np.asarray(pl["preds"]["test_id"], dtype=np.float64) > float(pl["locked_thresholds"]["tau_cal_all5"])).astype(np.int64) == y_tid).astype(np.float64)
                for pl in payload_list
            ]
            s_ar = [np.asarray(pl["preds"]["test_ood_ar"], dtype=np.float64) for pl in payload_list]
            s_ll = [np.asarray(pl["preds"]["test_ood_llada"], dtype=np.float64) for pl in payload_list]
            s_ll_lm = [np.asarray(pl["preds"]["test_ood_llada"], dtype=np.float64)[ll_lm_idx] for pl in payload_list]
            s_bm = [np.asarray(pl["preds"]["external_beemo"], dtype=np.float64) for pl in payload_list]
            s_ms = [np.asarray(pl["preds"]["external_mixset"], dtype=np.float64)[ms_pure_idx] for pl in payload_list]
            s_ms_all = [np.asarray(pl["preds"]["external_mixset"], dtype=np.float64) for pl in payload_list]
            s_mg = [np.asarray(pl["preds"]["external_mage_wild"], dtype=np.float64) for pl in payload_list]

            def tid_ebal(b_idx):
                gb = g_tid[b_idx]
                cnt = np.bincount(gb, minlength=10)[:8]
                if np.any(cnt == 0):
                    return float("nan")
                seed_vals = []
                for cr in tid_corrs:
                    g_acc = np.bincount(gb, weights=cr[b_idx], minlength=10)[:8] / cnt
                    seed_vals.append(float(np.mean(0.5 * (g_acc[0::2] + g_acc[1::2]))))
                return float(np.mean(seed_vals))

            def tid_wg(b_idx):
                gb = g_tid[b_idx]
                cnt = np.bincount(gb, minlength=10)[:8]
                if np.any(cnt == 0):
                    return float("nan")
                seed_vals = []
                for cr in tid_corrs:
                    g_acc = np.bincount(gb, weights=cr[b_idx], minlength=10)[:8] / cnt
                    seed_vals.append(float(np.min(g_acc)))
                return float(np.mean(seed_vals))

            return {
                "tid_wg": tid_wg,
                "tid_ebal": tid_ebal,
                "ar_auc": lambda b_idx: float(np.mean([safe_auroc(y_ar[b_idx], sv[b_idx]) for sv in s_ar])),
                "ll_auc": lambda b_idx: float(np.mean([safe_auroc(y_ll[b_idx], sv[b_idx]) for sv in s_ll])),
                "ll_lm_auc": lambda b_idx: float(np.mean([safe_auroc(y_ll_lm[b_idx], sv[b_idx]) for sv in s_ll_lm])),
                "bm_auc": lambda b_idx: float(np.mean([safe_auroc(y_bm[b_idx], sv[b_idx]) for sv in s_bm])),
                "ms_auc": lambda b_idx: float(np.mean([safe_auroc(y_ms_pure[b_idx], sv[b_idx]) for sv in s_ms])),
                "ms_all_auc": lambda b_idx: float(np.mean([safe_auroc(y_ms_all[b_idx], sv[b_idx]) for sv in s_ms_all])),
                "mg_auc": lambda b_idx: float(np.mean([safe_auroc(y_mg[b_idx], sv[b_idx]) for sv in s_mg])),
            }

        cedar_fns = make_seed_avg_fns(by_variant_payloads["mb_cedar_full"])
        competitor_fns = {}

        for comp_name in [
            "mb_unedited_erm",
            "mb_augmented_erm",
            "mb_multitask_2head",
            "mb_supcon_orth_disentangle",
            "mb_erm_paired_consist",
            "mb_erm_group_dro",
            "mb_ablation_no_nsd",
            "mb_ablation_no_dte",
            "mb_ablation_param_matched_mlp",
            "mb_ablation_no_pair",
            "mb_ablation_no_gdro",
            "mb_ablation_unprotected_nsd",
        ]:
            if comp_name in by_variant_payloads:
                competitor_fns[comp_name] = make_seed_avg_fns(by_variant_payloads[comp_name])

        cal_rows = raw_splits["val_calib"]
        cal_y = np.array([r["origin_label"] for r in cal_rows])
        cal_pids = [r.get("parent_id", r["id"]) for r in cal_rows]
        hum_cal_idx = np.where(cal_y == 0)[0]

        for pre_name in ["desklib_raid_v1", "binoculars_qwen3b", "fast_detectgpt_qwen3b", "radar_vicuna7b"]:
            if pre_name in pre_preds:
                p_cal = np.asarray(pre_preds[pre_name]["val"], dtype=np.float64)[val_calib_indices]
                p_tau = cluster_conformal_upper_threshold(
                    p_cal[hum_cal_idx], [cal_pids[i] for i in hum_cal_idx], alpha=0.05
                )
                pseudo_pl = [{
                    "preds": pre_preds[pre_name],
                    "locked_thresholds": {"tau_cal_all5": p_tau},
                }]
                competitor_fns[pre_name] = make_seed_avg_fns(pseudo_pl)

        for c_name, c_fns in competitor_fns.items():
            summary["paired_significance_tests"][f"mb_cedar_full_vs_{c_name}"] = {
                "test_id_worst_group_E0_E3": paired_parent_bootstrap_estimand_diff(
                    raw_splits["test_id"], cedar_fns["tid_wg"], c_fns["tid_wg"], n_boot=1000
                ),
                "test_id_mean_bal_E0_E3": paired_parent_bootstrap_estimand_diff(
                    raw_splits["test_id"], cedar_fns["tid_ebal"], c_fns["tid_ebal"], n_boot=1000
                ),
                "test_ood_ar_auroc": paired_parent_bootstrap_estimand_diff(
                    raw_splits["test_ood_ar"], cedar_fns["ar_auc"], c_fns["ar_auc"], n_boot=1000
                ),
                "test_ood_llada_auroc": paired_parent_bootstrap_estimand_diff(
                    raw_splits["test_ood_llada"], cedar_fns["ll_auc"], c_fns["ll_auc"], n_boot=1000
                ),
                "test_ood_llada_len120_220_auroc": paired_parent_bootstrap_estimand_diff(
                    ll_lm_items, cedar_fns["ll_lm_auc"], c_fns["ll_lm_auc"], n_boot=1000
                ),
                "external_beemo_auroc": paired_parent_bootstrap_estimand_diff(
                    raw_splits["external_beemo"], cedar_fns["bm_auc"], c_fns["bm_auc"], n_boot=1000
                ),
                "external_mixset_pure_auroc": paired_parent_bootstrap_estimand_diff(
                    ms_pure_items, cedar_fns["ms_auc"], c_fns["ms_auc"], n_boot=1000
                ),
                "external_mixset_all_auroc": paired_parent_bootstrap_estimand_diff(
                    ms_all_items, cedar_fns["ms_all_auc"], c_fns["ms_all_auc"], n_boot=1000
                ),
                "external_mage_wild_auroc": paired_parent_bootstrap_estimand_diff(
                    raw_splits["external_mage_wild"], cedar_fns["mg_auc"], c_fns["mg_auc"], n_boot=1000, allow_record_id=True
                ),
            }

        if "db_cedar_full" in by_variant_payloads:
            db_cedar_fns = make_seed_avg_fns(by_variant_payloads["db_cedar_full"])
            for db_comp in ["db_unedited_erm", "db_augmented_erm", "db_supcon_orth_disentangle"]:
                if db_comp in by_variant_payloads:
                    dc_fns = make_seed_avg_fns(by_variant_payloads[db_comp])
                    summary["paired_significance_tests"][f"db_cedar_full_vs_{db_comp}"] = {
                        "test_id_worst_group_E0_E3": paired_parent_bootstrap_estimand_diff(
                            raw_splits["test_id"], db_cedar_fns["tid_wg"], dc_fns["tid_wg"], n_boot=1000
                        ),
                        "test_id_mean_bal_E0_E3": paired_parent_bootstrap_estimand_diff(
                            raw_splits["test_id"], db_cedar_fns["tid_ebal"], dc_fns["tid_ebal"], n_boot=1000
                        ),
                        "test_ood_llada_auroc": paired_parent_bootstrap_estimand_diff(
                            raw_splits["test_ood_llada"], db_cedar_fns["ll_auc"], dc_fns["ll_auc"], n_boot=1000
                        ),
                        "test_ood_llada_len120_220_auroc": paired_parent_bootstrap_estimand_diff(
                            ll_lm_items, db_cedar_fns["ll_lm_auc"], dc_fns["ll_lm_auc"], n_boot=1000
                        ),
                        "external_mixset_pure_auroc": paired_parent_bootstrap_estimand_diff(
                            ms_pure_items, db_cedar_fns["ms_auc"], dc_fns["ms_auc"], n_boot=1000
                        ),
                        "external_mixset_all_auroc": paired_parent_bootstrap_estimand_diff(
                            ms_all_items, db_cedar_fns["ms_all_auc"], dc_fns["ms_all_auc"], n_boot=1000
                        ),
                    }

        # Source-separated primary calibration disclosure + Post-hoc stratified validation calibration sensitivity
        val_rows_all = raw_splits["val"]
        mg_val_pids = sorted({r["parent_id"] for r in val_rows_all if r["parent_id"].startswith("mg_")})
        nr_val_pids = sorted({r["parent_id"] for r in val_rows_all if r["parent_id"].startswith("nr_")})
        strat_cal_pids = set(mg_val_pids[30:]) | set(nr_val_pids[30:])
        strat_cal_idx = np.array([i for i, r in enumerate(val_rows_all) if r["parent_id"] in strat_cal_pids], dtype=np.int64)
        strat_cal_rows = [val_rows_all[i] for i in strat_cal_idx]
        strat_cal_y = np.array([r["origin_label"] for r in strat_cal_rows])
        strat_cal_hum = np.where(strat_cal_y == 0)[0]
        strat_cal_pids_list = [strat_cal_rows[i]["parent_id"] for i in strat_cal_hum]

        tid_rows_all = raw_splits["test_id"]
        tid_nr_hum_e03 = np.array([(r["parent_id"].startswith("nr_")) and (r["origin_label"] == 0) and (r["edit_level"] in {"E0","E1","E2","E3"}) for r in tid_rows_all])
        tid_mg_hum_e03 = np.array([(r["parent_id"].startswith("mg_")) and (r["origin_label"] == 0) and (r["edit_level"] in {"E0","E1","E2","E3"}) for r in tid_rows_all])
        tid_all_hum_e03 = tid_nr_hum_e03 | tid_mg_hum_e03

        cal_sens = {}
        for m_key in ["mb_unedited_erm", "mb_augmented_erm", "mb_supcon_orth_disentangle", "mb_cedar_full", "db_augmented_erm", "db_cedar_full"]:
            if m_key not in by_variant_payloads:
                continue
            nr_fprs, mg_fprs, strat_fprs, strat_wgs = [], [], [], []
            for pl in by_variant_payloads[m_key]:
                s_t = np.asarray(pl["preds"]["test_id"], dtype=np.float64)
                tau_loc = float(pl["locked_thresholds"]["tau_cal_all5"])
                nr_fprs.append(float((s_t[tid_nr_hum_e03] > tau_loc).mean()))
                mg_fprs.append(float((s_t[tid_mg_hum_e03] > tau_loc).mean()))

                s_v = np.asarray(pl["preds"]["val"], dtype=np.float64)[strat_cal_idx]
                tau_st = cluster_conformal_upper_threshold(s_v[strat_cal_hum], strat_cal_pids_list, alpha=0.05)
                strat_fprs.append(float((s_t[tid_all_hum_e03] > tau_st).mean()))
                g_st = []
                for lv in ["E0", "E1", "E2", "E3"]:
                    mh = (e_tid == lv) & (y_tid == 0)
                    mm = (e_tid == lv) & (y_tid == 1)
                    g_st.extend([1.0 - float((s_t[mh] > tau_st).mean()), float((s_t[mm] > tau_st).mean())])
                strat_wgs.append(float(min(g_st)))
            cal_sens[m_key] = {
                "locked_nr_to_nr_human_fpr_E0_E3": ms(nr_fprs),
                "locked_nr_to_mg_human_fpr_E0_E3": ms(mg_fprs),
                "posthoc_stratified_30mg_30nr_human_fpr_E0_E3": ms(strat_fprs),
                "posthoc_stratified_30mg_30nr_id_wg_E0_E3": ms(strat_wgs),
            }
        summary["validation_partition_and_calibration_sensitivity"] = cal_sens

        # Subgroup & Length-Reweighted Audits (Domain, Generator, Editor, and IPW Length-Matched LLaDA)
        wc_ll_lm = np.array([r.get("wc", len(r["text"].split())) for r in ll_lm_items], dtype=np.float64)
        bins_ll = [120, 145, 170, 195, 221]
        bin_ids_ll = np.digitize(wc_ll_lm, bins_ll) - 1  # 0..3
        ipw_w_ll = np.ones(len(ll_lm_items), dtype=np.float64)
        for b in range(4):
            mb0 = (bin_ids_ll == b) & (y_ll_lm == 0)
            mb1 = (bin_ids_ll == b) & (y_ll_lm == 1)
            if mb0.any() and mb1.any():
                ipw_w_ll[mb0] = 0.25 / float(mb0.sum())
                ipw_w_ll[mb1] = 0.25 / float(mb1.sum())

        tid_nr_mask = np.array([r["parent_id"].startswith("nr_") for r in tid_rows_all])
        tid_mg_mask = np.array([r["parent_id"].startswith("mg_") for r in tid_rows_all])
        tid_gens = sorted({r["draft_generator"] for r in tid_rows_all if r["origin_label"] == 1})
        ar_rows = raw_splits["test_ood_ar"]
        e_ar = np.array([r["edit_level"] for r in ar_rows])
        ar_phi_mask = np.array(["Phi-4" in r.get("editor_model", "") or "phi-4" in r.get("editor_model", "").lower() for r in ar_rows])
        ar_seen_ed_mask = (e_ar != "E0") & (~ar_phi_mask)

        sub_audits = {
            "llada_len120_220_counts": {"n_total": int(len(ll_lm_items)), "n_human": int((y_ll_lm == 0).sum()), "n_machine": int((y_ll_lm == 1).sum())},
            "models": {},
        }
        audit_keys = [
            "mb_unedited_erm", "mb_augmented_erm", "mb_supcon_orth_disentangle",
            "mb_erm_paired_consist", "mb_erm_group_dro", "mb_cedar_full",
            "db_unedited_erm", "db_augmented_erm", "db_supcon_orth_disentangle", "db_cedar_full",
        ]
        for ak in audit_keys:
            if ak not in by_variant_payloads:
                continue
            lm_aucs, ipw_aucs, nr_wgs, mg_wgs = [], [], [], []
            gen_tprs = defaultdict(list)
            phi_aucs, seen_ed_aucs = [], []
            for pl in by_variant_payloads[ak]:
                tau_loc = float(pl["locked_thresholds"]["tau_cal_all5"])
                s_ll_sub = np.asarray(pl["preds"]["test_ood_llada"], dtype=np.float64)[ll_lm_idx]
                lm_aucs.append(safe_auroc(y_ll_lm, s_ll_sub))
                ipw_aucs.append(float(roc_auc_score(y_ll_lm, s_ll_sub, sample_weight=ipw_w_ll)))

                s_t = np.asarray(pl["preds"]["test_id"], dtype=np.float64)
                for d_mask, d_list in [(tid_nr_mask, nr_wgs), (tid_mg_mask, mg_wgs)]:
                    g_d = []
                    for lv in ["E0", "E1", "E2", "E3"]:
                        mh = d_mask & (e_tid == lv) & (y_tid == 0)
                        mm = d_mask & (e_tid == lv) & (y_tid == 1)
                        g_d.extend([1.0 - float((s_t[mh] > tau_loc).mean()), float((s_t[mm] > tau_loc).mean())])
                    d_list.append(float(min(g_d)))
                for gn in tid_gens:
                    gm = np.array([(r["draft_generator"] == gn) and (r["edit_level"] in {"E0", "E1", "E2", "E3"}) for r in tid_rows_all])
                    gen_tprs[gn].append(float((s_t[gm] > tau_loc).mean()))

                s_ar_v = np.asarray(pl["preds"]["test_ood_ar"], dtype=np.float64)
                if ar_phi_mask.any():
                    phi_aucs.append(safe_auroc(y_ar[ar_phi_mask], s_ar_v[ar_phi_mask]))
                if ar_seen_ed_mask.any():
                    seen_ed_aucs.append(safe_auroc(y_ar[ar_seen_ed_mask], s_ar_v[ar_seen_ed_mask]))

            sub_audits["models"][ak] = {
                "llada_len120_220_auroc": ms(lm_aucs),
                "llada_len120_220_ipw_auroc": ms(ipw_aucs),
                "test_id_no_robots_wg_E0_E3": ms(nr_wgs),
                "test_id_mage_wg_E0_E3": ms(mg_wgs),
                "test_id_tpr_by_generator_E0_E3": {gn: ms(vals) for gn, vals in gen_tprs.items()},
                "test_ood_ar_unseen_phi4_editor_auroc": ms(phi_aucs),
                "test_ood_ar_seen_editors_auroc": ms(seen_ed_aucs),
            }
        summary["subgroup_and_length_audits"] = sub_audits

    out_summary = os.path.join(RES_DIR, "master_evaluation_summary_v2.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved master evaluation summary v2 -> {out_summary}")


if __name__ == "__main__":
    main()

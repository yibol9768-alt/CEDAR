import os
import re
import gc
import sys
import json
import time
import math
import hashlib
import random
import warnings
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "/content/scripts/v3_train_cedar_v2_suite.py"))))
if os.path.exists("/content/cedar_data") and os.path.exists("/content/cedar_data_v2"):
    _CODE_ROOT = "/content"
    V1_DIR = "/content/cedar_data"
    V2_DIR = "/content/cedar_data_v2"
    OUT_DIR = "/content/cedar_results_v3"
else:
    _CODE_ROOT = _REPO_ROOT
    V1_DIR = os.path.join(_REPO_ROOT, "data")
    V2_DIR = os.path.join(_REPO_ROOT, "data_v2")
    OUT_DIR = os.path.join(_REPO_ROOT, "results_v3")
sys.path.insert(0, _CODE_ROOT)
from src.cedar_model import CEDARDetector
from src.cedar_v2_model import CEDARv2Detector, eps_unit_norm

PRED_DIR = os.path.join(OUT_DIR, "model_preds_v3")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

CODE_VERSION = "cedar_v3_spotlight"
E2_DIR = "/content/cedar_data_e2"
E2_SPLITS = ["val_shift", "test_confirm2", "test_confirm3"]
SPLIT_FILE = {"test_confirm3": "/content/cedar_data_c3/split_test_confirm3.json"}  # ICML Confirm-3 (sealed; docs/CONFIRM3_SPEC_2026-11-15.md)

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can",
    "that", "which", "who", "whom", "this", "these", "those", "it", "its", "they", "them",
    "their", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "not", "no",
    "so", "if", "then", "than", "too", "very", "just", "into", "about", "over", "after",
}

SENT_SPLIT_RE = re.compile(r"[^.!?\n]+[.!?\n]*")
WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def exact_sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def det_rank_key(tag: str, pid: str) -> str:
    return hashlib.sha256(f"{tag}|{pid}".encode("utf-8")).hexdigest()


def cluster_conformal_upper_threshold(hum_scores: np.ndarray, hum_pids: list, alpha: float = 0.05) -> float:
    """
    Parent-cluster split-conformal risk control threshold for strict rule `score > tau`.
    Controls marginal expected cluster-averaged human FPR <= alpha under cluster exchangeability.
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


def compute_full_split_fingerprint(raw_splits: dict, val_manifest: dict) -> str:
    """
    Cryptographic fingerprint over FULL text SHA-256, family IDs, labels, edit levels, registers, and val_manifest.
    """
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


def load_and_verify_all_splits():
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

    # Verify exact 30 MAGE + 30 No Robots domain balance in BOTH val_select and val_calib!
    for vk in ["val_select", "val_calib"]:
        dom_p = {r["parent_id"]: r["domain"] for r in raw[vk]}
        dom_cnt = Counter(dom_p.values())
        assert dom_cnt["instruct_prose"] == 30 and dom_cnt["public_expository"] == 30, (
            f"Validation domain confounding detected in {vk}: {dom_cnt}"
        )

    for sp in ["test_id", "test_ood_ar", "test_ood_llada"]:
        raw[sp] = json.load(open(os.path.join(V1_DIR, f"split_{sp}.json")))
    for sp in E2_SPLITS:  # E2 (docs/E2_plan_2026-11-15.md): only present when the E2 data has been built
        fp = SPLIT_FILE.get(sp, os.path.join(E2_DIR, f"split_{sp}.json"))
        if os.path.exists(fp):
            raw[sp] = json.load(open(fp))
    for ext in ["external_esl_full", "external_mixset", "external_beemo", "external_mage_wild"]:
        raw[ext] = json.load(open(os.path.join(V1_DIR, f"{ext}.json")))

    # Also create deterministic 375 fit + 125 op source-stratified split of v1 train for the 2x2 bridge!
    v1_p_dom = {r["parent_id"]: r["domain"] for r in raw["train_v1"]}
    v1_nr = sorted([p for p, d in v1_p_dom.items() if d == "instruct_prose"], key=lambda p: det_rank_key("bridge_v1_op", p))
    v1_mg = sorted([p for p, d in v1_p_dom.items() if d == "public_expository"], key=lambda p: det_rank_key("bridge_v1_op", p))
    assert len(v1_nr) == 250 and len(v1_mg) == 250
    v1_fit_set = set(v1_nr[:188] + v1_mg[:187])  # 375 parents
    v1_op_set = set(v1_nr[188:] + v1_mg[187:])   # 125 disjoint parents
    raw["train_v1_fit"] = [r for r in raw["train_v1"] if r["parent_id"] in v1_fit_set]
    raw["train_v1_op"] = [r for r in raw["train_v1"] if r["parent_id"] in v1_op_set]

    # Strict parent disjointness & symmetric multiset verification across core v2 splits
    expected_lvs = {
        "train_fit_v2": ["E0", "E1", "E2", "E3"],
        "train_op_v2": ["E0", "E1", "E2", "E3"],
        "val_select": ["E0", "E1", "E2", "E3"],
        "val_calib": ["E0", "E1", "E2", "E3"],
        "test_id": ["E0", "E1", "E2", "E3", "E4"],
        "test_ood_ar": ["E0", "E1", "E2"],
        "test_ood_llada": ["E0", "E1", "E2"],
        "test_confirm": ["E0", "E1", "E2", "E3"],
    }
    for sp in E2_SPLITS:
        if sp in raw:
            expected_lvs[sp] = ["E0", "E1", "E2", "E3"]
    seen_parents = {}
    for k, lvs in expected_lvs.items():
        pids = {r["parent_id"] for r in raw[k]}
        for pid in pids:
            assert pid not in seen_parents, f"LEAKAGE DETECTED: {pid} in both {seen_parents[pid]} and {k}"
            seen_parents[pid] = k
        target_counter = Counter((o, lv) for o in (0, 1) for lv in lvs)
        by_p = defaultdict(list)
        sym_check = {}
        for r in raw[k]:
            o_int = int(r["origin_label"])
            lv = r["edit_level"]
            pid = r["parent_id"]
            by_p[pid].append((o_int, lv))
            sym_check[(pid, o_int, lv)] = (
                r.get("editor_model", "none"),
                r.get("edit_tmpl_idx", -1),
                r.get("edit_register", 0),
            )
        for pid, clist in by_p.items():
            assert Counter(clist) == target_counter, f"Cell multiset mismatch for {pid} in {k}"
            for lv in lvs:
                assert sym_check[(pid, 0, lv)] == sym_check[(pid, 1, lv)], (
                    f"Asymmetric edit metadata on {pid} {lv} in {k}: {sym_check[(pid,0,lv)]} vs {sym_check[(pid,1,lv)]}"
                )

    print(
        f"[INTEGRITY VERIFIED] {len(seen_parents)} disjoint parent families across train_fit(600), train_op(200), "
        f"val_select(30+30), val_calib(30+30), test_id(200), ood_ar(200), ood_llada(150), test_confirm(160)!"
    )
    return raw, val_manifest


def pretokenize_split(rows: list, tok, max_len: int = 384, max_sents: int = 16):
    texts = [r["text"] for r in rows]
    enc = tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc["offset_mapping"].numpy()
    attn = enc["attention_mask"].numpy()
    N, L = attn.shape

    sent_ids = np.full((N, L), -1, dtype=np.int64)
    sent_novelty = np.zeros((N, max_sents), dtype=np.float32)

    for i, txt in enumerate(texts):
        valid_pos = np.where((attn[i] > 0) & (offsets[i, :, 1] > offsets[i, :, 0]))[0]
        if len(valid_pos) == 0:
            continue
        retained_char_end = int(offsets[i, valid_pos[-1], 1])
        prefix_txt = txt[:retained_char_end]

        raw_spans = []
        for m in SENT_SPLIT_RE.finditer(prefix_txt):
            seg = m.group(0)
            if seg.strip():
                raw_spans.append((m.start(), m.end(), seg))
        if not raw_spans:
            raw_spans = [(0, len(prefix_txt), prefix_txt)]

        if len(raw_spans) <= max_sents:
            merged_spans = raw_spans
        else:
            head = raw_spans[:max_sents - 1]
            tail_start = raw_spans[max_sents - 1][0]
            tail_end = raw_spans[-1][1]
            merged_spans = head + [(tail_start, tail_end, prefix_txt[tail_start:tail_end])]

        seen_cw = set()
        for s_idx, (_, _, seg_txt) in enumerate(merged_spans):
            cws = {w.lower() for w in WORD_RE.findall(seg_txt) if w.lower() not in STOPWORDS}
            if cws:
                nov = len(cws - seen_cw) / float(len(cws))
                seen_cw.update(cws)
            else:
                nov = 0.0
            sent_novelty[i, s_idx] = nov

        ends = [sp[1] for sp in merged_spans]
        for t in valid_pos:
            c0, c1 = offsets[i, t]
            mid = (c0 + c1) // 2
            s_bucket = min(int(np.searchsorted(ends, mid, side="right")), len(merged_spans) - 1)
            sent_ids[i, t] = s_bucket

    edit_map = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 3}
    origin_y = torch.tensor([int(r["origin_label"]) for r in rows], dtype=torch.long)
    edit_lv = torch.tensor([edit_map.get(r.get("edit_level", "E0"), 0) for r in rows], dtype=torch.long)
    edit_reg = torch.tensor([int(r.get("edit_register", 1 if edit_map.get(r.get("edit_level", "E0"), 0) > 0 else 0)) for r in rows], dtype=torch.long)
    lex_dist = torch.tensor([float(r.get("measured_lex_dist", 0.0)) for r in rows], dtype=torch.float32)
    nom_str = torch.tensor([min(1.0, max(0.0, float(r.get("nominal_strength", 0.0)))) for r in rows], dtype=torch.float32)

    p_to_int = {}
    p_ints = []
    for i, r in enumerate(rows):
        pid = r.get("parent_id", r.get("prompt_id", f"ind_{i}"))
        if pid not in p_to_int:
            p_to_int[pid] = len(p_to_int)
        p_ints.append(p_to_int[pid])

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "sent_ids": torch.from_numpy(sent_ids),
        "sent_novelty": torch.from_numpy(sent_novelty),
        "origin_y": origin_y,
        "edit_lv": edit_lv,
        "edit_reg": edit_reg,
        "lex_dist": lex_dist,
        "nom_str": nom_str,
        "parent_idx": torch.tensor(p_ints, dtype=torch.long),
        "row_ids": [r["id"] for r in rows],
        "parent_ids": [r.get("parent_id", r.get("prompt_id", r["id"])) for r in rows],
    }


def compute_paired_alignment_loss_v3(
    z_orig: torch.Tensor,
    parent_idx: torch.Tensor,
    origin_y: torch.Tensor,
    edit_lv: torch.Tensor,
    lex_dist: torch.Tensor,
    use_retention_weight: bool = True,
):
    """
    Normalized retention-weighted (or unweighted) paired alignment on unit-normalized z_orig.
    Normalizes by sum(weights) so `use_retention_weight=True` vs `False` holds total loss scale identical!
    """
    z_norm = eps_unit_norm(z_orig, eps=1e-6)
    sq_dists = []
    weights = []
    for p in torch.unique(parent_idx):
        pm = (parent_idx == p)
        for o_val in (0, 1):
            pom = pm & (origin_y == o_val)
            i0 = torch.nonzero(pom & (edit_lv == 0), as_tuple=False).flatten()
            if i0.numel() == 0:
                continue
            z0 = z_norm[i0[0]]
            others = torch.nonzero(pom & (edit_lv > 0), as_tuple=False).flatten()
            for idx in others:
                w = (1.0 - lex_dist[idx]).clamp(min=0.25, max=1.0) if use_retention_weight else torch.tensor(1.0, device=z_orig.device)
                sq_dists.append(torch.sum((z_norm[idx] - z0) ** 2))
                weights.append(w)
    if not sq_dists:
        return torch.tensor(0.0, device=z_orig.device)
    w_vec = torch.stack(weights)
    d_vec = torch.stack(sq_dists)
    return torch.sum(w_vec * d_vec) / w_vec.sum().clamp(min=1e-8)


def compute_within_parent_xmargin_loss(
    origin_logits: torch.Tensor,
    parent_idx: torch.Tensor,
    origin_y: torch.Tensor,
    edit_lv: torch.Tensor,
    margin: float = 2.0,
):
    """
    Exact 16-pair within-parent cross-editorial origin margin loss:
      L_xmargin = (1 / |B_P|) sum_{i in B_P} (1 / 16) sum_{e_h=0}^3 sum_{e_m=0}^3 [m - (ell_{i,1,e_m} - ell_{i,0,e_h})]_+.
    Directly forces LLM-polished human (O=0, E>0) below humanized/edited machine (O=1, E>0) on the same topic W_i.
    """
    parent_losses = []
    for p in torch.unique(parent_idx):
        pm = (parent_idx == p)
        h_idx = torch.nonzero(pm & (origin_y == 0) & (edit_lv <= 3), as_tuple=False).flatten()
        m_idx = torch.nonzero(pm & (origin_y == 1) & (edit_lv <= 3), as_tuple=False).flatten()
        if h_idx.numel() == 4 and m_idx.numel() == 4:
            l_h = origin_logits[h_idx]  # (4,)
            l_m = origin_logits[m_idx]  # (4,)
            diffs = l_m.unsqueeze(1) - l_h.unsqueeze(0)  # (4, 4), 16 pairs
            parent_losses.append(F.relu(margin - diffs).mean())
    if not parent_losses:
        return torch.tensor(0.0, device=origin_logits.device)
    return torch.stack(parent_losses).mean()


def compute_supcon_loss(h: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1):
    z = eps_unit_norm(h, eps=1e-6)
    sim = (z @ z.T) / temperature
    B = z.shape[0]
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)
    has_pos = pos_mask.any(dim=-1)
    if not has_pos.any():
        return torch.tensor(0.0, device=h.device)
    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    pos_cnt = pos_mask.float().sum(dim=-1).clamp(min=1.0)
    per_row = -(log_prob * pos_mask.float()).sum(dim=-1) / pos_cnt
    return per_row[has_pos].mean()


@torch.inference_mode()
def evaluate_split(model: nn.Module, tensor_dict: dict, batch_size: int = 256):
    model.eval()
    N = tensor_dict["input_ids"].shape[0]
    probs_all, intens_all = [], []
    for st in range(0, N, batch_size):
        ed = min(st + batch_size, N)
        out = model(
            input_ids=tensor_dict["input_ids"][st:ed].to("cuda"),
            attention_mask=tensor_dict["attention_mask"][st:ed].to("cuda"),
            sent_ids=tensor_dict["sent_ids"][st:ed].to("cuda"),
            sent_novelty=tensor_dict["sent_novelty"][st:ed].to("cuda"),
        )
        probs = torch.sigmoid(out["origin_logit"]).double().cpu().tolist()
        intens = out["edit_intensity"].double().cpu().tolist()
        probs_all.extend(probs)
        intens_all.extend(intens)
    return probs_all, intens_all


def selection_worst_edit_bal_acc(probs: list, sel_tensor: dict):
    p = np.asarray(probs, dtype=np.float64)
    if not np.isfinite(p).all():
        return (-1.0, -1.0), 0.0, 0.0
    y = sel_tensor["origin_y"].numpy()
    e = sel_tensor["edit_lv"].numpy()
    hum_pids = [sel_tensor["parent_ids"][i] for i in np.where(y == 0)[0]]

    tau_sel = cluster_conformal_upper_threshold(p[y == 0], hum_pids, alpha=0.05)
    preds = (p > tau_sel).astype(np.int64)

    edit_bal_accs = []
    for ev in (0, 1, 2, 3):
        m0 = (e == ev) & (y == 0)
        m1 = (e == ev) & (y == 1)
        tnr_e = float((preds[m0] == 0).mean())
        tpr_e = float((preds[m1] == 1).mean())
        edit_bal_accs.append(0.5 * (tnr_e + tpr_e))

    worst_ebal = float(min(edit_bal_accs))
    mean_ebal = float(np.mean(edit_bal_accs))
    return (round(worst_ebal, 6), round(mean_ebal, 6)), worst_ebal, mean_ebal


EVAL_SPLIT_KEYS = [
    "val_select",
    "val_calib",
    "test_id",
    "test_ood_ar",
    "test_ood_llada",
    "test_confirm",
    "external_esl_full",
    "external_mixset",
    "external_beemo",
    "external_mage_wild",
]


def train_and_eval_job(
    job: dict,
    backbone_name: str,
    backbone_sha: str,
    tokenized_splits: dict,
    data_fp: str,
    lr_backbone: float,
    lr_head: float,
    epochs: int = 4,
):
    run_name = job["run_name"]
    seed = job["seed"]
    mode = job["mode"]
    train_key = job["train_key"]
    op_key = job.get("op_key", None)
    nsd_rank_override = job.get("nsd_rank", None)
    save_ckpt = bool(job.get("save_ckpt", False))

    v1_src = open(os.path.join(_CODE_ROOT, "src", "cedar_model.py"), "rb").read()
    v2_src = open(os.path.join(_CODE_ROOT, "src", "cedar_v2_model.py"), "rb").read()
    script_path = globals().get("__file__", os.path.join(_CODE_ROOT, "scripts", "v3_train_cedar_v2_suite.py"))
    sc_src = open(script_path, "rb").read() if os.path.exists(script_path) else b""
    src_hash = hashlib.sha256(v1_src + v2_src + sc_src).hexdigest()[:16]
    cfg_sig = hashlib.sha256(
        f"{CODE_VERSION}|{src_hash}|{data_fp}|{run_name}|{seed}|{backbone_name}|{backbone_sha}|{mode}|{train_key}|{op_key}|{nsd_rank_override}|{epochs}|{lr_backbone}|{lr_head}".encode()
    ).hexdigest()[:20]

    out_file = os.path.join(PRED_DIR, f"{run_name}_seed{seed}.json")
    ckpt_file = os.path.join(CKPT_DIR, f"{run_name}_seed{seed}.pt")
    if os.path.exists(out_file) and (not save_ckpt or os.path.exists(ckpt_file)):
        old = json.load(open(out_file))
        if old.get("data_fingerprint") == data_fp:
            print(f"  [SKIP VERIFIED CACHE] {run_name}_seed{seed}", flush=True)
            return

    t_fit_start = time.time()
    set_seed(seed)

    is_legacy_v1_model = (mode == "legacy_v1_cedar")
    if is_legacy_v1_model:
        model = CEDARDetector(
            backbone_name=backbone_name,
            proj_dim=512,
            use_dte=True,
            use_param_matched_mlp=False,
            nsd_rank=32,
            nsd_signal_alpha=0.5,
            joint_8class=False,
        ).to("cuda")
        model_kwargs = {"legacy_v1": True, "backbone_name": backbone_name, "proj_dim": 512, "nsd_rank": 32}
        nsd_rank = 32
        exact_finalize = False
    else:
        cedar_modes = {
            "cedar_split_op",
            "cedar_full800",
            "cedar_ema_only",
            "cedar_unweighted_pair",
            "cedar_no_xmargin",
            "cedar_no_dte",
            "cedar_param_matched_mlp",
            "cedar_unprotected_nsd",
        }
        combined_modes = {"strong_combined_no_nsd", "strong_combined_fit600_no_nsd",
                          "strong_combined_nodte_no_nsd", "strong_combined_nodte_fit600_no_nsd"}
        e2_gap_modes = {"spot_gap_pd_split", "spot_gap_pd_ema"}
        cedar_modes = cedar_modes | e2_gap_modes

        use_dte = (mode in (cedar_modes | combined_modes)) and (mode not in {"cedar_no_dte", "cedar_param_matched_mlp",
                                                                                  "strong_combined_nodte_no_nsd", "strong_combined_nodte_fit600_no_nsd"}
                                                                     | e2_gap_modes)
        use_pm_mlp = (mode == "cedar_param_matched_mlp")
        default_rk = (8 if mode in e2_gap_modes else 24) if (mode in cedar_modes) else 0
        nsd_rank = int(nsd_rank_override) if (nsd_rank_override is not None) else default_rk
        nsd_alpha = 0.0 if (mode == "cedar_unprotected_nsd") else 0.5
        joint_8 = (mode == "joint_8class_ce")
        use_cos = mode in (cedar_modes | combined_modes)
        exact_finalize = (nsd_rank > 0) and (mode not in {"cedar_ema_only", "spot_gap_pd_ema"}) and (op_key is not None)

        model_kwargs = {
            "backbone_name": backbone_name,
            "backbone_revision": backbone_sha,
            "proj_dim": 128,
            "use_dte": use_dte,
            "use_param_matched_mlp": use_pm_mlp,
            "nsd_rank": nsd_rank,
            "nsd_signal_alpha": nsd_alpha,
            "nsd_tau": 1e-4,
            "nsd_ema_beta": 0.08,
            "cosine_kappa": 12.0,
            "use_cosine_head": use_cos,
            "joint_8class": joint_8,
        }
        if mode in e2_gap_modes:
            model_kwargs["nsd_rank_mode"] = "gap"
        model = CEDARv2Detector(**model_kwargs).to("cuda")

    joint_8 = (mode == "joint_8class_ce")
    use_gdro = is_legacy_v1_model or (mode in {
        "erm_group_dro", "spot_gap_pd_split", "spot_gap_pd_ema", "strong_combined_no_nsd", "strong_combined_fit600_no_nsd",
        "strong_combined_nodte_no_nsd", "strong_combined_nodte_fit600_no_nsd",
        "cedar_split_op", "cedar_full800", "cedar_ema_only", "cedar_unweighted_pair",
        "cedar_no_xmargin", "cedar_no_dte", "cedar_param_matched_mlp", "cedar_unprotected_nsd",
    })
    use_pair = is_legacy_v1_model or (mode in {
        "erm_paired_consist", "spot_gap_pd_split", "spot_gap_pd_ema", "supcon_orth_pair_xm", "strong_combined_no_nsd", "strong_combined_fit600_no_nsd",
        "strong_combined_nodte_no_nsd", "strong_combined_nodte_fit600_no_nsd",
        "cedar_split_op", "cedar_full800", "cedar_ema_only", "cedar_unweighted_pair",
        "cedar_no_xmargin", "cedar_no_dte", "cedar_param_matched_mlp", "cedar_unprotected_nsd",
    })
    use_xmargin = (not is_legacy_v1_model) and (mode in {
        "spot_gap_pd_split", "spot_gap_pd_ema", "supcon_orth_pair_xm", "strong_combined_no_nsd", "strong_combined_fit600_no_nsd",
        "strong_combined_nodte_no_nsd", "strong_combined_nodte_fit600_no_nsd",
        "cedar_split_op", "cedar_full800", "cedar_ema_only", "cedar_unweighted_pair",
        "cedar_no_dte", "cedar_param_matched_mlp", "cedar_unprotected_nsd",
    })
    use_ret_weight = (mode != "cedar_unweighted_pair")
    use_aux_route = (not is_legacy_v1_model) and (nsd_rank > 0)

    bb_params = list(model.backbone.parameters())
    bb_ids = {id(p) for p in bb_params}
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]
    probe_params = [] if joint_8 else list(model.edit_intensity_head.parameters())
    probe_ids = {id(p) for p in probe_params}
    main_params = [p for p in model.parameters() if id(p) not in probe_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": lr_backbone, "weight_decay": 0.01},
            {"params": head_params, "lr": lr_head, "weight_decay": 0.01},
        ]
    )

    tr = tokenized_splits[train_key]
    parent_to_indices = defaultdict(list)
    for idx, pid in enumerate(tr["parent_idx"].tolist()):
        parent_to_indices[pid].append(idx)
    unique_parents = sorted(parent_to_indices.keys())
    steps_per_full_epoch = math.ceil(len(unique_parents) / 8)
    target_total_steps = epochs * steps_per_full_epoch

    all_epoch_batches = []
    if mode == "unedited_erm":
        e0_idx = torch.nonzero(tr["edit_lv"] == 0, as_tuple=False).flatten().tolist()
        steps_collected = 0
        while steps_collected < target_total_steps:
            perm = e0_idx.copy()
            random.shuffle(perm)
            ep_b = []
            for st in range(0, len(perm), 64):
                if steps_collected >= target_total_steps:
                    break
                ep_b.append(perm[st:st + 64])
                steps_collected += 1
            if ep_b:
                all_epoch_batches.append(ep_b)
    else:
        for _ in range(epochs):
            p_perm = unique_parents.copy()
            random.shuffle(p_perm)
            ep_b = []
            for st in range(0, len(p_perm), 8):
                b_idx = []
                for pid in p_perm[st:st + 8]:
                    b_idx.extend(parent_to_indices[pid])
                ep_b.append(b_idx)
            all_epoch_batches.append(ep_b)

    exact_total_steps = sum(len(ep_b) for ep_b in all_epoch_batches)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exact_total_steps, eta_min=1e-6)

    log_q = torch.zeros(8, device="cuda", dtype=torch.float32) - math.log(8.0)
    gdro_eta = 0.05

    best_tuple = (-1.0, -1.0)
    best_state = None
    best_val_info = {}
    best_nsd_diag = {}
    steps_done = 0

    for ep_i, batches in enumerate(all_epoch_batches):
        model.train()
        for b_idx in batches:
            idx_t = torch.tensor(b_idx, dtype=torch.long)
            in_ids = tr["input_ids"][idx_t].to("cuda")
            attn = tr["attention_mask"][idx_t].to("cuda")
            s_ids = tr["sent_ids"][idx_t].to("cuda")
            s_nov = tr["sent_novelty"][idx_t].to("cuda")
            y_o = tr["origin_y"][idx_t].to("cuda")
            y_e = tr["edit_lv"][idx_t].to("cuda")
            y_r = tr["edit_reg"][idx_t].to("cuda")
            l_dist = tr["lex_dist"][idx_t].to("cuda")
            n_str = tr["nom_str"][idx_t].to("cuda")
            p_id = tr["parent_idx"][idx_t].to("cuda")

            out = model(in_ids, attn, s_ids, s_nov)

            if joint_8:
                loss = F.cross_entropy(out["joint_logits"], y_o * 4 + y_e)
            else:
                per_ex_bce = F.binary_cross_entropy_with_logits(out["origin_logit"], y_o.float(), reduction="none")
                if use_gdro:
                    g_ids = y_o * 4 + y_e
                    g_losses = torch.zeros(8, device="cuda", dtype=torch.float32)
                    g_present = torch.zeros(8, dtype=torch.bool, device="cuda")
                    for g in range(8):
                        gm = (g_ids == g)
                        if gm.any():
                            g_losses[g] = per_ex_bce[gm].mean()
                            g_present[g] = True
                    with torch.no_grad():
                        delta_lq = torch.where(g_present, gdro_eta * g_losses.detach(), torch.zeros_like(g_losses))
                        log_q = F.log_softmax(log_q + delta_lq, dim=0)
                    q_probs = log_q.exp() * g_present.float()
                    q_probs = q_probs / q_probs.sum().clamp(min=1e-8)
                    cls_loss = (q_probs * g_losses).sum()
                else:
                    cls_loss = per_ex_bce.mean()

                loss = cls_loss

                if mode == "multitask_2head":
                    loss = loss + 0.25 * F.cross_entropy(out["edit_logits"], y_e) + 0.15 * F.cross_entropy(out["reg_logits"], y_r)
                elif mode in ("supcon_orth_disentangle", "supcon_orth_pair_xm"):
                    supcon = compute_supcon_loss(out["h_hat"], y_o, temperature=0.1)
                    w_o = eps_unit_norm(model.origin_head.weight, eps=1e-6)
                    if w_o.dim() == 1:
                        w_o = w_o.unsqueeze(0)
                    w_e = eps_unit_norm(model.edit_cls_head.weight, eps=1e-6)
                    orth_pen = ((w_o @ w_e.T) ** 2).mean()
                    loss = loss + 0.35 * supcon + 0.25 * F.cross_entropy(out["edit_logits"], y_e) + 0.5 * orth_pen

                if use_pair:
                    pair_loss = compute_paired_alignment_loss_v3(
                        out["z_orig"], p_id, y_o, y_e, l_dist, use_retention_weight=use_ret_weight
                    )
                    loss = loss + 0.5 * pair_loss

                if use_xmargin:
                    xm_loss = compute_within_parent_xmargin_loss(out["origin_logit"], p_id, y_o, y_e, margin=2.0)
                    loss = loss + 0.35 * xm_loss

                if use_aux_route:
                    route_loss = 0.20 * F.cross_entropy(out["edit_logits"], y_e) + 0.20 * F.cross_entropy(out["reg_logits"], y_r)
                    loss = loss + route_loss

                if is_legacy_v1_model or use_cos:
                    int_loss = F.smooth_l1_loss(out["edit_intensity"], n_str)
                    loss = loss + 0.20 * int_loss

            assert torch.isfinite(loss), f"Non-finite loss in {run_name}_seed{seed} at step {steps_done}"
            h_for_nsd = (out["h"] if is_legacy_v1_model else out["h_hat"]).detach()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            assert torch.isfinite(gn), f"Non-finite main grad norm in {run_name}_seed{seed}"
            if probe_params:
                pgn = torch.nn.utils.clip_grad_norm_(probe_params, 1.0)
                assert torch.isfinite(pgn), f"Non-finite probe grad norm in {run_name}_seed{seed}"
            optimizer.step()
            scheduler.step()
            steps_done += 1

            if nsd_rank > 0:
                if is_legacy_v1_model:
                    model.update_nsd_subspace_post_step(h_for_nsd, p_id, y_o, y_e, warmup_updates=15)
                else:
                    model.update_nsd_subspace_post_step(h_for_nsd, p_id, y_o, y_e, warmup_updates=10)

        eval_due = (mode != "unedited_erm") or ((ep_i + 1) % 4 == 0) or (ep_i == len(all_epoch_batches) - 1)
        if eval_due:
            # CRITICAL: Snapshot online train_fit EMA buffers, install exact float64 operator from `op_key`
            # BEFORE evaluating val_select, save finalized candidate if winning, then restore online buffers!
            if exact_finalize:
                ema_snap = model.snapshot_nsd_buffers()
                cand_diag = model.finalize_exact_nsd_from_pool(tokenized_splits[op_key], batch_size=128)
            else:
                cand_diag = model.get_nsd_diagnostics()

            sel_probs, _ = evaluate_split(model, tokenized_splits["val_select"])
            sel_tuple, w_ebal, m_ebal = selection_worst_edit_bal_acc(sel_probs, tokenized_splits["val_select"])
            if sel_tuple > best_tuple:
                best_tuple = sel_tuple
                best_val_info = {
                    "epoch": ep_i + 1,
                    "step": steps_done,
                    "val_select_worst_edit_bal_acc": round(w_ebal, 5),
                    "val_select_mean_edit_bal_acc": round(m_ebal, 5),
                }
                best_nsd_diag = cand_diag
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if exact_finalize:
                model.restore_nsd_buffers(ema_snap)

    assert best_state is not None, f"No valid checkpoint selected for {run_name}_seed{seed}"
    model.load_state_dict(best_state)
    fit_and_select_seconds = round(time.time() - t_fit_start, 2)

    # Calibration on strictly disjoint source-stratified val_calib
    t_cal_start = time.time()
    cal_probs, _ = evaluate_split(model, tokenized_splits["val_calib"])
    cal_p = np.asarray(cal_probs, dtype=np.float64)
    cal_y = tokenized_splits["val_calib"]["origin_y"].numpy()
    cal_e = tokenized_splits["val_calib"]["edit_lv"].numpy()
    cal_pids = tokenized_splits["val_calib"]["parent_ids"]

    hum_all_idx = np.where(cal_y == 0)[0]
    hum_e0_idx = np.where((cal_y == 0) & (cal_e == 0))[0]
    tau_cal_all5 = cluster_conformal_upper_threshold(cal_p[hum_all_idx], [cal_pids[i] for i in hum_all_idx], alpha=0.05)
    tau_cal_e0_5 = cluster_conformal_upper_threshold(cal_p[hum_e0_idx], [cal_pids[i] for i in hum_e0_idx], alpha=0.05)
    calib_seconds = round(time.time() - t_cal_start, 3)

    # Test & External Inference Timing
    t_eval_start = time.time()
    preds_map = {"val_calib": cal_probs}
    row_ids_map = {"val_calib": tokenized_splits["val_calib"]["row_ids"]}
    intens_map = {}
    for sp_name in EVAL_SPLIT_KEYS + [k for k in E2_SPLITS if k in tokenized_splits]:
        t_dict = tokenized_splits[sp_name]
        pr, ei = evaluate_split(model, t_dict)
        preds_map[sp_name] = pr
        row_ids_map[sp_name] = t_dict["row_ids"]
        if (not joint_8) and (is_legacy_v1_model or ("cedar" in mode) or ("combined" in mode)):
            intens_map[sp_name] = ei
    eval_inference_seconds = round(time.time() - t_eval_start, 2)

    # Persist full finalized checkpoint + verify clean fresh-model reload determinism!
    ckpt_meta = None
    if save_ckpt and (not is_legacy_v1_model):
        ckpt_payload = {
            "run_name": run_name,
            "seed": seed,
            "code_version": CODE_VERSION,
            "backbone_commit_sha": backbone_sha,
            "model_kwargs": model_kwargs,
            "best_val_select": best_val_info,
            "nsd_diagnostics": best_nsd_diag,
            "locked_thresholds": {"tau_cal_all5": tau_cal_all5, "tau_cal_e0_5": tau_cal_e0_5},
            "model_state_dict": best_state,
        }
        torch.save(ckpt_payload, ckpt_file)
        # Clean reload verification on a brand-new model instance
        fresh_model = CEDARv2Detector(**model_kwargs).to("cuda")
        fresh_model.load_state_dict(torch.load(ckpt_file, map_location="cpu")["model_state_dict"], strict=True)
        re_probs, _ = evaluate_split(fresh_model, tokenized_splits["test_confirm"])
        max_diff = float(np.max(np.abs(np.asarray(re_probs) - np.asarray(preds_map["test_confirm"]))))
        assert max_diff < 1e-3, f"Checkpoint reload mismatch {max_diff} in {run_name}_seed{seed}"
        ckpt_sha = hashlib.sha256(open(ckpt_file, "rb").read()).hexdigest()
        ckpt_meta = {
            "checkpoint_filename": os.path.basename(ckpt_file),
            "sha256": ckpt_sha,
            "size_bytes": os.path.getsize(ckpt_file),
            "reload_test_confirm_max_abs_diff": max_diff,
        }
        del fresh_model

    eval_payload = {
        "config_hash": cfg_sig,
        "data_fingerprint": data_fp,
        "code_version": CODE_VERSION,
        "run_name": run_name,
        "seed": seed,
        "backbone": backbone_name,
        "backbone_commit_sha": backbone_sha,
        "mode": mode,
        "train_split": train_key,
        "operator_split": op_key,
        "exact_optimizer_steps": steps_done,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_val_select": best_val_info,
        "nsd_diagnostics": best_nsd_diag,
        "locked_thresholds": {
            "tau_cal_all5": tau_cal_all5,
            "tau_cal_e0_5": tau_cal_e0_5,
            "realized_cal_all_fpr": float((cal_p[hum_all_idx] > tau_cal_all5).mean()),
            "realized_cal_e0_fpr": float((cal_p[hum_e0_idx] > tau_cal_e0_5).mean()),
        },
        "stage_timings": {
            "fit_and_select_seconds": fit_and_select_seconds,
            "calib_seconds": calib_seconds,
            "eval_inference_seconds": eval_inference_seconds,
            "total_seconds": round(fit_and_select_seconds + calib_seconds + eval_inference_seconds, 2),
        },
        "checkpoint_archive": ckpt_meta,
        "row_ids": row_ids_map,
        "preds": preds_map,
        "edit_intensity": intens_map,
    }

    tmp_out = out_file + ".tmp"
    with open(tmp_out, "w") as f:
        json.dump(eval_payload, f)
    os.replace(tmp_out, out_file)

    print(
        f"  [DONE] {run_name}_seed{seed} ({eval_payload['stage_timings']['total_seconds']}s) | "
        f"val_sel_w_ebal={best_val_info['val_select_worst_edit_bal_acc']:.4f} | nsd={best_nsd_diag.get('frac_edit_energy_removed','-')}",
        flush=True,
    )
    del model, optimizer, best_state
    gc.collect()
    torch.cuda.empty_cache()


def build_job_queues():
    seeds5 = [42, 1042, 2042, 3042, 4042]
    seeds3 = [42, 1042, 2042]

    mb_jobs = []
    # 1. Core 5-Seed Comparison on 800-Parent Bidirectional v2 Benchmark
    core_mb_specs = [
        ("mb_v3_unedited_erm", "unedited_erm", "train_all_v2", None, False),
        ("mb_v3_augmented_erm", "augmented_erm", "train_all_v2", None, False),
        ("mb_v3_joint_8class_ce", "joint_8class_ce", "train_all_v2", None, False),
        ("mb_v3_multitask_2head", "multitask_2head", "train_all_v2", None, False),
        ("mb_v3_erm_paired_consist", "erm_paired_consist", "train_all_v2", None, False),
        ("mb_v3_erm_group_dro", "erm_group_dro", "train_all_v2", None, False),
        ("mb_v3_supcon_orth_disentangle", "supcon_orth_disentangle", "train_all_v2", None, False),
        ("mb_v3_strong_combined_no_nsd", "strong_combined_no_nsd", "train_all_v2", None, True),
        ("mb_v3_cedar_split_op", "cedar_split_op", "train_fit_v2", "train_op_v2", True),
        ("mb_v3_cedar_full800", "cedar_full800", "train_all_v2", "train_all_v2", False),
    ]
    for r_name, mode, tr_k, op_k, ckpt_s42 in core_mb_specs:
        for sd in seeds5:
            mb_jobs.append({
                "run_name": r_name,
                "seed": sd,
                "mode": mode,
                "train_key": tr_k,
                "op_key": op_k,
                "save_ckpt": bool(ckpt_s42 and sd == 42),
            })

    # 2. Complete 3-Seed Component & Rank Ablations
    abl_specs = [
        ("mb_v3_ablation_ema_only", "cedar_ema_only", "train_fit_v2", None, None),
        ("mb_v3_ablation_unweighted_pair", "cedar_unweighted_pair", "train_fit_v2", "train_op_v2", None),
        ("mb_v3_ablation_no_xmargin", "cedar_no_xmargin", "train_fit_v2", "train_op_v2", None),
        ("mb_v3_ablation_no_dte", "cedar_no_dte", "train_fit_v2", "train_op_v2", None),
        ("mb_v3_ablation_param_matched_mlp", "cedar_param_matched_mlp", "train_fit_v2", "train_op_v2", None),
        ("mb_v3_ablation_unprotected_nsd", "cedar_unprotected_nsd", "train_fit_v2", "train_op_v2", None),
        ("mb_v3_strong_combined_fit600_no_nsd", "strong_combined_fit600_no_nsd", "train_fit_v2", None, None),
        ("mb_v3_cedar_rank8", "cedar_split_op", "train_fit_v2", "train_op_v2", 8),
        ("mb_v3_cedar_rank48", "cedar_split_op", "train_fit_v2", "train_op_v2", 48),
    ]
    for r_name, mode, tr_k, op_k, rk in abl_specs:
        for sd in seeds3:
            mb_jobs.append({
                "run_name": r_name,
                "seed": sd,
                "mode": mode,
                "train_key": tr_k,
                "op_key": op_k,
                "nsd_rank": rk,
                "save_ckpt": False,
            })

    # 3. 2x2 Old-vs-New Data x Old-vs-New Recipe Attribution Bridge (on v1 unidirectional 500-parent train, stratified val)
    bridge_specs = [
        ("bridge_v1data_aug_erm", "augmented_erm", "train_v1", None),
        ("bridge_v1data_v1cedar", "legacy_v1_cedar", "train_v1", None),
        ("bridge_v1data_v2cedar", "cedar_split_op", "train_v1_fit", "train_v1_op"),
    ]
    for r_name, mode, tr_k, op_k in bridge_specs:
        for sd in seeds3:
            mb_jobs.append({
                "run_name": r_name,
                "seed": sd,
                "mode": mode,
                "train_key": tr_k,
                "op_key": op_k,
                "save_ckpt": False,
            })

    # 4. DeBERTa-v3-large 5-Seed Generality Suite (6 core methods x 5 seeds = 30 runs)
    db_jobs = []
    db_specs = [
        ("db_v3_unedited_erm", "unedited_erm", "train_all_v2", None, False),
        ("db_v3_augmented_erm", "augmented_erm", "train_all_v2", None, False),
        ("db_v3_erm_paired_consist", "erm_paired_consist", "train_all_v2", None, False),
        ("db_v3_supcon_orth_disentangle", "supcon_orth_disentangle", "train_all_v2", None, False),
        ("db_v3_strong_combined_no_nsd", "strong_combined_no_nsd", "train_all_v2", None, False),
        ("db_v3_cedar_split_op", "cedar_split_op", "train_fit_v2", "train_op_v2", True),
    ]
    for r_name, mode, tr_k, op_k, ckpt_s42 in db_specs:
        for sd in seeds5:
            db_jobs.append({
                "run_name": r_name,
                "seed": sd,
                "mode": mode,
                "train_key": tr_k,
                "op_key": op_k,
                "save_ckpt": bool(ckpt_s42 and sd == 42),
            })

    return mb_jobs, db_jobs


def build_extension_jobs_e1():
    """E1 (2026-11-14, pre-registered in docs/v3_results_2026-11-14.md): complete the DTE x NSD factorial.
    Selection of the no-DTE CEDAR variant was made on val_select (worst-edit balanced accuracy), not on test sets."""
    seeds5 = [42, 1042, 2042, 3042, 4042]
    mb, db = [], []
    for sd in [3042, 4042]:
        mb.append({"run_name": "mb_v3_ablation_no_dte", "seed": sd, "mode": "cedar_no_dte",
                   "train_key": "train_fit_v2", "op_key": "train_op_v2", "nsd_rank": None, "save_ckpt": False})
    for sd in seeds5:
        mb.append({"run_name": "mb_v3_nodte_fit600_no_nsd", "seed": sd, "mode": "strong_combined_nodte_fit600_no_nsd",
                   "train_key": "train_fit_v2", "op_key": None, "nsd_rank": None, "save_ckpt": False})
        mb.append({"run_name": "mb_v3_nodte_no_nsd", "seed": sd, "mode": "strong_combined_nodte_no_nsd",
                   "train_key": "train_all_v2", "op_key": None, "nsd_rank": None, "save_ckpt": False})
        db.append({"run_name": "db_v3_cedar_no_dte", "seed": sd, "mode": "cedar_no_dte",
                   "train_key": "train_fit_v2", "op_key": "train_op_v2", "nsd_rank": None, "save_ckpt": False})
        db.append({"run_name": "db_v3_nodte_no_nsd", "seed": sd, "mode": "strong_combined_nodte_no_nsd",
                   "train_key": "train_all_v2", "op_key": None, "nsd_rank": None, "save_ckpt": False})
    return mb, db


def build_extension_jobs_e2():
    """E2 (pre-registered in docs/E2_plan_2026-11-15.md). Names start with e2_ because these runs also score
    val_shift and test_confirm2. Phase B uses seeds 42/1042 on ModernBERT; phase C uses all 5 seeds on both backbones."""
    cfg = {  # name -> (mode, train_key, op_key)
        "spot": ("strong_combined_no_nsd", "train_all_v2", None),
        "supcon": ("supcon_orth_disentangle", "train_all_v2", None),
        "aug": ("augmented_erm", "train_all_v2", None),
        "unedited": ("unedited_erm", "train_all_v2", None),
        "joint8": ("joint_8class_ce", "train_all_v2", None),
        "multitask": ("multitask_2head", "train_all_v2", None),
        "gdro": ("erm_group_dro", "train_all_v2", None),
        "pair": ("erm_paired_consist", "train_all_v2", None),
        "gappd_split": ("spot_gap_pd_split", "train_fit_v2", "train_op_v2"),
        "gappd_ema": ("spot_gap_pd_ema", "train_all_v2", None),
        "supcon_pairxm": ("supcon_orth_pair_xm", "train_all_v2", None),
        # attribution controls for the chosen C1 (added before phase C, see E2 plan Outcome 03:25 SGT)
        "spot_nodte": ("strong_combined_nodte_no_nsd", "train_all_v2", None),
        "spot_nodte600": ("strong_combined_nodte_fit600_no_nsd", "train_fit_v2", None),
    }
    mb, db = [], []
    for bb, out in (("mb", mb), ("db", db)):
        for name, (mode, tk, ok) in cfg.items():
            for sd in [42, 1042, 2042, 3042, 4042]:
                out.append({"run_name": f"e2_{bb}_{name}", "seed": sd, "mode": mode, "train_key": tk, "op_key": ok,
                            "nsd_rank": None, "save_ckpt": False})
    return mb, db


def main():
    worker_id = int(os.environ.get("CEDAR_WORKER_ID", "0"))
    num_workers = int(os.environ.get("CEDAR_NUM_WORKERS", "2"))
    print(f"=== v3_train_cedar_v2_suite | worker={worker_id}/{num_workers} ===", flush=True)

    prov = json.load(open(os.path.join(V2_DIR, "provenance_lock.json")))
    m_shas = prov["hf_model_commit_shas"]
    raw_splits, val_manifest = load_and_verify_all_splits()
    data_fp = compute_full_split_fingerprint(raw_splits, val_manifest)
    mb_jobs_all, db_jobs_all = build_job_queues()

    my_mb_jobs = [jb for i, jb in enumerate(mb_jobs_all) if (i % num_workers) == worker_id]
    my_db_jobs = [jb for i, jb in enumerate(db_jobs_all) if (i % num_workers) == worker_id]
    print(f"  Assigned {len(my_mb_jobs)} ModernBERT runs + {len(my_db_jobs)} DeBERTa runs (data_fp={data_fp})", flush=True)

    # Pre-warm exact pinned HF snapshots once, then lock offline mode so 58 runs make ZERO HTTP calls!
    mb_id = "answerdotai/ModernBERT-large"
    mb_sha = m_shas[mb_id]
    db_id = "microsoft/deberta-v3-large"
    db_sha = m_shas[db_id]
    snapshot_download(repo_id=mb_id)
    snapshot_download(repo_id=mb_id, revision=mb_sha)
    snapshot_download(repo_id=db_id)
    snapshot_download(repo_id=db_id, revision=db_sha)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Suite 1: ModernBERT-large
    print(f"\n>>> Pre-tokenizing splits for {mb_id} @ {mb_sha[:10]} ...", flush=True)
    mb_tok = AutoTokenizer.from_pretrained(mb_id, revision=mb_sha, local_files_only=True)
    mb_tok_splits = {k: pretokenize_split(v, mb_tok, max_len=384) for k, v in raw_splits.items()}

    for jb in my_mb_jobs:
        train_and_eval_job(jb, mb_id, mb_sha, mb_tok_splits, data_fp, lr_backbone=2.5e-5, lr_head=2.5e-4, epochs=4)

    del mb_tok_splits, mb_tok
    gc.collect()
    torch.cuda.empty_cache()

    # Suite 2: DeBERTa-v3-large
    print(f"\n>>> Pre-tokenizing splits for {db_id} @ {db_sha[:10]} ...", flush=True)
    db_tok = AutoTokenizer.from_pretrained(db_id, revision=db_sha, local_files_only=True)
    db_tok_splits = {k: pretokenize_split(v, db_tok, max_len=384) for k, v in raw_splits.items()}

    for jb in my_db_jobs:
        train_and_eval_job(jb, db_id, db_sha, db_tok_splits, data_fp, lr_backbone=1.5e-5, lr_head=2.0e-4, epochs=4)

    print(f"\n=== WORKER {worker_id}/{num_workers} COMPLETED ALL ASSIGNED V3 TRAINING RUNS! ===", flush=True)


if __name__ == "__main__":
    main()

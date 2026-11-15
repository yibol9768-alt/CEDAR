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
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, "/content")
from src.cedar_model import CEDARDetector

DATA_DIR = "/content/cedar_data"
PRED_DIR = os.path.join(DATA_DIR, "model_preds_v2")
os.makedirs(PRED_DIR, exist_ok=True)

CODE_VERSION = "cedar_v2_codex_audited"

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


def cluster_conformal_upper_threshold(hum_scores: np.ndarray, hum_pids: list, alpha: float = 0.05) -> float:
    """
    Parent-cluster split-conformal risk control threshold for strict rule `score > tau`.
    Given K exchangeable human parent clusters i=1..K with within-cluster empirical FPR
    L_i(tau) = (1/m_i) sum_{e=1}^{m_i} 1[S_{i,e} > tau], returns
      inf { tau in unique(S) U {+inf} : (1 + sum_{i=1}^K L_i(tau)) / (K + 1) <= alpha }.
    Exact under arbitrary within-parent correlation and arbitrary score ties; reduces identically
    to standard split-conformal order statistic when m_i = 1 (e.g., E0-only calibration).
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


def validate_and_partition_splits(raw_splits: dict):
    """
    1. Partition split_val deterministically by parent_id into disjoint val_select (60 parents) and val_calib (60 parents).
    2. Strictly assert ZERO parent_id overlap, exact (origin, edit) multiset cells (no duplicates), and
       exact symmetric editor assignment across (O=0, O=1) for every controlled parent and edit level.
    """
    val_rows = raw_splits["val"]
    val_pids = sorted({r["parent_id"] for r in val_rows})
    assert len(val_pids) == 120, f"Expected 120 val parents, got {len(val_pids)}"
    sel_set = set(val_pids[:60])
    cal_set = set(val_pids[60:])

    raw_splits["val_select"] = [r for r in val_rows if r["parent_id"] in sel_set]
    raw_splits["val_calib"] = [r for r in val_rows if r["parent_id"] in cal_set]

    expected_lvs = {
        "train": ["E0", "E1", "E2", "E3"],
        "val_select": ["E0", "E1", "E2", "E3"],
        "val_calib": ["E0", "E1", "E2", "E3"],
        "test_id": ["E0", "E1", "E2", "E3", "E4"],
        "test_ood_ar": ["E0", "E1", "E2"],
        "test_ood_llada": ["E0", "E1", "E2"],
    }

    seen_parents = {}
    for k, lvs in expected_lvs.items():
        pids = {r["parent_id"] for r in raw_splits[k]}
        for pid in pids:
            assert pid not in seen_parents, f"LEAKAGE DETECTED: parent_id {pid} in both {seen_parents[pid]} and {k}"
            seen_parents[pid] = k

        target_counter = Counter((o, lv) for o in (0, 1) for lv in lvs)
        by_p = defaultdict(list)
        ed_check = {}
        seen_ids = set()
        for r in raw_splits[k]:
            assert r["id"] not in seen_ids, f"Duplicate row id {r['id']} in {k}"
            seen_ids.add(r["id"])
            assert len(r["text"].strip()) > 0, f"Empty text in {k}: {r['id']}"
            assert 0.0 <= float(r["nominal_strength"]) <= 1.0, f"Invalid nominal_strength in {r['id']}"
            o_int = int(r["origin_label"])
            lv = r["edit_level"]
            pid = r["parent_id"]
            by_p[pid].append((o_int, lv))
            ed_check[(pid, o_int, lv)] = r.get("editor_model", "none")

        for pid, cell_list in by_p.items():
            assert Counter(cell_list) == target_counter, f"Cell multiset mismatch for {pid} in {k}: {Counter(cell_list)}"
            for lv in lvs:
                assert ed_check[(pid, 0, lv)] == ed_check[(pid, 1, lv)], (
                    f"Asymmetric editor on {pid} {lv} in {k}: {ed_check[(pid,0,lv)]} vs {ed_check[(pid,1,lv)]}"
                )

    print(
        f"  [INTEGRITY VERIFIED] Zero parent overlap & exact symmetric multisets across {len(seen_parents)} parent families! "
        f"val_select={len(raw_splits['val_select'])}, val_calib={len(raw_splits['val_calib'])}"
    )
    return raw_splits


def pretokenize_split(rows: list, tok, max_len: int = 384, max_sents: int = 16):
    """
    Computes sentence spans and content-word novelty STRICTLY within the retained non-truncated
    character span seen by the backbone, and merges overflow sentences (>= max_sents-1) BEFORE
    computing novelty so token span and novelty match 100%.
    """
    texts = [r["text"] if len(r["text"].strip()) > 0 else "." for r in rows]
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
        # Find exact visible character horizon retained after truncation
        vis_end = 0
        for t in range(L):
            if attn[i, t] > 0 and offsets[i, t, 1] > offsets[i, t, 0]:
                vis_end = max(vis_end, int(offsets[i, t, 1]))
        if vis_end <= 0:
            vis_end = len(txt)
        vis_txt = txt[:vis_end]

        raw_spans = []
        for m in SENT_SPLIT_RE.finditer(vis_txt):
            if m.group(0).strip():
                raw_spans.append((m.start(), m.end()))
        if not raw_spans:
            raw_spans = [(0, len(vis_txt))]

        # Merge trailing spans into bucket max_sents - 1 BEFORE computing novelty!
        if len(raw_spans) > max_sents:
            merged_spans = raw_spans[: max_sents - 1] + [(raw_spans[max_sents - 1][0], raw_spans[-1][1])]
        else:
            merged_spans = raw_spans

        seen_stems = set()
        for s_idx, (c_start, c_end) in enumerate(merged_spans):
            seg = vis_txt[c_start:c_end]
            words = [w.lower()[:6] for w in WORD_RE.findall(seg) if w.lower() not in STOPWORDS]
            if words:
                novel = sum(1 for w in words if w not in seen_stems)
                sent_novelty[i, s_idx] = float(novel) / float(len(words))
                seen_stems.update(words)
            else:
                sent_novelty[i, s_idx] = 0.0

        ends = np.array([sp[1] for sp in merged_spans], dtype=np.int64)
        for t in range(L):
            if attn[i, t] == 0:
                continue
            c0, c1 = offsets[i, t]
            if c1 <= c0:
                continue
            mid = (c0 + c1) // 2
            s_bucket = int(np.searchsorted(ends, mid, side="right"))
            if s_bucket >= len(merged_spans):
                s_bucket = len(merged_spans) - 1
            sent_ids[i, t] = s_bucket

    edit_map = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 3}
    origin_y = torch.tensor([int(r["origin_label"]) for r in rows], dtype=torch.long)
    edit_lv = torch.tensor([edit_map.get(r.get("edit_level", "E0"), 0) for r in rows], dtype=torch.long)
    lex_dist = torch.tensor([float(r.get("measured_lex_dist", 0.0)) for r in rows], dtype=torch.float32)
    nom_str = torch.tensor([min(1.0, max(0.0, float(r.get("nominal_strength", 0.0)))) for r in rows], dtype=torch.float32)

    p_to_int = {}
    p_ints = []
    for i, r in enumerate(rows):
        pid = r.get("parent_id", f"ind_{i}")
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
        "lex_dist": lex_dist,
        "nom_str": nom_str,
        "parent_idx": torch.tensor(p_ints, dtype=torch.long),
        "row_ids": [r["id"] for r in rows],
        "parent_ids": [r.get("parent_id", r.get("prompt_id", r["id"])) for r in rows],
    }


def compute_paired_alignment_loss(z_orig: torch.Tensor, parent_idx: torch.Tensor, origin_y: torch.Tensor, edit_lv: torch.Tensor, lex_dist: torch.Tensor):
    z_norm = F.normalize(z_orig, p=2, dim=-1, eps=1e-6)
    unique_p = torch.unique(parent_idx)
    losses = []
    for p in unique_p:
        pm = (parent_idx == p)
        for o_val in (0, 1):
            pom = pm & (origin_y == o_val)
            i0 = torch.nonzero(pom & (edit_lv == 0), as_tuple=False).flatten()
            if i0.numel() == 0:
                continue
            z0 = z_norm[i0[0]]
            others = torch.nonzero(pom & (edit_lv > 0), as_tuple=False).flatten()
            for idx in others:
                w_ret = (1.0 - lex_dist[idx]).clamp(min=0.25, max=1.0)
                losses.append(w_ret * torch.sum((z_norm[idx] - z0) ** 2))
    if not losses:
        return torch.tensor(0.0, device=z_orig.device)
    return torch.stack(losses).mean()


def compute_supcon_loss(h: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1):
    z = F.normalize(h, p=2, dim=-1, eps=1e-6)
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
def evaluate_split(model: CEDARDetector, tensor_dict: dict, batch_size: int = 128):
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
    """
    Strictly lexicographic checkpoint selection on disjoint `val_select`:
      1. Compute exact parent-cluster conformal 5%-FPR threshold tau_sel on `val_select` human examples (`score > tau_sel`).
      2. Primary key: Worst-Edit Balanced Accuracy min_{e in {0,1,2,3}} 0.5 * (TNR_e + TPR_e).
      3. Secondary key (used ONLY on exact tie in primary key): Mean-Edit Balanced Accuracy.
    """
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
        assert m0.sum() > 0 and m1.sum() > 0, f"Missing cell at edit={ev} in val_select"
        tnr_e = float((preds[m0] == 0).mean())
        tpr_e = float((preds[m1] == 1).mean())
        edit_bal_accs.append(0.5 * (tnr_e + tpr_e))

    worst_ebal = float(min(edit_bal_accs))
    mean_ebal = float(np.mean(edit_bal_accs))
    return (round(worst_ebal, 6), round(mean_ebal, 6)), worst_ebal, mean_ebal


def train_and_eval_variant(
    run_name: str,
    seed: int,
    backbone_name: str,
    tokenized_splits: dict,
    data_fingerprint: str,
    mode: str,
    epochs: int = 4,
    lr_backbone: float = 2.5e-5,
    lr_head: float = 2e-4,
):
    model_src_bytes = open("/content/src/cedar_model.py", "rb").read() if os.path.exists("/content/src/cedar_model.py") else b""
    script_path = globals().get("__file__", "/content/scripts/step4_train_cedar_and_baselines.py")
    script_src_bytes = open(script_path, "rb").read() if os.path.exists(script_path) else b""
    src_hash = hashlib.sha256(model_src_bytes + script_src_bytes).hexdigest()[:12]
    cfg_sig = hashlib.sha256(
        f"{CODE_VERSION}|{src_hash}|{data_fingerprint}|{run_name}|{seed}|{backbone_name}|{mode}|{epochs}|{lr_backbone}|{lr_head}".encode()
    ).hexdigest()[:16]
    out_file = os.path.join(PRED_DIR, f"{run_name}_seed{seed}.json")
    if os.path.exists(out_file):
        with open(out_file) as f:
            old = json.load(f)
        if old.get("config_hash") == cfg_sig:
            print(f"  [SKIP VERIFIED CACHE] {run_name}_seed{seed}")
            return

    t0 = time.time()
    set_seed(seed)

    use_dte = mode in {"cedar_full", "cedar_no_nsd", "cedar_no_pair", "cedar_no_gdro", "cedar_unprotected_nsd"}
    use_pm_mlp = (mode == "cedar_param_matched_mlp")
    nsd_rank = 32 if mode in {"cedar_full", "cedar_no_dte", "cedar_param_matched_mlp", "cedar_no_pair", "cedar_no_gdro", "cedar_unprotected_nsd"} else 0
    nsd_alpha = 0.0 if mode == "cedar_unprotected_nsd" else 0.5
    joint_8 = (mode == "joint_8class_ce")

    use_gdro = mode in {"erm_group_dro", "cedar_full", "cedar_no_nsd", "cedar_no_dte", "cedar_param_matched_mlp", "cedar_no_pair", "cedar_unprotected_nsd"}
    use_pair = mode in {"erm_paired_consist", "cedar_full", "cedar_no_nsd", "cedar_no_dte", "cedar_param_matched_mlp", "cedar_no_gdro", "cedar_unprotected_nsd"}

    model = CEDARDetector(
        backbone_name=backbone_name,
        proj_dim=512,
        use_dte=use_dte,
        use_param_matched_mlp=use_pm_mlp,
        nsd_rank=nsd_rank,
        nsd_signal_alpha=nsd_alpha,
        joint_8class=joint_8,
    ).to("cuda")

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

    tr = tokenized_splits["train"]
    parent_to_indices = defaultdict(list)
    for idx, pid in enumerate(tr["parent_idx"].tolist()):
        parent_to_indices[pid].append(idx)
    unique_parents = sorted(parent_to_indices.keys())
    steps_per_full_epoch = math.ceil(len(unique_parents) / 8)
    target_total_steps = epochs * steps_per_full_epoch

    # Pre-construct exact epoch batch schedule so CosineAnnealingLR T_max is 100% exact!
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

    # Log-space Group-DRO weights over 8 (Origin, Edit) groups (overflow/underflow proof!)
    log_q = torch.zeros(8, device="cuda", dtype=torch.float32) - math.log(8.0)
    gdro_eta = 0.05

    best_tuple = (-1.0, -1.0)
    best_state = None
    best_val_info = {}
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
            l_dist = tr["lex_dist"][idx_t].to("cuda")
            n_str = tr["nom_str"][idx_t].to("cuda")
            p_id = tr["parent_idx"][idx_t].to("cuda")

            out = model(in_ids, attn, s_ids, s_nov)

            if joint_8:
                y_joint = y_o * 4 + y_e
                loss = F.cross_entropy(out["joint_logits"], y_joint)
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
                    loss = loss + 0.3 * F.cross_entropy(out["edit_logits"], y_e)
                elif mode == "supcon_orth_disentangle":
                    supcon = compute_supcon_loss(out["h"], y_o, temperature=0.1)
                    w_o = F.normalize(model.origin_head.weight, dim=-1)
                    w_e = F.normalize(model.edit_cls_head.weight, dim=-1)
                    orth_pen = ((w_o @ w_e.T) ** 2).mean()
                    loss = loss + 0.35 * supcon + 0.25 * F.cross_entropy(out["edit_logits"], y_e) + 0.5 * orth_pen

                if use_pair:
                    pair_loss = compute_paired_alignment_loss(out["z_orig"], p_id, y_o, y_e, l_dist)
                    loss = loss + 0.5 * pair_loss

                if mode.startswith("cedar"):
                    int_loss = F.smooth_l1_loss(out["edit_intensity"], n_str)
                    loss = loss + 0.2 * int_loss

            assert torch.isfinite(loss), f"Non-finite loss in {run_name}_seed{seed} at step {steps_done}"
            h_for_nsd = out["h"].detach()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clip encoder+origin parameters strictly separately from detached probe for 100% zero interference!
            grad_norm = torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            assert torch.isfinite(grad_norm), f"Non-finite main gradient norm in {run_name}_seed{seed} at step {steps_done}"
            if probe_params:
                p_gnorm = torch.nn.utils.clip_grad_norm_(probe_params, 1.0)
                assert torch.isfinite(p_gnorm), f"Non-finite probe gradient norm in {run_name}_seed{seed} at step {steps_done}"
            optimizer.step()
            scheduler.step()
            steps_done += 1

            # Update NSD subspace STRICTLY AFTER backward() and step()!
            if nsd_rank > 0:
                model.update_nsd_subspace_post_step(h_for_nsd, p_id, y_o, y_e, warmup_updates=15)

        eval_due = (mode != "unedited_erm") or ((ep_i + 1) % 4 == 0) or (ep_i == len(all_epoch_batches) - 1)
        if eval_due:
            sel_probs, _ = evaluate_split(model, tokenized_splits["val_select"])
            sel_tuple, w_ebal, m_ebal = selection_worst_edit_bal_acc(sel_probs, tokenized_splits["val_select"])
            if sel_tuple > best_tuple:
                best_tuple = sel_tuple
                best_val_info = {
                    "step": steps_done,
                    "val_select_worst_edit_bal_acc": round(w_ebal, 5),
                    "val_select_mean_edit_bal_acc": round(m_ebal, 5),
                }
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None, f"No valid checkpoint selected for {run_name}_seed{seed}"
    model.load_state_dict(best_state)
    nsd_diag = model.get_nsd_diagnostics()

    # Compute locked deployment thresholds on UNTOUCHED `val_calib` partition via parent-cluster conformal risk control!
    cal_probs, _ = evaluate_split(model, tokenized_splits["val_calib"])
    cal_p = np.asarray(cal_probs, dtype=np.float64)
    cal_y = tokenized_splits["val_calib"]["origin_y"].numpy()
    cal_e = tokenized_splits["val_calib"]["edit_lv"].numpy()
    cal_pids = tokenized_splits["val_calib"]["parent_ids"]

    hum_all_idx = np.where(cal_y == 0)[0]
    hum_e0_idx = np.where((cal_y == 0) & (cal_e == 0))[0]

    tau_cal_all5 = cluster_conformal_upper_threshold(cal_p[hum_all_idx], [cal_pids[i] for i in hum_all_idx], alpha=0.05)
    tau_cal_e0_5 = cluster_conformal_upper_threshold(cal_p[hum_e0_idx], [cal_pids[i] for i in hum_e0_idx], alpha=0.05)

    eval_payload = {
        "config_hash": cfg_sig,
        "data_fingerprint": data_fingerprint,
        "code_version": CODE_VERSION,
        "run_name": run_name,
        "seed": seed,
        "backbone": backbone_name,
        "mode": mode,
        "exact_optimizer_steps": steps_done,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_val_select": best_val_info,
        "nsd_diagnostics": nsd_diag,
        "locked_thresholds": {
            "tau_cal_all5": tau_cal_all5,
            "tau_cal_e0_5": tau_cal_e0_5,
            "realized_cal_all_fpr": float((cal_p[hum_all_idx] > tau_cal_all5).mean()),
            "realized_cal_e0_fpr": float((cal_p[hum_e0_idx] > tau_cal_e0_5).mean()),
        },
        "train_seconds": round(time.time() - t0, 1),
        "row_ids": {},
        "preds": {},
        "edit_intensity": {},
    }

    for sp_name, t_dict in tokenized_splits.items():
        if sp_name == "train":
            continue
        pr, ei = evaluate_split(model, t_dict)
        eval_payload["row_ids"][sp_name] = t_dict["row_ids"]
        eval_payload["preds"][sp_name] = pr  # Full float64 precision!
        if mode.startswith("cedar"):
            eval_payload["edit_intensity"][sp_name] = ei

    with open(out_file, "w") as f:
        json.dump(eval_payload, f)

    print(
        f"  [DONE] {run_name}_seed{seed} ({eval_payload['train_seconds']}s, steps={steps_done}) "
        f"| val_sel_worst_ebal={best_val_info['val_select_worst_edit_bal_acc']:.4f} | nsd={nsd_diag}"
    )
    del model, optimizer, best_state
    gc.collect()
    torch.cuda.empty_cache()


def load_raw_splits():
    splits = {}
    for sp in ["train", "val", "test_id", "test_ood_ar", "test_ood_llada"]:
        with open(os.path.join(DATA_DIR, f"split_{sp}.json")) as f:
            splits[sp] = json.load(f)
    for ext in ["external_esl_full", "external_mixset", "external_beemo", "external_mage_wild"]:
        with open(os.path.join(DATA_DIR, f"{ext}.json")) as f:
            splits[ext] = json.load(f)
    return validate_and_partition_splits(splits)


def main():
    raw_splits = load_raw_splits()
    data_fp = compute_dataset_fingerprint(raw_splits)
    seeds = [42, 1042, 2042]

    # =====================================================================
    # SUITE 1: Primary Backbone (answerdotai/ModernBERT-large, 3 Seeds)
    #   8 Same-Data Baselines/Main + 6 Complete Component Ablations
    # =====================================================================
    mb_id = "answerdotai/ModernBERT-large"
    print(f"\n>>> Pre-tokenizing all splits for {mb_id} (data_fp={data_fp}) ...")
    mb_tok = AutoTokenizer.from_pretrained(mb_id)
    mb_splits = {k: pretokenize_split(v, mb_tok, max_len=384) for k, v in raw_splits.items()}

    mb_variants = [
        ("mb_unedited_erm", "unedited_erm"),
        ("mb_augmented_erm", "augmented_erm"),
        ("mb_joint_8class_ce", "joint_8class_ce"),
        ("mb_multitask_2head", "multitask_2head"),
        ("mb_erm_paired_consist", "erm_paired_consist"),
        ("mb_erm_group_dro", "erm_group_dro"),
        ("mb_supcon_orth_disentangle", "supcon_orth_disentangle"),
        ("mb_cedar_full", "cedar_full"),
        # Complete Leave-One-Out & Control Ablations
        ("mb_ablation_no_nsd", "cedar_no_nsd"),
        ("mb_ablation_no_dte", "cedar_no_dte"),
        ("mb_ablation_param_matched_mlp", "cedar_param_matched_mlp"),
        ("mb_ablation_no_pair", "cedar_no_pair"),
        ("mb_ablation_no_gdro", "cedar_no_gdro"),
        ("mb_ablation_unprotected_nsd", "cedar_unprotected_nsd"),
    ]

    for r_name, mode in mb_variants:
        for sd in seeds:
            train_and_eval_variant(r_name, sd, mb_id, mb_splits, data_fp, mode=mode, epochs=4, lr_backbone=2.5e-5, lr_head=2e-4)

    del mb_splits, mb_tok
    gc.collect()

    # =====================================================================
    # SUITE 2: Second Backbone Generality Verification (microsoft/deberta-v3-large, 3 Seeds)
    # =====================================================================
    db_id = "microsoft/deberta-v3-large"
    print(f"\n>>> Pre-tokenizing all splits for {db_id} ...")
    db_tok = AutoTokenizer.from_pretrained(db_id)
    db_splits = {k: pretokenize_split(v, db_tok, max_len=384) for k, v in raw_splits.items()}

    db_variants = [
        ("db_unedited_erm", "unedited_erm"),
        ("db_augmented_erm", "augmented_erm"),
        ("db_supcon_orth_disentangle", "supcon_orth_disentangle"),
        ("db_cedar_full", "cedar_full"),
    ]
    for r_name, mode in db_variants:
        for sd in seeds:
            train_and_eval_variant(r_name, sd, db_id, db_splits, data_fp, mode=mode, epochs=4, lr_backbone=1.5e-5, lr_head=1.5e-4)

    print("\n=== ALL V2 TRAINING, BASELINE, ABLATION & MULTI-BACKBONE RUNS COMPLETE! ===")


if __name__ == "__main__":
    main()

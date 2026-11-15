import os
import gc
import sys
import json
import time
import math
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoTokenizer

sys.path.insert(0, "/content")
from src.cedar_model import CEDARDetector
from scripts.step4_train_cedar_and_baselines import (
    DATA_DIR,
    PRED_DIR,
    CODE_VERSION,
    set_seed,
    cluster_conformal_upper_threshold,
    compute_dataset_fingerprint,
    load_raw_splits,
    pretokenize_split,
    compute_paired_alignment_loss,
    evaluate_split,
    selection_worst_edit_bal_acc,
)


def run_cedar_with_rank(rank: int, seed: int, mb_id: str, tokenized_splits: dict, data_fp: str, save_geometry: bool = False):
    run_name = f"mb_cedar_rank{rank}"
    model_src_bytes = open("/content/src/cedar_model.py", "rb").read() if os.path.exists("/content/src/cedar_model.py") else b""
    script_path = globals().get("__file__", "/content/scripts/step4b_nsd_rank_and_spectrum.py")
    script_src_bytes = open(script_path, "rb").read() if os.path.exists(script_path) else b""
    src_hash = hashlib.sha256(model_src_bytes + script_src_bytes).hexdigest()[:12]
    cfg_sig = hashlib.sha256(f"{CODE_VERSION}|{src_hash}|{data_fp}|{run_name}|{seed}|{mb_id}|rank{rank}".encode()).hexdigest()[:16]
    out_file = os.path.join(PRED_DIR, f"{run_name}_seed{seed}.json")
    geom_file = os.path.join(DATA_DIR, "nsd_spectrum_and_geometry.json")

    if os.path.exists(out_file) and (not save_geometry or os.path.exists(geom_file)):
        with open(out_file) as f:
            old = json.load(f)
        if old.get("config_hash") == cfg_sig:
            print(f"  [SKIP VERIFIED CACHE] {run_name}_seed{seed}")
            return

    t0 = time.time()
    set_seed(seed)
    model = CEDARDetector(
        backbone_name=mb_id,
        proj_dim=512,
        use_dte=True,
        use_param_matched_mlp=False,
        nsd_rank=rank,
        nsd_signal_alpha=0.5,
        joint_8class=False,
    ).to("cuda")

    bb_params = list(model.backbone.parameters())
    bb_ids = {id(p) for p in bb_params}
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]
    probe_params = list(model.edit_intensity_head.parameters())
    probe_ids = {id(p) for p in probe_params}
    main_params = [p for p in model.parameters() if id(p) not in probe_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": 2.5e-5, "weight_decay": 0.01},
            {"params": head_params, "lr": 2e-4, "weight_decay": 0.01},
        ]
    )

    tr = tokenized_splits["train"]
    from collections import defaultdict
    import random

    parent_to_indices = defaultdict(list)
    for idx, pid in enumerate(tr["parent_idx"].tolist()):
        parent_to_indices[pid].append(idx)
    unique_parents = sorted(parent_to_indices.keys())

    all_epoch_batches = []
    for _ in range(4):
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

    best_tuple = (-1.0, -1.0)
    best_state = None
    best_val_info = {}
    steps_done = 0

    for batches in all_epoch_batches:
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
            per_ex_bce = F.binary_cross_entropy_with_logits(out["origin_logit"], y_o.float(), reduction="none")
            g_ids = y_o * 4 + y_e
            g_losses = torch.zeros(8, device="cuda", dtype=torch.float32)
            g_present = torch.zeros(8, dtype=torch.bool, device="cuda")
            for g in range(8):
                gm = (g_ids == g)
                if gm.any():
                    g_losses[g] = per_ex_bce[gm].mean()
                    g_present[g] = True
            with torch.no_grad():
                delta_lq = torch.where(g_present, 0.05 * g_losses.detach(), torch.zeros_like(g_losses))
                log_q = F.log_softmax(log_q + delta_lq, dim=0)
            q_probs = log_q.exp() * g_present.float()
            q_probs = q_probs / q_probs.sum().clamp(min=1e-8)
            cls_loss = (q_probs * g_losses).sum()

            pair_loss = compute_paired_alignment_loss(out["z_orig"], p_id, y_o, y_e, l_dist)
            int_loss = F.smooth_l1_loss(out["edit_intensity"], n_str)
            loss = cls_loss + 0.5 * pair_loss + 0.2 * int_loss
            assert torch.isfinite(loss), f"Non-finite loss in {run_name}_seed{seed}"

            h_for_nsd = out["h"].detach()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            pgnorm = torch.nn.utils.clip_grad_norm_(probe_params, 1.0)
            assert torch.isfinite(gnorm) and torch.isfinite(pgnorm), f"Non-finite grad norm in {run_name}_seed{seed}"
            optimizer.step()
            scheduler.step()
            steps_done += 1
            if rank > 0:
                model.update_nsd_subspace_post_step(h_for_nsd, p_id, y_o, y_e, warmup_updates=15)

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

    assert best_state is not None, f"No valid checkpoint for {run_name}_seed{seed}"
    model.load_state_dict(best_state)
    nsd_diag = model.get_nsd_diagnostics()

    cal_probs, _ = evaluate_split(model, tokenized_splits["val_calib"])
    cal_p = np.asarray(cal_probs, dtype=np.float64)
    cal_y = tokenized_splits["val_calib"]["origin_y"].numpy()
    cal_e = tokenized_splits["val_calib"]["edit_lv"].numpy()
    cal_pids = tokenized_splits["val_calib"]["parent_ids"]
    hum_all_idx = np.where(cal_y == 0)[0]
    hum_e0_idx = np.where((cal_y == 0) & (cal_e == 0))[0]
    tau_cal_all5 = cluster_conformal_upper_threshold(cal_p[hum_all_idx], [cal_pids[i] for i in hum_all_idx], alpha=0.05)
    tau_cal_e0_5 = cluster_conformal_upper_threshold(cal_p[hum_e0_idx], [cal_pids[i] for i in hum_e0_idx], alpha=0.05)

    if rank in {8, 16, 64}:
        eval_payload = {
            "config_hash": cfg_sig,
            "data_fingerprint": data_fp,
            "code_version": CODE_VERSION,
            "run_name": run_name,
            "seed": seed,
            "backbone": mb_id,
            "mode": f"cedar_rank{rank}",
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
            eval_payload["preds"][sp_name] = pr
            eval_payload["edit_intensity"][sp_name] = ei
        with open(out_file, "w") as f:
            json.dump(eval_payload, f)
        print(f"  [DONE] {run_name}_seed{seed} ({eval_payload['train_seconds']}s) | nsd={nsd_diag}")

    if save_geometry:
        print("  >>> Extracting empirical NSD spectrum & cross-editor subspace transfer geometry ...")
        model.eval()
        with torch.no_grad():
            S_e = 0.5 * (model.ema_sec_edit + model.ema_sec_edit.T)
            S_o = 0.5 * (model.ema_sec_orig + model.ema_sec_orig.T)
            M_n = S_e - 0.5 * S_o
            evals_e = torch.linalg.eigvalsh(S_e).flip(0)[:64].cpu().tolist()
            evals_o = torch.linalg.eigvalsh(S_o).flip(0)[:64].cpu().tolist()
            evals_m, evecs_m = torch.linalg.eigh(M_n)
            evals_m_top = evals_m.flip(0)[:64].cpu().tolist()
            U_top64 = evecs_m[:, -64:].flip(1)  # (D, 64)

            def extract_reps(t_dict):
                hs, zos, zns = [], [], []
                N = t_dict["input_ids"].shape[0]
                for st in range(0, N, 128):
                    ed = min(st + 128, N)
                    h, zo, zn = model.encode(
                        t_dict["input_ids"][st:ed].to("cuda"),
                        t_dict["attention_mask"][st:ed].to("cuda"),
                        t_dict["sent_ids"][st:ed].to("cuda"),
                        t_dict["sent_novelty"][st:ed].to("cuda"),
                    )
                    hs.append(h.cpu())
                    zos.append(zo.cpu())
                    zns.append(zn.cpu())
                return torch.cat(hs, 0), torch.cat(zos, 0), torch.cat(zns, 0)

            def empirical_second_moments(H_mat, t_dict):
                p_idx = t_dict["parent_idx"]
                y_o = t_dict["origin_y"]
                e_lv = t_dict["edit_lv"]
                deltas, gammas = [], []
                for p in torch.unique(p_idx):
                    pm = (p_idx == p)
                    for ov in (0, 1):
                        pom = pm & (y_o == ov)
                        i0 = torch.nonzero(pom & (e_lv == 0), as_tuple=False).flatten()
                        if i0.numel() > 0:
                            for ev in (1, 2):
                                ik = torch.nonzero(pom & (e_lv == ev), as_tuple=False).flatten()
                                if ik.numel() > 0:
                                    deltas.append(H_mat[ik[0]] - H_mat[i0[0]])
                    for ev in (0, 1, 2):
                        a0 = torch.nonzero(pm & (y_o == 0) & (e_lv == ev), as_tuple=False).flatten()
                        a1 = torch.nonzero(pm & (y_o == 1) & (e_lv == ev), as_tuple=False).flatten()
                        if a0.numel() > 0 and a1.numel() > 0:
                            gammas.append(H_mat[a1[0]] - H_mat[a0[0]])
                D_mat = torch.stack(deltas, 0).to("cuda")
                G_mat = torch.stack(gammas, 0).to("cuda")
                Se_emp = (D_mat.T @ D_mat) / float(D_mat.shape[0])
                So_emp = (G_mat.T @ G_mat) / float(G_mat.shape[0])
                return Se_emp, So_emp

            rank_grid = [1, 2, 4, 8, 16, 24, 32, 48, 64]
            cross_editor_curves = {}
            probe_disentanglement = {}

            tr_H, tr_Zo, tr_Zn = extract_reps(tokenized_splits["train"])
            tr_yo = tokenized_splits["train"]["origin_y"].numpy()
            tr_ye = tokenized_splits["train"]["edit_lv"].numpy()
            # Filter E0..E2 for clean 3-way edit probe comparison across ID & OOD
            tr_m02 = (tr_ye <= 2)

            clf_o_h = LogisticRegression(max_iter=300).fit(tr_H.numpy()[tr_m02], tr_yo[tr_m02])
            clf_e_h = LogisticRegression(max_iter=300).fit(tr_H.numpy()[tr_m02], tr_ye[tr_m02])
            clf_o_zo = LogisticRegression(max_iter=300).fit(tr_Zo.numpy()[tr_m02], tr_yo[tr_m02])
            clf_e_zo = LogisticRegression(max_iter=300).fit(tr_Zo.numpy()[tr_m02], tr_ye[tr_m02])
            clf_o_zn = LogisticRegression(max_iter=300).fit(tr_Zn.numpy()[tr_m02], tr_yo[tr_m02])
            clf_e_zn = LogisticRegression(max_iter=300).fit(tr_Zn.numpy()[tr_m02], tr_ye[tr_m02])

            for sp_eval in ["test_id", "test_ood_ar", "test_ood_llada"]:
                H_sp, Zo_sp, Zn_sp = extract_reps(tokenized_splits[sp_eval])
                Se_sp, So_sp = empirical_second_moments(H_sp, tokenized_splits[sp_eval])
                tr_Se = float(torch.trace(Se_sp).item())
                tr_So = float(torch.trace(So_sp).item())
                curve_sp = {}
                for rk in rank_grid:
                    Ur = U_top64[:, :rk]
                    Pr = Ur @ Ur.T
                    cap_e = float(torch.sum(Pr * Se_sp).item()) / max(1e-8, tr_Se)
                    cap_o = float(torch.sum(Pr * So_sp).item()) / max(1e-8, tr_So)
                    curve_sp[str(rk)] = {
                        "edit_energy_captured": round(cap_e, 4),
                        "orig_energy_captured": round(cap_o, 4),
                    }
                cross_editor_curves[sp_eval] = curve_sp

                yo_sp = tokenized_splits[sp_eval]["origin_y"].numpy()
                ye_sp = tokenized_splits[sp_eval]["edit_lv"].numpy()
                m02 = (ye_sp <= 2)
                probe_disentanglement[sp_eval] = {
                    "pre_deflation_h": {
                        "origin_auroc": round(float(roc_auc_score(yo_sp[m02], clf_o_h.predict_proba(H_sp.numpy()[m02])[:, 1])), 4),
                        "edit3_acc": round(float(accuracy_score(ye_sp[m02], clf_e_h.predict(H_sp.numpy()[m02]))), 4),
                    },
                    "post_deflation_z_orig": {
                        "origin_auroc": round(float(roc_auc_score(yo_sp[m02], clf_o_zo.predict_proba(Zo_sp.numpy()[m02])[:, 1])), 4),
                        "edit3_acc": round(float(accuracy_score(ye_sp[m02], clf_e_zo.predict(Zo_sp.numpy()[m02]))), 4),
                    },
                    "deflated_nuisance_z_nuis": {
                        "origin_auroc": round(float(roc_auc_score(yo_sp[m02], clf_o_zn.predict_proba(Zn_sp.numpy()[m02])[:, 1])), 4),
                        "edit3_acc": round(float(accuracy_score(ye_sp[m02], clf_e_zn.predict(Zn_sp.numpy()[m02]))), 4),
                    },
                }

        geom_payload = {
            "top64_eigenvalues": {
                "S_edit": [round(x, 5) for x in evals_e],
                "S_orig": [round(x, 5) for x in evals_o],
                "M_nuis": [round(x, 5) for x in evals_m_top],
            },
            "cross_editor_variance_capture_by_rank": cross_editor_curves,
            "linear_probe_disentanglement": probe_disentanglement,
        }
        with open(geom_file, "w") as f:
            json.dump(geom_payload, f, indent=2)
        print(f"  [SAVED GEOMETRY] -> {geom_file}")

    del model, optimizer, best_state
    gc.collect()
    torch.cuda.empty_cache()


def main():
    raw_splits = load_raw_splits()
    data_fp = compute_dataset_fingerprint(raw_splits)
    mb_id = "answerdotai/ModernBERT-large"
    mb_tok = AutoTokenizer.from_pretrained(mb_id)
    mb_splits = {k: pretokenize_split(v, mb_tok, max_len=384) for k, v in raw_splits.items()}

    for rank in [8, 16, 64]:
        for sd in [42, 1042, 2042]:
            run_cedar_with_rank(rank, sd, mb_id, mb_splits, data_fp, save_geometry=False)

    # Extract full spectrum & cross-editor subspace geometry on canonical rank=32, seed=42
    run_cedar_with_rank(32, 42, mb_id, mb_splits, data_fp, save_geometry=True)
    print("\n=== NSD RANK SENSITIVITY SWEEP & GEOMETRIC SPECTRUM EXTRACTION COMPLETE! ===")


if __name__ == "__main__":
    main()

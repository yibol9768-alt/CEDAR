import os
import sys
import json
import math
import random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "/content/scripts/v3_theorem2_convergence_and_geometry.py"))))
sys.path.insert(0, "/content" if os.path.exists("/content/src") else _REPO_ROOT)
from src.cedar_v2_model import CEDARv2Detector, eps_unit_norm
from scripts.v3_train_cedar_v2_suite import (
    V2_DIR,
    OUT_DIR,
    CKPT_DIR,
    load_and_verify_all_splits,
    pretokenize_split,
)


@torch.inference_mode()
def extract_all_reps(model: CEDARv2Detector, t_dict: dict, batch_size: int = 128):
    model.eval()
    N = t_dict["input_ids"].shape[0]
    hs, zos, zns = [], [], []
    for st in range(0, N, batch_size):
        ed = min(st + batch_size, N)
        h_hat, z_orig, z_nuis = model.encode(
            t_dict["input_ids"][st:ed].to("cuda"),
            t_dict["attention_mask"][st:ed].to("cuda"),
            t_dict["sent_ids"][st:ed].to("cuda"),
            t_dict["sent_novelty"][st:ed].to("cuda"),
        )
        hs.append(h_hat.cpu())
        zos.append(z_orig.cpu())
        zns.append(z_nuis.cpu())
    return torch.cat(hs, 0), torch.cat(zos, 0), torch.cat(zns, 0)


def build_per_parent_operators_fp64(H_cpu: torch.Tensor, t_dict: dict, rows_meta: list, lambda_sig: float = 0.5):
    """
    For each complete parent octet i (with cells o in {0,1}, e in {0,1,2,3}), computes its exact
    float64 symmetric operator A_i in R^{d x d}, S_edit,i, and S_orig,i.
    """
    H64 = H_cpu.double()
    D = H64.shape[1]
    p_ids = t_dict["parent_ids"]
    y_o = t_dict["origin_y"].tolist()
    e_lv = t_dict["edit_lv"].tolist()
    raw_lvs = [r.get("edit_level", "E0") for r in rows_meta]
    dom_map = {r["parent_id"]: r["domain"] for r in rows_meta if "parent_id" in r}

    by_p = {}
    for idx, pid in enumerate(p_ids):
        if raw_lvs[idx] == "E4":
            continue
        by_p.setdefault(pid, {})[(y_o[idx], e_lv[idx])] = H64[idx]

    records = []
    for pid, cell_vecs in sorted(by_p.items()):
        if len(cell_vecs) < 8:
            continue
        if not all((ov, ev) in cell_vecs for ov in (0, 1) for ev in (0, 1, 2, 3)):
            continue
        deltas = [cell_vecs[(ov, ev)] - cell_vecs[(ov, 0)] for ov in (0, 1) for ev in (1, 2, 3)]
        gammas = [cell_vecs[(1, ev)] - cell_vecs[(0, ev)] for ev in (0, 1, 2, 3)]
        D_i = torch.stack(deltas, 0)  # (6, D)
        G_i = torch.stack(gammas, 0)  # (4, D)
        Se_i = (D_i.T @ D_i) / 6.0
        So_i = (G_i.T @ G_i) / 4.0
        A_i = 0.5 * ((Se_i - lambda_sig * So_i) + (Se_i - lambda_sig * So_i).T)
        records.append({
            "parent_id": pid,
            "domain": dom_map.get(pid, "unknown"),
            "A": A_i,
            "Se": Se_i,
            "So": So_i,
        })
    return records


def operator_to_shrinker_and_proj(M64: torch.Tensor, rank: int = 24, tau: float = 1e-4):
    M_sym = 0.5 * (M64 + M64.T)
    evals, evecs = torch.linalg.eigh(M_sym)
    r = min(rank, M_sym.shape[0])
    top_vals = evals[-r:].clamp(min=0.0)
    top_vecs = evecs[:, -r:]
    gates = top_vals / (top_vals + tau)
    P_N = (top_vecs * gates.unsqueeze(0)) @ top_vecs.T
    Pi_r = top_vecs @ top_vecs.T
    mu_r = float(evals[-r].item())
    gap_r = float((evals[-r] - evals[-r - 1]).item())
    return 0.5 * (P_N + P_N.T), 0.5 * (Pi_r + Pi_r.T), evals, mu_r, gap_r


def empirical_second_moments_eval(H_cpu: torch.Tensor, t_dict: dict, max_edit_lv: int = 3):
    H64 = H_cpu.double()
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
                for ev in range(1, max_edit_lv + 1):
                    ik = torch.nonzero(pom & (e_lv == ev), as_tuple=False).flatten()
                    if ik.numel() > 0:
                        deltas.append(H64[ik[0]] - H64[i0[0]])
        for ev in range(0, max_edit_lv + 1):
            a0 = torch.nonzero(pm & (y_o == 0) & (e_lv == ev), as_tuple=False).flatten()
            a1 = torch.nonzero(pm & (y_o == 1) & (e_lv == ev), as_tuple=False).flatten()
            if a0.numel() > 0 and a1.numel() > 0:
                gammas.append(H64[a1[0]] - H64[a0[0]])
    D_mat = torch.stack(deltas, 0)
    G_mat = torch.stack(gammas, 0)
    return (D_mat.T @ D_mat) / float(D_mat.shape[0]), (G_mat.T @ G_mat) / float(G_mat.shape[0])


def main():
    ckpt_path = os.path.join(CKPT_DIR, "mb_v3_cedar_split_op_seed42.pt")
    assert os.path.exists(ckpt_path), f"Missing canonical checkpoint: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    prov = json.load(open(os.path.join(V2_DIR, "provenance_lock.json")))
    mb_id = ckpt["model_kwargs"]["backbone_name"]
    mb_sha = prov["hf_model_commit_shas"][mb_id]
    tok = AutoTokenizer.from_pretrained(mb_id, revision=mb_sha)

    raw_splits, _ = load_and_verify_all_splits()
    needed = ["train_fit_v2", "train_op_v2", "val_select", "val_calib", "test_id", "test_ood_ar", "test_ood_llada", "test_confirm"]
    tok_splits = {k: pretokenize_split(raw_splits[k], tok, max_len=384) for k in needed}

    model = CEDARv2Detector(**ckpt["model_kwargs"]).to("cuda")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # =========================================================================
    # 1. Theorem 2 Finite-Sample Operator, Subspace, Shrinker & Score Stability
    #    Evaluated strictly on octets UNTOUCHED by trunk training (`train_fit_v2` excluded!)
    # =========================================================================
    print(">>> [1/3] Running Theorem 2 Disjoint-Pool Finite-Sample Perturbation Experiment (200 trials) ...")
    untouched_records = []
    for sp_k in ["train_op_v2", "val_select", "val_calib", "test_id"]:
        H_sp, _, _ = extract_all_reps(model, tok_splits[sp_k])
        recs = build_per_parent_operators_fp64(H_sp, tok_splits[sp_k], raw_splits[sp_k], lambda_sig=0.5)
        untouched_records.extend(recs)

    nr_recs = [r for r in untouched_records if r["domain"] == "instruct_prose"]
    mg_recs = [r for r in untouched_records if r["domain"] == "public_expository"]
    assert len(nr_recs) == 260 and len(mg_recs) == 260, f"Expected 260 NR + 260 MAGE untouched octets, got {len(nr_recs)}, {len(mg_recs)}"

    # Stack onto CUDA float64 for fast 200-trial exact spectral evaluations
    A_nr = torch.stack([r["A"] for r in nr_recs], 0).to("cuda")  # (260, 128, 128)
    A_mg = torch.stack([r["A"] for r in mg_recs], 0).to("cuda")  # (260, 128, 128)
    op_norms_all = torch.linalg.matrix_norm(torch.cat([A_nr, A_mg], 0), ord=2).cpu().numpy()

    H_conf, _, _ = extract_all_reps(model, tok_splits["test_confirm"])
    H_conf_64 = H_conf.double().to("cuda")
    w_hat_64 = model.origin_head.normalized_weight().detach().double().to("cuda")
    b_64 = float(model.origin_head.bias.detach().item())
    kappa = float(model.cosine_kappa)

    N_grid = [16, 32, 64, 128, 256]
    N_ref = 260
    num_trials = 200
    rng = random.Random(20261114)

    trial_metrics = {N: {"op_err": [], "frob_sq": [], "subspace_sin_theta": [], "shrinker_err": [], "max_logit_diff": [], "p99_logit_diff": []} for N in N_grid}
    ref_eigengaps = []
    ref_mu_rs = []

    for _ in range(num_trials):
        perm_nr = list(range(260))
        perm_mg = list(range(260))
        rng.shuffle(perm_nr)
        rng.shuffle(perm_mg)

        # Strictly disjoint estimation pool (130 NR + 130 MAGE = 260) vs reference pool (130 NR + 130 MAGE = 260)
        est_idx_nr, ref_idx_nr = perm_nr[:130], perm_nr[130:]
        est_idx_mg, ref_idx_mg = perm_mg[:130], perm_mg[130:]

        tau_ckpt = float(model.nsd_tau)
        A_ref = 0.5 * (A_nr[ref_idx_nr].mean(0) + A_mg[ref_idx_mg].mean(0))
        P_ref, Pi_ref, _, mu_r_ref, gap_ref = operator_to_shrinker_and_proj(A_ref, rank=24, tau=tau_ckpt)
        ref_mu_rs.append(mu_r_ref)
        ref_eigengaps.append(gap_ref)

        Z_ref = H_conf_64 - (H_conf_64 @ P_ref)
        Z_ref_hat = Z_ref / torch.linalg.vector_norm(Z_ref, ord=2, dim=-1, keepdim=True).clamp(min=1e-6)
        logits_ref = kappa * (Z_ref_hat @ w_hat_64) + b_64

        for N in N_grid:
            half_n = N // 2
            A_est = 0.5 * (A_nr[est_idx_nr[:half_n]].mean(0) + A_mg[est_idx_mg[:half_n]].mean(0))
            P_est, Pi_est, _, _, _ = operator_to_shrinker_and_proj(A_est, rank=24, tau=tau_ckpt)

            op_err = float(torch.linalg.matrix_norm(A_est - A_ref, ord=2).item())
            frob_sq = float(torch.sum((A_est - A_ref) ** 2).item())
            sin_th = float(torch.linalg.matrix_norm(Pi_est - Pi_ref, ord=2).item())
            sh_err = float(torch.linalg.matrix_norm(P_est - P_ref, ord=2).item())

            Z_est = H_conf_64 - (H_conf_64 @ P_est)
            Z_est_hat = Z_est / torch.linalg.vector_norm(Z_est, ord=2, dim=-1, keepdim=True).clamp(min=1e-6)
            logits_est = kappa * (Z_est_hat @ w_hat_64) + b_64
            ldiff = torch.abs(logits_est - logits_ref)

            trial_metrics[N]["op_err"].append(op_err)
            trial_metrics[N]["frob_sq"].append(frob_sq)
            trial_metrics[N]["subspace_sin_theta"].append(sin_th)
            trial_metrics[N]["shrinker_err"].append(sh_err)
            trial_metrics[N]["max_logit_diff"].append(float(ldiff.max().item()))
            trial_metrics[N]["p99_logit_diff"].append(float(torch.quantile(ldiff, 0.99).item()))

    convergence_summary = {}
    eff_x = []
    op_y = []
    sh_y = []
    for N in N_grid:
        eff_factor = math.sqrt((1.0 / float(N)) + (1.0 / float(N_ref)))
        eff_x.append(math.log(eff_factor))
        m_op = float(np.mean(trial_metrics[N]["op_err"]))
        m_sh = float(np.mean(trial_metrics[N]["shrinker_err"]))
        op_y.append(math.log(m_op))
        sh_y.append(math.log(m_sh))
        convergence_summary[str(N)] = {
            "effective_inv_sqrt_n_factor": round(eff_factor, 6),
            "op_err_mean": round(m_op, 6),
            "op_err_sd": round(float(np.std(trial_metrics[N]["op_err"])), 6),
            "shrinker_err_mean": round(m_sh, 6),
            "shrinker_err_sd": round(float(np.std(trial_metrics[N]["shrinker_err"])), 6),
            "subspace_sin_theta_mean": round(float(np.mean(trial_metrics[N]["subspace_sin_theta"])), 6),
            "subspace_sin_theta_sd": round(float(np.std(trial_metrics[N]["subspace_sin_theta"])), 6),
            "max_logit_diff_mean": round(float(np.mean(trial_metrics[N]["max_logit_diff"])), 6),
            "p99_logit_diff_mean": round(float(np.mean(trial_metrics[N]["p99_logit_diff"])), 6),
            "rms_frob_err": round(math.sqrt(float(np.mean(trial_metrics[N]["frob_sq"]))), 6),
        }

    # Log-log slope against exact disjoint finite-reference factor sqrt(1/N + 1/N_ref) (theoretical = +1.00, i.e. -0.50 in N!)
    slope_op = float(np.polyfit(eff_x, op_y, 1)[0])
    slope_sh = float(np.polyfit(eff_x, sh_y, 1)[0])

    # =========================================================================
    # 2. Spectrum, Cross-Editor Variance Capture & Linear Probe Disentanglement
    # =========================================================================
    print(">>> [2/3] Extracting Finalized Spectrum, Cross-Editor Capture & Linear Probe Disentanglement ...")
    Se_op = model.ema_sec_edit.double()
    So_op = model.ema_sec_orig.double()
    Mn_op = 0.5 * ((Se_op - 0.5 * So_op) + (Se_op - 0.5 * So_op).T)
    evals_e = torch.linalg.eigvalsh(0.5 * (Se_op + Se_op.T)).flip(0)[:64].cpu().tolist()
    evals_o = torch.linalg.eigvalsh(0.5 * (So_op + So_op.T)).flip(0)[:64].cpu().tolist()
    evals_m, evecs_m = torch.linalg.eigh(Mn_op)
    evals_m_top = evals_m.flip(0)[:64].cpu().tolist()
    U_top64 = evecs_m[:, -64:].flip(1)

    tr_H, tr_Zo, tr_Zn = extract_all_reps(model, tok_splits["train_fit_v2"])
    tr_yo = tok_splits["train_fit_v2"]["origin_y"].numpy()
    tr_ye = tok_splits["train_fit_v2"]["edit_lv"].numpy()
    tr_yr = tok_splits["train_fit_v2"]["edit_reg"].numpy()
    tr_m02 = (tr_ye <= 2)

    clf_o_h = LogisticRegression(max_iter=500).fit(tr_H.numpy()[tr_m02], tr_yo[tr_m02])
    clf_e_h = LogisticRegression(max_iter=500).fit(tr_H.numpy()[tr_m02], tr_ye[tr_m02])
    clf_r_h = LogisticRegression(max_iter=500).fit(tr_H.numpy(), tr_yr)

    clf_o_zo = LogisticRegression(max_iter=500).fit(tr_Zo.numpy()[tr_m02], tr_yo[tr_m02])
    clf_e_zo = LogisticRegression(max_iter=500).fit(tr_Zo.numpy()[tr_m02], tr_ye[tr_m02])
    clf_r_zo = LogisticRegression(max_iter=500).fit(tr_Zo.numpy(), tr_yr)

    clf_o_zn = LogisticRegression(max_iter=500).fit(tr_Zn.numpy()[tr_m02], tr_yo[tr_m02])
    clf_e_zn = LogisticRegression(max_iter=500).fit(tr_Zn.numpy()[tr_m02], tr_ye[tr_m02])
    clf_r_zn = LogisticRegression(max_iter=500).fit(tr_Zn.numpy(), tr_yr)

    rank_grid = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    cross_editor_curves = {}
    probe_disentanglement = {}
    residual_norm_audit = {}

    for sp_eval, max_ev in [("test_id", 3), ("test_ood_ar", 2), ("test_ood_llada", 2), ("test_confirm", 3)]:
        H_sp, Zo_sp, Zn_sp = extract_all_reps(model, tok_splits[sp_eval])
        res_norms = torch.linalg.vector_norm(Zo_sp, ord=2, dim=-1).numpy()
        residual_norm_audit[sp_eval] = {
            "min": round(float(res_norms.min()), 5),
            "p01": round(float(np.quantile(res_norms, 0.01)), 5),
            "p05": round(float(np.quantile(res_norms, 0.05)), 5),
            "mean": round(float(res_norms.mean()), 5),
        }

        Se_sp, So_sp = empirical_second_moments_eval(H_sp, tok_splits[sp_eval], max_edit_lv=max_ev)
        tr_Se = float(torch.trace(Se_sp).item())
        tr_So = float(torch.trace(So_sp).item())
        curve_sp = {}
        for rk in rank_grid:
            Ur = U_top64[:, :rk]
            Pr = (Ur @ Ur.T).double().cpu()
            cap_e = float(torch.sum(Pr * Se_sp.double().cpu()).item()) / max(1e-12, tr_Se)
            cap_o = float(torch.sum(Pr * So_sp.double().cpu()).item()) / max(1e-12, tr_So)
            curve_sp[str(rk)] = {
                "edit_energy_captured": round(cap_e, 4),
                "orig_energy_captured": round(cap_o, 4),
            }
        cross_editor_curves[sp_eval] = curve_sp

        yo_sp = tok_splits[sp_eval]["origin_y"].numpy()
        ye_sp = tok_splits[sp_eval]["edit_lv"].numpy()
        yr_sp = tok_splits[sp_eval]["edit_reg"].numpy()
        m02 = (ye_sp <= 2)
        p_entry = {
            "pre_deflation_h_hat": {
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
        if sp_eval == "test_confirm":
            p_entry["pre_deflation_h_hat"]["register3_acc"] = round(float(accuracy_score(yr_sp, clf_r_h.predict(H_sp.numpy()))), 4)
            p_entry["post_deflation_z_orig"]["register3_acc"] = round(float(accuracy_score(yr_sp, clf_r_zo.predict(Zo_sp.numpy()))), 4)
            p_entry["deflated_nuisance_z_nuis"]["register3_acc"] = round(float(accuracy_score(yr_sp, clf_r_zn.predict(Zn_sp.numpy()))), 4)
        probe_disentanglement[sp_eval] = p_entry

    out_payload = {
        "theorem2_empirical_verification": {
            "untouched_pool_total_octets": 520,
            "disjoint_reference_octets_per_trial": N_ref,
            "num_trials": num_trials,
            "empirical_max_A_i_op_norm": round(float(op_norms_all.max()), 5),
            "theoretical_A_i_op_bound": 4.0,
            "mean_ref_mu_24": round(float(np.mean(ref_mu_rs)), 6),
            "mean_ref_eigengap_24": round(float(np.mean(ref_eigengaps)), 6),
            "loglog_exponent_vs_eff_inv_sqrt_n": {
                "operator_error_exponent": round(slope_op, 4),
                "shrinker_error_exponent": round(slope_sh, 4),
                "theoretical_exponent": 1.0,
            },
            "by_N": convergence_summary,
        },
        "residual_norm_audit": residual_norm_audit,
        "top64_eigenvalues": {
            "S_edit": [round(x, 6) for x in evals_e],
            "S_orig": [round(x, 6) for x in evals_o],
            "M_nuis": [round(x, 6) for x in evals_m_top],
        },
        "cross_editor_variance_capture_by_rank": cross_editor_curves,
        "linear_probe_disentanglement": probe_disentanglement,
    }

    out_file = os.path.join(OUT_DIR, "nsd_v3_theorem2_and_geometry.json")
    with open(out_file, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"=== SAVED {out_file} ===")
    print("  Theorem 2 exponents (ideal=1.00):", out_payload["theorem2_empirical_verification"]["loglog_exponent_vs_eff_inv_sqrt_n"])
    print("  test_confirm probes:", probe_disentanglement["test_confirm"])
    print("  test_id rank-24 capture:", cross_editor_curves["test_id"]["24"])


if __name__ == "__main__":
    main()

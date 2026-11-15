import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src.cedar_model import ParameterMatchedDocMLP, DiscourseTrajectoryEncoder


def eps_unit_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Epsilon-safe Euclidean normalization: x / max(||x||_2, eps).
    Guarantees ||eps_unit_norm(x)||_2 <= 1 everywhere and exact unit norm whenever ||x||_2 >= eps.
    """
    norms = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True).clamp(min=eps)
    return x / norms


class CosineMarginOriginHead(nn.Module):
    """
    Bounded cosine-margin origin scorer:
      ell_orig(z) = kappa * <w_orig / max(||w_orig||_2, eps), z / max(||z||_2, eps)> + b_orig.
    Eliminates unconstrained weight-norm growth and cross-domain feature-norm drift.
    """
    def __init__(self, dim: int, kappa: float = 12.0, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.kappa = float(kappa)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(float(dim)))

    def normalized_weight(self) -> torch.Tensor:
        return eps_unit_norm(self.weight, eps=self.eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_hat = eps_unit_norm(z, eps=self.eps)
        w_hat = self.normalized_weight()
        return self.kappa * torch.mv(z_hat, w_hat) + self.bias


class CEDARv2Detector(nn.Module):
    """
    CEDAR v2: Compact Unit-Sphere Representation + Discourse Trajectory Kinematics (DTE)
    + Split-Pool Exact Editorial Nuisance-Subspace Deflation (NSD) + Bounded Cosine-Margin Head.

    Key theoretical & architectural guarantees:
      1. Compact unit-bounded trunk h_hat in R^d (d=128, ||h_hat||_2 <= 1) prevents 512-D subspace evasion
         and bounds every parent-octet difference operator ||A_i||_op <= 4 * max(1, lambda_sig).
      2. Exact equal-parent octet operator includes ALL three training edit levels (E1, E2, E3) in S_edit
         and ALL four levels (E0, E1, E2, E3) in S_orig:
           A_i = (1/6) sum_{o=0}^1 sum_{e=1}^3 delta_{i,o,e} delta_{i,o,e}^T
                 - (lambda_sig/4) sum_{e=0}^3 gamma_{i,e} gamma_{i,e}^T.
      3. Supports snapshot/restore of online train_fit EMA buffers so candidate checkpoints at each epoch
         can be finalized on strictly disjoint `train_op` octets (float64, eval mode, deterministic) BEFORE
         scoring `val_select`, preserving exact conditional independence (train_op _||_ {theta_1..theta_E})
         for Theorem 2 AND eliminating EMA-to-exact selection mismatch.
    """
    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-large",
        backbone_revision: str = None,
        proj_dim: int = 128,
        use_dte: bool = True,
        use_param_matched_mlp: bool = False,
        nsd_rank: int = 24,
        nsd_signal_alpha: float = 0.5,
        nsd_tau: float = 1e-4,
        nsd_ema_beta: float = 0.08,
        cosine_kappa: float = 12.0,
        norm_eps: float = 1e-6,
        num_edit_classes: int = 4,
        num_reg_classes: int = 3,
        use_cosine_head: bool = True,
        joint_8class: bool = False,
        nsd_rank_mode: str = "fixed",
    ):
        super().__init__()
        self.cfg = AutoConfig.from_pretrained(backbone_name, revision=backbone_revision)
        if hasattr(self.cfg, "reference_compile"):
            self.cfg.reference_compile = False
        try:
            self.backbone = AutoModel.from_pretrained(
                backbone_name, config=self.cfg, revision=backbone_revision, attn_implementation="sdpa", dtype=torch.float32
            ).float()
        except Exception:
            self.backbone = AutoModel.from_pretrained(
                backbone_name, config=self.cfg, revision=backbone_revision, dtype=torch.float32
            ).float()
        d_b = self.cfg.hidden_size

        self.proj_dim = proj_dim
        self.use_dte = use_dte
        self.use_param_matched_mlp = use_param_matched_mlp
        self.nsd_rank = nsd_rank
        # "fixed": always the top `nsd_rank` directions; "gap": r* = argmax eigengap among the top `nsd_rank`
        # positive eigenvalues of the protected operator (E2, docs/E2_plan_2026-11-15.md)
        assert nsd_rank_mode in ("fixed", "gap")
        self.nsd_rank_mode = nsd_rank_mode
        self.last_selected_rank = 0
        self.nsd_signal_alpha = float(nsd_signal_alpha)
        self.nsd_tau = float(nsd_tau)
        self.nsd_ema_beta = float(nsd_ema_beta)
        self.cosine_kappa = float(cosine_kappa)
        self.norm_eps = float(norm_eps)
        self.num_edit_classes = num_edit_classes
        self.num_reg_classes = num_reg_classes
        self.use_cosine_head = use_cosine_head
        self.joint_8class = joint_8class

        self.ctx_proj = nn.Sequential(
            nn.Linear(d_b * 2, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        ref_dte = DiscourseTrajectoryEncoder(feat_dim=7, gru_hidden=64, out_dim=proj_dim)
        dte_param_cnt = sum(p.numel() for p in ref_dte.parameters())

        if use_dte:
            self.dte = ref_dte
            self.aux_fuse = nn.Linear(proj_dim, proj_dim, bias=False)
        elif use_param_matched_mlp:
            self.doc_mlp = ParameterMatchedDocMLP(doc_dim=d_b * 2, out_dim=proj_dim, target_params=dte_param_cnt)
            mlp_param_cnt = sum(p.numel() for p in self.doc_mlp.parameters())
            rel_diff = abs(mlp_param_cnt - dte_param_cnt) / float(dte_param_cnt)
            assert rel_diff < 0.03, f"Param mismatch: DTE={dte_param_cnt} vs MLP={mlp_param_cnt}"
            self.aux_fuse = nn.Linear(proj_dim, proj_dim, bias=False)
        else:
            self.aux_fuse = None

        self.fuse_norm = nn.LayerNorm(proj_dim)

        # Moment & gated shrinker buffers (persisted in state_dict for exact checkpoint reload)
        self.register_buffer("ema_sec_edit", torch.zeros(proj_dim, proj_dim, dtype=torch.float32))
        self.register_buffer("ema_sec_orig", torch.zeros(proj_dim, proj_dim, dtype=torch.float32))
        self.register_buffer("proj_nuis", torch.zeros(proj_dim, proj_dim, dtype=torch.float32))
        self.register_buffer("nsd_edit_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("nsd_orig_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("nsd_is_exact_finalized", torch.tensor(0, dtype=torch.long))

        if joint_8class:
            self.joint_head = nn.Linear(proj_dim, 2 * num_edit_classes)
        else:
            if use_cosine_head:
                self.origin_head = CosineMarginOriginHead(proj_dim, kappa=cosine_kappa, eps=norm_eps)
            else:
                self.origin_head = nn.Linear(proj_dim, 1)

            # Auxiliary editorial heads: routed through z_nuis when NSD is active, or h_hat for multitask/supcon
            self.edit_cls_head = nn.Linear(proj_dim, num_edit_classes)
            self.reg_cls_head = nn.Linear(proj_dim, num_reg_classes)

            # Standardized post-encoder diagnostic edit-intensity probe on detached h_hat
            self.edit_intensity_head = nn.Sequential(
                nn.Linear(proj_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

    @staticmethod
    def _compute_batch_octet_moments_fp64(
        h_hat: torch.Tensor,
        parent_idx: torch.Tensor,
        origin_y: torch.Tensor,
        edit_lv: torch.Tensor,
    ):
        """
        Computes exact equal-parent second moments over complete octets present in (h_hat, parent_idx, origin_y, edit_lv):
          S_edit,i = (1/6) sum_{o in {0,1}} sum_{e in {1,2,3}} (h_{i,o,e} - h_{i,o,0})(...)^T
          S_orig,i = (1/4) sum_{e in {0,1,2,3}} (h_{i,1,e} - h_{i,0,e})(...)^T
        Accumulated in float64 for numerical precision. Returns (S_edit_mean, S_orig_mean, num_complete_parents).
        """
        h64 = h_hat.detach().double()
        D = h64.shape[1]
        sec_e_sum = torch.zeros(D, D, device=h64.device, dtype=torch.float64)
        sec_o_sum = torch.zeros(D, D, device=h64.device, dtype=torch.float64)
        n_complete = 0

        for p in torch.unique(parent_idx):
            pm = (parent_idx == p)
            if int(pm.sum().item()) < 8:
                continue
            cell_vecs = {}
            ok = True
            for ov in (0, 1):
                for ev in (0, 1, 2, 3):
                    idx = torch.nonzero(pm & (origin_y == ov) & (edit_lv == ev), as_tuple=False).flatten()
                    if idx.numel() == 0:
                        ok = False
                        break
                    cell_vecs[(ov, ev)] = h64[idx[0]]
                if not ok:
                    break
            if not ok:
                continue

            deltas_i = []
            for ov in (0, 1):
                h0 = cell_vecs[(ov, 0)]
                for ev in (1, 2, 3):
                    deltas_i.append(cell_vecs[(ov, ev)] - h0)
            gammas_i = [cell_vecs[(1, ev)] - cell_vecs[(0, ev)] for ev in (0, 1, 2, 3)]

            D_i = torch.stack(deltas_i, dim=0)  # (6, D)
            G_i = torch.stack(gammas_i, dim=0)  # (4, D)
            sec_e_sum.add_((D_i.T @ D_i) / 6.0)
            sec_o_sum.add_((G_i.T @ G_i) / 4.0)
            n_complete += 1

        if n_complete == 0:
            return None, None, 0
        return sec_e_sum / float(n_complete), sec_o_sum / float(n_complete), n_complete

    def _build_shrinker_from_moments_fp64(self, sec_edit_64: torch.Tensor, sec_orig_64: torch.Tensor):
        M_nuis = sec_edit_64 - self.nsd_signal_alpha * sec_orig_64
        M_nuis = 0.5 * (M_nuis + M_nuis.T)
        evals, evecs = torch.linalg.eigh(M_nuis)  # ascending in float64
        D = M_nuis.shape[0]
        r = min(self.nsd_rank, D)
        if self.nsd_rank_mode == "gap" and r >= 1:
            desc = evals.flip(0)
            n_pos = int((desc[:r] > 0).sum().item())
            if n_pos == 0:
                r = 1
            else:
                gaps = desc[:n_pos] - desc[1:n_pos + 1]
                r = int(torch.argmax(gaps).item()) + 1
        self.last_selected_rank = int(r)
        top_vals = evals[-r:].clamp(min=0.0)
        top_vecs = evecs[:, -r:]
        gates = top_vals / (top_vals + self.nsd_tau)
        P_N = (top_vecs * gates.unsqueeze(0)) @ top_vecs.T
        P_N = 0.5 * (P_N + P_N.T)
        mu_r = float(evals[-r].item()) if r >= 1 else 0.0
        mu_next = float(evals[-r - 1].item()) if r < D else -1e9
        return P_N.float(), evals, mu_r, (mu_r - mu_next)

    @torch.no_grad()
    def update_nsd_subspace_post_step(
        self,
        h_hat_detached: torch.Tensor,
        parent_idx: torch.Tensor,
        origin_y: torch.Tensor,
        edit_lv: torch.Tensor,
        warmup_updates: int = 10,
    ):
        """
        Online training heuristic on `train_fit` ONLY, called strictly after backward() and optimizer.step().
        Uses exact equal-parent E1..E3 / E0..E3 moment definitions on unit-bounded h_hat.
        """
        if self.nsd_rank <= 0:
            return
        se_64, so_64, n_comp = self._compute_batch_octet_moments_fp64(h_hat_detached, parent_idx, origin_y, edit_lv)
        if n_comp < 2:
            return

        beta = 1.0 if int(self.nsd_edit_updates.item()) == 0 else self.nsd_ema_beta
        self.ema_sec_edit.mul_(1.0 - beta).add_(se_64.float(), alpha=beta)
        self.ema_sec_orig.mul_(1.0 - beta).add_(so_64.float(), alpha=beta)
        self.nsd_edit_updates.add_(1)
        self.nsd_orig_updates.add_(1)
        self.nsd_is_exact_finalized.zero_()

        if int(self.nsd_edit_updates.item()) >= warmup_updates:
            P_N, _, _, _ = self._build_shrinker_from_moments_fp64(
                self.ema_sec_edit.double(), self.ema_sec_orig.double()
            )
            if torch.isfinite(P_N).all():
                self.proj_nuis.copy_(P_N)

    @torch.no_grad()
    def snapshot_nsd_buffers(self) -> dict:
        return {
            "ema_sec_edit": self.ema_sec_edit.detach().clone(),
            "ema_sec_orig": self.ema_sec_orig.detach().clone(),
            "proj_nuis": self.proj_nuis.detach().clone(),
            "nsd_edit_updates": self.nsd_edit_updates.detach().clone(),
            "nsd_orig_updates": self.nsd_orig_updates.detach().clone(),
            "nsd_is_exact_finalized": self.nsd_is_exact_finalized.detach().clone(),
        }

    @torch.no_grad()
    def restore_nsd_buffers(self, snap: dict):
        self.ema_sec_edit.copy_(snap["ema_sec_edit"])
        self.ema_sec_orig.copy_(snap["ema_sec_orig"])
        self.proj_nuis.copy_(snap["proj_nuis"])
        self.nsd_edit_updates.copy_(snap["nsd_edit_updates"])
        self.nsd_orig_updates.copy_(snap["nsd_orig_updates"])
        self.nsd_is_exact_finalized.copy_(snap["nsd_is_exact_finalized"])

    @torch.inference_mode()
    def finalize_exact_nsd_from_pool(self, op_tensor_dict: dict, batch_size: int = 128) -> dict:
        """
        Computes the EXACT equal-parent empirical operator M_hat_{N_op} in float64 on a frozen trunk
        in deterministic eval() mode over all complete parent octets in `op_tensor_dict` (e.g., `train_op`),
        installs the exact gated top-r shrinker P_N, and returns spectral & residual-norm diagnostics.
        """
        if self.nsd_rank <= 0:
            return {"active": False}

        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device
        N_rows = op_tensor_dict["input_ids"].shape[0]
        h_chunks = []
        for st in range(0, N_rows, batch_size):
            ed = min(st + batch_size, N_rows)
            h_hat, _, _ = self.encode(
                input_ids=op_tensor_dict["input_ids"][st:ed].to(dev),
                attention_mask=op_tensor_dict["attention_mask"][st:ed].to(dev),
                sent_ids=op_tensor_dict["sent_ids"][st:ed].to(dev),
                sent_novelty=op_tensor_dict["sent_novelty"][st:ed].to(dev),
            )
            h_chunks.append(h_hat.cpu())

        H_all = torch.cat(h_chunks, dim=0).to(dev)
        p_idx = op_tensor_dict["parent_idx"].to(dev)
        y_o = op_tensor_dict["origin_y"].to(dev)
        e_lv = op_tensor_dict["edit_lv"].to(dev)

        se_64, so_64, n_octets = self._compute_batch_octet_moments_fp64(H_all, p_idx, y_o, e_lv)
        assert n_octets >= 10, f"Expected >=10 complete octets in operator pool, got {n_octets}"

        P_N, evals_64, mu_r, eigengap_r = self._build_shrinker_from_moments_fp64(se_64, so_64)
        assert torch.isfinite(P_N).all(), "Non-finite exact P_N!"

        self.ema_sec_edit.copy_(se_64.float())
        self.ema_sec_orig.copy_(so_64.float())
        self.proj_nuis.copy_(P_N)
        self.nsd_edit_updates.fill_(n_octets)
        self.nsd_orig_updates.fill_(n_octets)
        self.nsd_is_exact_finalized.fill_(1)

        # Verify residual norms ||(I - P_N) h_hat||_2 on the operator pool
        Z_orig_pool = H_all - (H_all @ P_N)
        res_norms = torch.linalg.vector_norm(Z_orig_pool, ord=2, dim=-1)

        P64 = P_N.double()
        eff_op = 2.0 * P64 - (P64 @ P64)
        tr_e = float(torch.trace(se_64).item())
        tr_o = float(torch.trace(so_64).item())
        rem_e = float(torch.sum(eff_op * se_64).item())
        rem_o = float(torch.sum(eff_op * so_64).item())

        if was_training:
            self.train()

        return {
            "active": True,
            "exact_finalized": True,
            "num_complete_octets": int(n_octets),
            "rank": int(self.nsd_rank),
            "selected_rank": int(self.last_selected_rank),
            "mu_r": round(mu_r, 6),
            "eigengap_r": round(eigengap_r, 6),
            "top5_evals": [round(float(x), 6) for x in evals_64.flip(0)[:5].cpu().tolist()],
            "frac_edit_energy_removed": round(rem_e / max(1e-12, tr_e), 4),
            "frac_orig_energy_removed": round(rem_o / max(1e-12, tr_o), 4),
            "min_residual_norm": round(float(res_norms.min().item()), 5),
            "p05_residual_norm": round(float(torch.quantile(res_norms, 0.05).item()), 5),
            "mean_residual_norm": round(float(res_norms.mean().item()), 5),
        }

    @torch.no_grad()
    def get_nsd_diagnostics(self) -> dict:
        if self.nsd_rank <= 0 or int(self.nsd_edit_updates.item()) == 0:
            return {"active": False}
        P = self.proj_nuis.double()
        se = self.ema_sec_edit.double()
        so = self.ema_sec_orig.double()
        eff_op = 2.0 * P - (P @ P)
        tr_e = float(torch.trace(se).item())
        tr_o = float(torch.trace(so).item())
        rem_e = float(torch.sum(eff_op * se).item())
        rem_o = float(torch.sum(eff_op * so).item())
        return {
            "active": True,
            "exact_finalized": bool(int(self.nsd_is_exact_finalized.item()) == 1),
            "octets_or_updates": int(self.nsd_edit_updates.item()),
            "frac_edit_energy_removed": round(rem_e / max(1e-12, tr_e), 4),
            "frac_orig_energy_removed": round(rem_o / max(1e-12, tr_o), 4),
        }

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sent_ids: torch.Tensor = None,
        sent_novelty: torch.Tensor = None,
    ):
        use_amp = input_ids.is_cuda
        with torch.amp.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            tok_states = out.last_hidden_state

        tok_f = tok_states.float()
        valid_mask = (attention_mask > 0)
        mask_f = valid_mask.unsqueeze(-1).float()
        clean_tok = torch.where(valid_mask.unsqueeze(-1), tok_f, torch.zeros_like(tok_f))

        mean_pool = clean_tok.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        has_any_tok = (valid_mask.sum(dim=1, keepdim=True) > 0)
        max_raw = tok_f.masked_fill(~valid_mask.unsqueeze(-1), -1e9).max(dim=1).values
        max_pool = torch.where(has_any_tok, max_raw, torch.zeros_like(max_raw))
        doc_cat = torch.cat([mean_pool, max_pool], dim=-1)

        h_ctx = self.ctx_proj(doc_cat)
        if self.use_dte:
            assert sent_ids is not None and sent_novelty is not None
            s_embs, s_valid, s_lens = DiscourseTrajectoryEncoder.pool_sentence_spans(
                tok_f, attention_mask, sent_ids, max_sents=sent_novelty.shape[1]
            )
            h_disc = self.dte(s_embs, s_valid, s_lens, sent_novelty)
            h_raw = self.fuse_norm(h_ctx + self.aux_fuse(h_disc))
        elif self.use_param_matched_mlp:
            h_mlp = self.doc_mlp(doc_cat)
            h_raw = self.fuse_norm(h_ctx + self.aux_fuse(h_mlp))
        else:
            h_raw = self.fuse_norm(h_ctx)

        h_hat = eps_unit_norm(h_raw, eps=self.norm_eps)

        if self.nsd_rank > 0:
            P_snap = self.proj_nuis.detach().clone()
            z_nuis = h_hat @ P_snap
            z_orig = h_hat - z_nuis
        else:
            z_nuis = torch.zeros_like(h_hat)
            z_orig = h_hat

        return h_hat, z_orig, z_nuis

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sent_ids: torch.Tensor = None,
        sent_novelty: torch.Tensor = None,
    ):
        h_hat, z_orig, z_nuis = self.encode(input_ids, attention_mask, sent_ids, sent_novelty)
        if self.joint_8class:
            K = self.num_edit_classes
            j_logits = self.joint_head(eps_unit_norm(z_orig, eps=self.norm_eps))
            origin_logit = torch.logsumexp(j_logits[:, K:], dim=-1) - torch.logsumexp(j_logits[:, :K], dim=-1)
            return {
                "h_hat": h_hat,
                "z_orig": z_orig,
                "z_nuis": z_nuis,
                "joint_logits": j_logits,
                "origin_logit": origin_logit,
                "edit_intensity": torch.zeros_like(origin_logit),
            }

        if self.use_cosine_head:
            origin_logit = self.origin_head(z_orig)
        else:
            origin_logit = self.origin_head(z_orig).squeeze(-1)

        # When NSD projector is active, route editorial supervision strictly through z_nuis = h_hat @ P_snap!
        has_active_proj = (self.nsd_rank > 0) and bool((self.proj_nuis.abs().max() > 1e-12).item())
        aux_in = z_nuis if has_active_proj else h_hat
        edit_logits = self.edit_cls_head(aux_in)
        reg_logits = self.reg_cls_head(aux_in)

        edit_intensity = torch.sigmoid(self.edit_intensity_head(h_hat.detach()).squeeze(-1))
        return {
            "h_hat": h_hat,
            "z_orig": z_orig,
            "z_nuis": z_nuis,
            "origin_logit": origin_logit,
            "edit_logits": edit_logits,
            "reg_logits": reg_logits,
            "edit_intensity": edit_intensity,
        }

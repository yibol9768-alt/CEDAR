import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoConfig, AutoModel


class ParameterMatchedDocMLP(nn.Module):
    """
    Parameter-matched document-level MLP control for DiscourseTrajectoryEncoder (DTE).
    Operates strictly on pooled document hidden state doc_cat (no sentence-level trajectory features),
    automatically sizing its bottleneck hidden dimension to match DTE's trainable parameter count within <0.2%.
    """
    def __init__(self, doc_dim: int, out_dim: int = 128, target_params: int = 122127):
        super().__init__()
        # Solve (doc_dim + 1) * h + (h + 1) * out_dim + 2 * out_dim ~= target_params
        denom = doc_dim + out_dim + 1
        rem = max(denom, target_params - 3 * out_dim)
        hidden_dim = max(8, int(round(rem / float(denom))))
        self.net = nn.Sequential(
            nn.Linear(doc_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, doc_cat: torch.Tensor) -> torch.Tensor:
        return self.net(doc_cat)


class DiscourseTrajectoryEncoder(nn.Module):
    """
    Single-pass Contextual Discourse-Trajectory Kinematics Encoder (DTE).
    Computes inter-sentence geometric progression features over float32 span-pooled contextual states,
    packs valid contiguous transition sequences so padded suffixes never affect bidirectional GRU states,
    and explicitly handles single-sentence (zero-transition) documents.
    """
    def __init__(self, feat_dim: int = 7, gru_hidden: int = 64, out_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.gru_hidden = gru_hidden
        self.out_dim = out_dim
        self.in_norm = nn.LayerNorm(feat_dim)
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.attn_w = nn.Linear(gru_hidden * 2, 1)
        self.proj = nn.Sequential(
            nn.Linear(gru_hidden * 2 + feat_dim * 3, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(out_dim),
        )
        self.zero_trans_fallback = nn.Parameter(torch.zeros(1, out_dim))

    @staticmethod
    def pool_sentence_spans(tok_states: torch.Tensor, attention_mask: torch.Tensor, sent_ids: torch.Tensor, max_sents: int = 16):
        """
        Float32 masked span pooling without NaN*0 hazards.
        Sentences per document are indexed contiguously 0 .. m_b - 1 over retained non-padding tokens.
        """
        B, L, D = tok_states.shape
        tok_f = tok_states.float()
        valid_tok = (attention_mask > 0) & (sent_ids >= 0) & (sent_ids < max_sents)
        safe_idx = torch.where(valid_tok, sent_ids, torch.zeros_like(sent_ids)).clamp(0, max_sents - 1)
        clean_states = torch.where(valid_tok.unsqueeze(-1), tok_f, torch.zeros_like(tok_f))

        sent_embs = torch.zeros(B, max_sents, D, device=tok_states.device, dtype=torch.float32)
        sent_embs.scatter_add_(1, safe_idx.unsqueeze(-1).expand(-1, -1, D), clean_states)

        sent_counts = torch.zeros(B, max_sents, device=tok_states.device, dtype=torch.float32)
        sent_counts.scatter_add_(1, safe_idx, valid_tok.float())

        sent_valid = sent_counts > 0.5
        sent_embs = sent_embs / sent_counts.clamp(min=1.0).unsqueeze(-1)
        return sent_embs, sent_valid, sent_counts

    def compute_kinematics(self, sent_embs: torch.Tensor, sent_valid: torch.Tensor, sent_lens: torch.Tensor, sent_novelty: torch.Tensor):
        B, M, D = sent_embs.shape
        assert M >= 2, "max_sents must be >= 2"
        trans_mask = sent_valid[:, :-1] & sent_valid[:, 1:]  # (B, M-1)

        e_curr = sent_embs[:, :-1, :]
        e_next = sent_embs[:, 1:, :]
        u_curr = F.normalize(e_curr, p=2, dim=-1, eps=1e-6)
        u_next = F.normalize(e_next, p=2, dim=-1, eps=1e-6)

        cos_step = 1.0 - (u_curr * u_next).sum(dim=-1)
        delta = e_next - e_curr
        euclid_step = torch.norm(delta, p=2, dim=-1) / math.sqrt(float(D))

        u_delta = F.normalize(delta, p=2, dim=-1, eps=1e-6)
        if M > 2:
            turn_inner = 1.0 - (u_delta[:, :-1, :] * u_delta[:, 1:, :]).sum(dim=-1)
            valid_turn = trans_mask[:, :-1] & trans_mask[:, 1:]
            turn_inner = torch.where(valid_turn, turn_inner, torch.zeros_like(turn_inner))
            turn_angle = torch.cat([torch.zeros(B, 1, device=sent_embs.device, dtype=torch.float32), turn_inner], dim=1)
        else:
            turn_angle = torch.zeros(B, 1, device=sent_embs.device, dtype=torch.float32)

        l_curr = torch.log1p(sent_lens[:, :-1])
        l_ratio = torch.log((sent_lens[:, 1:] + 1.0) / (sent_lens[:, :-1] + 1.0))
        nov_next = sent_novelty[:, 1:].float()

        total_sents = sent_valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        pos_idx = torch.arange(1, M, device=sent_embs.device, dtype=torch.float32).unsqueeze(0) / total_sents

        kin = torch.stack([cos_step, euclid_step, turn_angle, l_curr, l_ratio, nov_next, pos_idx], dim=-1)
        kin = torch.where(trans_mask.unsqueeze(-1), kin, torch.zeros_like(kin))
        return kin, trans_mask

    def forward(self, sent_embs: torch.Tensor, sent_valid: torch.Tensor, sent_lens: torch.Tensor, sent_novelty: torch.Tensor) -> torch.Tensor:
        kin, trans_mask = self.compute_kinematics(sent_embs, sent_valid, sent_lens, sent_novelty)
        B, T, F_dim = kin.shape
        lengths = trans_mask.long().sum(dim=1)  # (B,)
        has_trans = lengths > 0

        out = self.zero_trans_fallback.expand(B, -1).clone()
        if not has_trans.any():
            return out

        idx_valid = torch.nonzero(has_trans, as_tuple=False).flatten()
        kin_sub = kin[idx_valid]                # (B_v, T, F)
        mask_sub = trans_mask[idx_valid]        # (B_v, T)
        len_sub = lengths[idx_valid]            # (B_v,)

        # Compact active transitions per item to strictly contiguous prefix 0..len_sub[b]-1
        B_v = kin_sub.shape[0]
        max_t = int(len_sub.max().item())
        compact_kin = torch.zeros(B_v, max_t, F_dim, device=kin.device, dtype=torch.float32)
        compact_mask = torch.zeros(B_v, max_t, device=kin.device, dtype=torch.bool)
        for b in range(B_v):
            active_rows = kin_sub[b][mask_sub[b]]
            L_b = active_rows.shape[0]
            compact_kin[b, :L_b] = active_rows
            compact_mask[b, :L_b] = True

        m_float = compact_mask.unsqueeze(-1).float()
        denom = m_float.sum(dim=1).clamp(min=1.0)
        k_mean = (compact_kin * m_float).sum(dim=1) / denom
        k_var = (((compact_kin - k_mean.unsqueeze(1)) ** 2) * m_float).sum(dim=1) / denom
        # Only compute std across >=2 transitions; 0 for single-transition docs
        k_std = torch.where((len_sub >= 2).unsqueeze(-1), torch.sqrt(k_var + 1e-8), torch.zeros_like(k_mean))
        k_max = compact_kin.masked_fill(~compact_mask.unsqueeze(-1), -1e9).max(dim=1).values
        k_min = compact_kin.masked_fill(~compact_mask.unsqueeze(-1), 1e9).min(dim=1).values
        k_range = torch.where((len_sub >= 2).unsqueeze(-1), k_max - k_min, torch.zeros_like(k_mean))

        kin_norm = torch.where(compact_mask.unsqueeze(-1), self.in_norm(compact_kin), torch.zeros_like(compact_kin))
        packed = pack_padded_sequence(kin_norm, len_sub.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=max_t)

        attn_logits = self.attn_w(gru_out).squeeze(-1).masked_fill(~compact_mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=-1)
        seq_summary = (gru_out * attn_weights.unsqueeze(-1)).sum(dim=1)

        sub_out = self.proj(torch.cat([seq_summary, k_mean, k_std, k_range], dim=-1))
        out[idx_valid] = sub_out
        return out


class CEDARDetector(nn.Module):
    """
    CEDAR + Controlled Baselines Unified Model.
    Maintains float32 master weights (using bf16 autocast inside backbone forward) and updates
    Paired Editorial Nuisance-Subspace Deflation (NSD) strictly post-backward/step to preserve autograd safety.
    """
    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-large",
        proj_dim: int = 512,
        use_dte: bool = True,
        use_param_matched_mlp: bool = False,
        nsd_rank: int = 32,
        nsd_signal_alpha: float = 0.5,
        nsd_tau: float = 0.02,
        nsd_ema_beta: float = 0.05,
        num_edit_classes: int = 4,
        joint_8class: bool = False,
    ):
        super().__init__()
        self.cfg = AutoConfig.from_pretrained(backbone_name)
        if hasattr(self.cfg, "reference_compile"):
            self.cfg.reference_compile = False
        # Explicit float32 master weights (critical for checkpoints like deberta-v3-large stored in fp16 on Hub)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=self.cfg, dtype=torch.float32).float()
        d_b = self.cfg.hidden_size

        self.use_dte = use_dte
        self.use_param_matched_mlp = use_param_matched_mlp
        self.nsd_rank = nsd_rank
        self.nsd_signal_alpha = nsd_signal_alpha
        self.nsd_tau = nsd_tau
        self.nsd_ema_beta = nsd_ema_beta
        self.num_edit_classes = num_edit_classes
        self.joint_8class = joint_8class
        self.proj_dim = proj_dim

        self.ctx_proj = nn.Sequential(
            nn.Linear(d_b * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(proj_dim),
        )

        ref_dte = DiscourseTrajectoryEncoder(feat_dim=7, gru_hidden=64, out_dim=128)
        dte_param_cnt = sum(p.numel() for p in ref_dte.parameters())

        if use_dte:
            self.dte = ref_dte
            self.aux_fuse = nn.Linear(128, proj_dim, bias=False)
        elif use_param_matched_mlp:
            self.doc_mlp = ParameterMatchedDocMLP(doc_dim=d_b * 2, out_dim=128, target_params=dte_param_cnt)
            mlp_param_cnt = sum(p.numel() for p in self.doc_mlp.parameters())
            rel_diff = abs(mlp_param_cnt - dte_param_cnt) / float(dte_param_cnt)
            assert rel_diff < 0.02, f"Param mismatch: DTE={dte_param_cnt} vs MLP={mlp_param_cnt}"
            self.aux_fuse = nn.Linear(128, proj_dim, bias=False)
        else:
            self.aux_fuse = None

        self.fuse_norm = nn.LayerNorm(proj_dim)

        # Uncentered paired-difference second-moment buffers + frozen-per-step projector
        self.register_buffer("ema_sec_edit", torch.zeros(proj_dim, proj_dim, dtype=torch.float32))
        self.register_buffer("ema_sec_orig", torch.zeros(proj_dim, proj_dim, dtype=torch.float32))
        self.register_buffer("proj_nuis", torch.zeros(proj_dim, proj_dim, dtype=torch.float32))
        self.register_buffer("nsd_edit_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("nsd_orig_updates", torch.tensor(0, dtype=torch.long))

        if joint_8class:
            self.joint_head = nn.Linear(proj_dim, 2 * num_edit_classes)
        else:
            self.origin_head = nn.Linear(proj_dim, 1)
            self.edit_cls_head = nn.Linear(proj_dim, num_edit_classes)
            # Standardized post-encoder diagnostic edit-intensity probe on detached h
            self.edit_intensity_head = nn.Sequential(
                nn.Linear(proj_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

    @torch.no_grad()
    def update_nsd_subspace_post_step(
        self,
        h_detached: torch.Tensor,
        parent_idx: torch.Tensor,
        origin_y: torch.Tensor,
        edit_lv: torch.Tensor,
        warmup_updates: int = 15,
    ):
        """
        MUST be called AFTER loss.backward() and optimizer.step() using detached h.
        Maintains separate valid-update counters for S_edit and S_orig so EMA initialization
        always gives full weight 1.0 to the first valid batch of pairs.
        """
        if self.nsd_rank <= 0:
            return

        h_det = h_detached.detach().float()
        D = h_det.shape[1]
        unique_parents = torch.unique(parent_idx)

        deltas = []
        gammas = []
        for p in unique_parents:
            p_mask = (parent_idx == p)
            for o_val in (0, 1):
                po_mask = p_mask & (origin_y == o_val)
                e0_idx = torch.nonzero(po_mask & (edit_lv == 0), as_tuple=False).flatten()
                if e0_idx.numel() > 0:
                    h0 = h_det[e0_idx[0]]
                    for ev in (1, 2):
                        ek_idx = torch.nonzero(po_mask & (edit_lv == ev), as_tuple=False).flatten()
                        if ek_idx.numel() > 0:
                            deltas.append(h_det[ek_idx[0]] - h0)
            for ev in (0, 1, 2):
                i0 = torch.nonzero(p_mask & (origin_y == 0) & (edit_lv == ev), as_tuple=False).flatten()
                i1 = torch.nonzero(p_mask & (origin_y == 1) & (edit_lv == ev), as_tuple=False).flatten()
                if i0.numel() > 0 and i1.numel() > 0:
                    gammas.append(h_det[i1[0]] - h_det[i0[0]])

        updated_any = False
        if len(deltas) >= 4:
            delta_mat = torch.stack(deltas, dim=0)
            s_e = (delta_mat.T @ delta_mat) / float(delta_mat.shape[0])
            beta_e = 1.0 if int(self.nsd_edit_updates.item()) == 0 else self.nsd_ema_beta
            self.ema_sec_edit.mul_(1.0 - beta_e).add_(s_e, alpha=beta_e)
            self.nsd_edit_updates.add_(1)
            updated_any = True

        if len(gammas) >= 4:
            gamma_mat = torch.stack(gammas, dim=0)
            s_o = (gamma_mat.T @ gamma_mat) / float(gamma_mat.shape[0])
            beta_o = 1.0 if int(self.nsd_orig_updates.item()) == 0 else self.nsd_ema_beta
            self.ema_sec_orig.mul_(1.0 - beta_o).add_(s_o, alpha=beta_o)
            self.nsd_orig_updates.add_(1)
            updated_any = True

        ready = (int(self.nsd_edit_updates.item()) >= warmup_updates) and (
            self.nsd_signal_alpha == 0.0 or int(self.nsd_orig_updates.item()) >= warmup_updates
        )
        if updated_any and ready:
            M_nuis = self.ema_sec_edit - self.nsd_signal_alpha * self.ema_sec_orig
            M_nuis = 0.5 * (M_nuis + M_nuis.T)
            if torch.isfinite(M_nuis).all():
                evals, evecs = torch.linalg.eigh(M_nuis)
                r = min(self.nsd_rank, D)
                top_vals = evals[-r:].clamp(min=0.0)
                top_vecs = evecs[:, -r:]
                gates = top_vals / (top_vals + self.nsd_tau)
                P_N = (top_vecs * gates.unsqueeze(0)) @ top_vecs.T
                if torch.isfinite(P_N).all():
                    self.proj_nuis.copy_(P_N)

    @torch.no_grad()
    def get_nsd_diagnostics(self) -> dict:
        if self.nsd_rank <= 0 or int(self.nsd_edit_updates.item()) == 0:
            return {"active": False}
        P = self.proj_nuis
        # Exact second-moment reduction after h -> (I - P)h is tr((2P - P^2) S) / tr(S)
        eff_op = 2.0 * P - (P @ P)
        tr_e = float(torch.trace(self.ema_sec_edit).item())
        tr_o = float(torch.trace(self.ema_sec_orig).item())
        rem_e = float(torch.sum(eff_op * self.ema_sec_edit).item())
        rem_o = float(torch.sum(eff_op * self.ema_sec_orig).item())
        nuis_e = float(torch.sum((P @ P) * self.ema_sec_edit).item())
        return {
            "active": True,
            "edit_updates": int(self.nsd_edit_updates.item()),
            "orig_updates": int(self.nsd_orig_updates.item()),
            "frac_edit_energy_removed": round(rem_e / max(1e-8, tr_e), 4),
            "frac_orig_energy_removed": round(rem_o / max(1e-8, tr_o), 4),
            "frac_nuis_subspace_edit_energy": round(nuis_e / max(1e-8, tr_e), 4),
        }

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sent_ids: torch.Tensor = None, sent_novelty: torch.Tensor = None):
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
            h = self.fuse_norm(h_ctx + self.aux_fuse(h_disc))
        elif self.use_param_matched_mlp:
            h_mlp = self.doc_mlp(doc_cat)
            h = self.fuse_norm(h_ctx + self.aux_fuse(h_mlp))
        else:
            h = self.fuse_norm(h_ctx)

        if self.nsd_rank > 0:
            # Clone frozen snapshot for autograd graph independence
            P_snap = self.proj_nuis.detach().clone()
            z_nuis_vec = h @ P_snap
            z_orig = h - z_nuis_vec
        else:
            z_nuis_vec = torch.zeros_like(h)
            z_orig = h

        return h, z_orig, z_nuis_vec

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sent_ids: torch.Tensor = None, sent_novelty: torch.Tensor = None):
        h, z_orig, z_nuis_vec = self.encode(input_ids, attention_mask, sent_ids, sent_novelty)
        if self.joint_8class:
            K = self.num_edit_classes
            j_logits = self.joint_head(z_orig)
            origin_logit = torch.logsumexp(j_logits[:, K:], dim=-1) - torch.logsumexp(j_logits[:, :K], dim=-1)
            return {
                "h": h,
                "z_orig": z_orig,
                "joint_logits": j_logits,
                "origin_logit": origin_logit,
                "edit_intensity": torch.zeros_like(origin_logit),
            }
        origin_logit = self.origin_head(z_orig).squeeze(-1)
        edit_logits = self.edit_cls_head(h)
        # Standardized diagnostic edit-intensity probe on detached h
        edit_intensity = torch.sigmoid(self.edit_intensity_head(h.detach()).squeeze(-1))
        return {
            "h": h,
            "z_orig": z_orig,
            "origin_logit": origin_logit,
            "edit_logits": edit_logits,
            "edit_intensity": edit_intensity,
        }

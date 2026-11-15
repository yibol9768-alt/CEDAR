import os
import re
import gc
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = "/content/cedar_data"


def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def load_all_eval_sets():
    datasets = {}
    for sp in ["train", "val", "test_id", "test_ood_ar", "test_ood_llada"]:
        with open(os.path.join(DATA_DIR, f"split_{sp}.json")) as f:
            datasets[sp] = json.load(f)
    for ext in ["external_esl_full", "external_mixset", "external_beemo", "external_mage_wild"]:
        with open(os.path.join(DATA_DIR, f"{ext}.json")) as f:
            datasets[ext] = json.load(f)
    return datasets


def extract_surface_features(texts):
    feats = []
    for t in texts:
        chars = max(1, len(t))
        words = t.split()
        nw = max(1, len(words))
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        ns = max(1, len(sents))
        slen = [len(s.split()) for s in sents] if sents else [nw]
        feats.append([
            nw,
            chars / nw,
            nw / ns,
            float(np.std(slen)),
            t.count(",") / nw,
            t.count(";") / nw,
            (t.count("-") + t.count("—")) / nw,
            (t.count('"') + t.count("'") + t.count("”")) / nw,
            t.count("\n") / nw,
            sum(c.isdigit() for c in t) / chars,
            sum(c.isupper() for c in t) / chars,
            len(set(w.lower() for w in words)) / nw,
        ])
    return np.asarray(feats, dtype=np.float32)


def run_artifact_probes(datasets):
    print("\n=== [1/4] Running Surface, Length & Lexical Shortcut Control Probes ===")
    tr = datasets["train"]
    X_tr_txt = [r["text"] for r in tr]
    y_tr = np.array([r["origin_label"] for r in tr], dtype=np.int64)
    F_tr = extract_surface_features(X_tr_txt)

    tr_e0 = [r for r in tr if r["edit_level"] == "E0"]
    X_tr_e0_txt = [r["text"] for r in tr_e0]
    y_tr_e0 = np.array([r["origin_label"] for r in tr_e0], dtype=np.int64)

    clf_len = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    clf_len.fit(F_tr[:, :1], y_tr)

    clf_surf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    clf_surf.fit(F_tr, y_tr)

    clf_tfidf_e0 = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), LogisticRegression(max_iter=500))
    clf_tfidf_e0.fit(X_tr_e0_txt, y_tr_e0)

    clf_tfidf_all = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), LogisticRegression(max_iter=500))
    clf_tfidf_all.fit(X_tr_txt, y_tr)

    probe_preds = {
        "probe_length": {},
        "probe_surface_12feat": {},
        "probe_tfidf_e0": {},
        "probe_tfidf_aug": {},
    }

    for sp_name, rows in datasets.items():
        if sp_name == "train":
            continue
        txts = [r["text"] for r in rows]
        F_sp = extract_surface_features(txts)
        probe_preds["probe_length"][sp_name] = clf_len.predict_proba(F_sp[:, :1])[:, 1].tolist()
        probe_preds["probe_surface_12feat"][sp_name] = clf_surf.predict_proba(F_sp)[:, 1].tolist()
        probe_preds["probe_tfidf_e0"][sp_name] = clf_tfidf_e0.predict_proba(txts)[:, 1].tolist()
        probe_preds["probe_tfidf_aug"][sp_name] = clf_tfidf_all.predict_proba(txts)[:, 1].tolist()

    tid = datasets["test_id"]
    for lv in ["E0", "E1", "E2", "E3", "E4"]:
        idx = [i for i, r in enumerate(tid) if r["edit_level"] == lv]
        yv = [tid[i]["origin_label"] for i in idx]
        auc_len = safe_auroc(yv, [probe_preds["probe_length"]["test_id"][i] for i in idx])
        auc_surf = safe_auroc(yv, [probe_preds["probe_surface_12feat"]["test_id"][i] for i in idx])
        auc_tf = safe_auroc(yv, [probe_preds["probe_tfidf_aug"]["test_id"][i] for i in idx])
        print(f"  test_id {lv}: Length-AUROC={auc_len:.3f} | Surface12-AUROC={auc_surf:.3f} | TFIDF-Aug-AUROC={auc_tf:.3f}")

    return probe_preds


@torch.inference_mode()
def run_fast_detectgpt_and_binoculars(
    datasets,
    obs_id="Qwen/Qwen2.5-3B",
    perf_id="Qwen/Qwen2.5-3B-Instruct",
    batch_size=8,
):
    """
    Micro-batched, memory-safe, exact implementations:
      1) Fast-DetectGPT (same-model analytical conditional probability curvature on observer LM, Bao et al. ICLR 2024):
         Uses centered second moment sum_v p(v) * (log p(v) - mu)^2 to prevent catastrophic cancellation.
      2) Binoculars (Hans et al., ICML 2024, adapted to Qwen2.5-3B observer / Qwen2.5-3B-Instruct performer):
         B(x) = [ -mean_t log p_perf(x_t | x_<t) ] / [ -mean_t sum_v p_obs(v | x_<t) log p_perf(v | x_<t) ]
         Off-the-shelf detection score = -B(x) (lower perplexity-to-cross-perplexity ratio indicates MGT).
    """
    print(f"\n=== [2/4] Running Fast-DetectGPT + Binoculars ({obs_id} observer / {perf_id} performer) ===")
    tok = AutoTokenizer.from_pretrained(obs_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    obs_model = AutoModelForCausalLM.from_pretrained(obs_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    perf_model = AutoModelForCausalLM.from_pretrained(perf_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    fd_preds, bino_preds = {}, {}

    for sp_name, rows in datasets.items():
        if sp_name == "train":
            continue
        txts = [r["text"] for r in rows]
        fd_scores, bino_scores = [], []
        for st in range(0, len(txts), batch_size):
            batch_txt = txts[st:st + batch_size]
            enc = tok(batch_txt, return_tensors="pt", padding=True, truncation=True, max_length=384).to("cuda")
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            assert input_ids.shape[1] >= 2, f"Invalid sequence length <2 in {sp_name} at {st}"

            logits_obs = obs_model(input_ids=input_ids, attention_mask=attn).logits[:, :-1, :]
            logits_perf = perf_model(input_ids=input_ids, attention_mask=attn).logits[:, :-1, :]
            targets = input_ids[:, 1:]
            mask = attn[:, 1:].float()
            tok_cnt = mask.sum(dim=-1)
            assert float(tok_cnt.min().item()) >= 1.0, f"Zero valid next-token targets in {sp_name} at {st}"

            B_cur, T_cur, _ = logits_obs.shape
            diff_sum = torch.zeros(B_cur, device="cuda", dtype=torch.float32)
            var_sum = torch.zeros(B_cur, device="cuda", dtype=torch.float32)
            nll_perf_sum = torch.zeros(B_cur, device="cuda", dtype=torch.float32)
            xce_sum = torch.zeros(B_cur, device="cuda", dtype=torch.float32)

            # Sequence-chunked float32 vocabulary reduction (<250MB peak memory)
            for t0 in range(0, T_cur, 64):
                t1 = min(t0 + 64, T_cur)
                m_c = mask[:, t0:t1]
                tgt_c = targets[:, t0:t1]

                lp_obs = F.log_softmax(logits_obs[:, t0:t1, :].float(), dim=-1)
                p_obs = lp_obs.exp()
                obs_tok_lp = torch.gather(lp_obs, dim=-1, index=tgt_c.unsqueeze(-1)).squeeze(-1)
                exp_lp = (p_obs * lp_obs).sum(dim=-1)
                var_lp = (p_obs * ((lp_obs - exp_lp.unsqueeze(-1)) ** 2)).sum(dim=-1).clamp(min=1e-8)
                del lp_obs

                diff_sum += ((obs_tok_lp - exp_lp) * m_c).sum(dim=-1)
                var_sum += (var_lp * m_c).sum(dim=-1)

                lp_perf = F.log_softmax(logits_perf[:, t0:t1, :].float(), dim=-1)
                perf_tok_lp = torch.gather(lp_perf, dim=-1, index=tgt_c.unsqueeze(-1)).squeeze(-1)
                nll_perf_sum += ((-perf_tok_lp) * m_c).sum(dim=-1)
                xce_sum += ((-(p_obs * lp_perf).sum(dim=-1)) * m_c).sum(dim=-1)
                del p_obs, lp_perf

            del logits_obs, logits_perf
            fd_batch = diff_sum / torch.sqrt(var_sum.clamp(min=1e-8))
            fd_scores.extend(fd_batch.cpu().tolist())

            nll_perf = nll_perf_sum / tok_cnt
            xce_obs_perf = xce_sum / tok_cnt
            b_ratio = nll_perf / xce_obs_perf.clamp(min=1e-8)
            bino_scores.extend((-b_ratio).cpu().tolist())

        fd_preds[sp_name] = fd_scores
        bino_preds[sp_name] = bino_scores

    val_e0_idx = [i for i, r in enumerate(datasets["val"]) if r["edit_level"] == "E0"]
    y_ve0 = [datasets["val"][i]["origin_label"] for i in val_e0_idx]
    print(
        f"  Read-only val E0 check: Fast-DetectGPT AUROC={safe_auroc(y_ve0, [fd_preds['val'][i] for i in val_e0_idx]):.4f} "
        f"| Binoculars AUROC={safe_auroc(y_ve0, [bino_preds['val'][i] for i in val_e0_idx]):.4f}"
    )

    del obs_model, perf_model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return fd_preds, bino_preds


@torch.inference_mode()
def run_radar(datasets, batch_size=32):
    """
    TrustSafeAI/RADAR-Vicuna-7B documented checkpoint convention (Hu et al., NeurIPS 2023):
    index 0 ('LABEL_0') is the AI-generated probability; evaluated strictly off-the-shelf without label-based flipping.
    """
    model_id = "TrustSafeAI/RADAR-Vicuna-7B"
    print(f"\n=== [3/4] Running Off-the-Shelf RADAR ({model_id}) ===")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    ai_col = 0  # Documented TrustSafeAI convention: probs[:, 0] = P(AI-generated)

    radar_preds = {}
    for sp_name, rows in datasets.items():
        if sp_name == "train":
            continue
        scores = []
        txts = [r["text"] for r in rows]
        for st in range(0, len(txts), batch_size):
            enc = tok(txts[st:st + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            p = F.softmax(model(**enc).logits.float(), dim=-1)[:, ai_col]
            scores.extend(p.cpu().tolist())
        radar_preds[sp_name] = scores

    val_e0_idx = [i for i, r in enumerate(datasets["val"]) if r["edit_level"] == "E0"]
    y_ve0 = [datasets["val"][i]["origin_label"] for i in val_e0_idx]
    print(f"  Read-only val E0 check: RADAR (col=0) AUROC={safe_auroc(y_ve0, [radar_preds['val'][i] for i in val_e0_idx]):.4f}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return radar_preds


class DesklibDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        mask = (attention_mask > 0).unsqueeze(-1).float()
        clean_out = torch.where(mask > 0, out.float(), torch.zeros_like(out.float()))
        pooled = clean_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled).squeeze(-1)


@torch.inference_mode()
def run_desklib(datasets, batch_size=32):
    model_id = "desklib/ai-text-detector-v1.01"
    print(f"\n=== [4/4] Running Desklib RAID-#1 Detector ({model_id}) ===")
    cfg = AutoConfig.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    net = DesklibDetector(cfg)
    ckpt_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    sd = load_file(ckpt_path)

    # Strict verification of classifier + backbone weights!
    assert "classifier.weight" in sd and "classifier.bias" in sd, f"Missing classifier keys in {model_id}: {list(sd.keys())[:10]}"
    missing, unexpected = net.load_state_dict(sd, strict=False)
    allow_missing = {"model.embeddings.position_ids"}
    disallowed_missing = set(missing) - allow_missing
    assert len(disallowed_missing) == 0, f"CRITICAL missing keys in Desklib checkpoint: {disallowed_missing}"
    assert len(unexpected) == 0, f"CRITICAL unexpected keys in Desklib checkpoint: {unexpected}"
    print("  Desklib strict weight verification PASSED (0 disallowed missing, 0 unexpected)")

    net = net.to("cuda").eval()
    desk_preds = {}
    for sp_name, rows in datasets.items():
        if sp_name == "train":
            continue
        scores = []
        txts = [r["text"] for r in rows]
        for st in range(0, len(txts), batch_size):
            enc = tok(txts[st:st + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            logits = net(enc["input_ids"], enc["attention_mask"])
            probs = torch.sigmoid(logits).cpu().tolist()
            scores.extend(probs)
        desk_preds[sp_name] = scores

    val_e0_idx = [i for i, r in enumerate(datasets["val"]) if r["edit_level"] == "E0"]
    y_ve0 = [datasets["val"][i]["origin_label"] for i in val_e0_idx]
    print(f"  Read-only val E0 check: Desklib AUROC={safe_auroc(y_ve0, [desk_preds['val'][i] for i in val_e0_idx]):.4f}")

    del net, tok
    gc.collect()
    torch.cuda.empty_cache()
    return desk_preds


def main():
    t0 = time.time()
    datasets = load_all_eval_sets()
    all_preds = {}
    all_preds.update(run_artifact_probes(datasets))
    fd_p, bi_p = run_fast_detectgpt_and_binoculars(datasets)
    all_preds["fast_detectgpt_qwen3b"] = fd_p
    all_preds["binoculars_qwen3b"] = bi_p
    all_preds["radar_vicuna7b"] = run_radar(datasets)
    all_preds["desklib_raid_v1"] = run_desklib(datasets)

    out_path = os.path.join(DATA_DIR, "preds_pretrained_and_probes.json")
    with open(out_path, "w") as f:
        json.dump(all_preds, f)
    print(f"\n=== ALL PRETRAINED & PROBE PREDICTIONS SAVED ({time.time() - t0:.1f}s) -> {out_path} ===")


if __name__ == "__main__":
    main()

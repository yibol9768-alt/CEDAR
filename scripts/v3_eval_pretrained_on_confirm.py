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
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

V1_DIR = "/content/cedar_data"
V2_DIR = "/content/cedar_data_v2"
OUT_DIR = "/content/cedar_results_v3"
os.makedirs(OUT_DIR, exist_ok=True)


def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


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


def load_all_splits_v3():
    datasets = {
        "train_v2": json.load(open(os.path.join(V2_DIR, "split_train_v2.json"))),
        "test_confirm": json.load(open(os.path.join(V2_DIR, "split_test_confirm.json"))),
    }
    for sp in ["val", "test_id", "test_ood_ar", "test_ood_llada"]:
        datasets[sp] = json.load(open(os.path.join(V1_DIR, f"split_{sp}.json")))
    for ext in ["external_esl_full", "external_mixset", "external_beemo", "external_mage_wild"]:
        datasets[ext] = json.load(open(os.path.join(V1_DIR, f"{ext}.json")))
    return datasets


def run_probes_v2(datasets):
    print("\n=== [1/4] Fitting Surface, Length & TF-IDF Shortcut Probes on 800-Parent Bidirectional train_v2 ===")
    tr = datasets["train_v2"]
    X_tr_txt = [r["text"] for r in tr]
    y_tr = np.array([r["origin_label"] for r in tr], dtype=np.int64)
    F_tr = extract_surface_features(X_tr_txt)

    tr_e0 = [r for r in tr if r["edit_level"] == "E0"]
    X_tr_e0_txt = [r["text"] for r in tr_e0]
    y_tr_e0 = np.array([r["origin_label"] for r in tr_e0], dtype=np.int64)

    clf_len = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(F_tr[:, :1], y_tr)
    clf_surf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(F_tr, y_tr)
    clf_tfidf_e0 = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), LogisticRegression(max_iter=500)).fit(X_tr_e0_txt, y_tr_e0)
    clf_tfidf_all = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), LogisticRegression(max_iter=500)).fit(X_tr_txt, y_tr)

    probe_preds = {k: {} for k in ["probe_length", "probe_surface_12feat", "probe_tfidf_e0", "probe_tfidf_aug"]}
    for sp_name, rows in datasets.items():
        if sp_name == "train_v2":
            continue
        txts = [r["text"] for r in rows]
        F_sp = extract_surface_features(txts)
        probe_preds["probe_length"][sp_name] = clf_len.predict_proba(F_sp[:, :1])[:, 1].tolist()
        probe_preds["probe_surface_12feat"][sp_name] = clf_surf.predict_proba(F_sp)[:, 1].tolist()
        probe_preds["probe_tfidf_e0"][sp_name] = clf_tfidf_e0.predict_proba(txts)[:, 1].tolist()
        probe_preds["probe_tfidf_aug"][sp_name] = clf_tfidf_all.predict_proba(txts)[:, 1].tolist()

    for sp_chk in ["test_id", "test_confirm", "external_mixset"]:
        yv = [r["origin_label"] for r in datasets[sp_chk]]
        print(
            f"  [{sp_chk}] Surface12 AUROC={safe_auroc(yv, probe_preds['probe_surface_12feat'][sp_chk]):.3f} | "
            f"TFIDF-Aug AUROC={safe_auroc(yv, probe_preds['probe_tfidf_aug'][sp_chk]):.3f}"
        )
    return probe_preds


@torch.inference_mode()
def score_fd_bino_on_split(rows, obs_sha, perf_sha, obs_id="Qwen/Qwen2.5-3B", perf_id="Qwen/Qwen2.5-3B-Instruct", batch_size=8):
    tok = AutoTokenizer.from_pretrained(obs_id, revision=obs_sha)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    obs_model = AutoModelForCausalLM.from_pretrained(obs_id, revision=obs_sha, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    perf_model = AutoModelForCausalLM.from_pretrained(perf_id, revision=perf_sha, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    txts = [r["text"] for r in rows]
    fd_scores, bino_scores = [], []
    for st in range(0, len(txts), batch_size):
        batch_txt = txts[st:st + batch_size]
        enc = tok(batch_txt, return_tensors="pt", padding=True, truncation=True, max_length=384).to("cuda")
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        logits_obs = obs_model(input_ids=input_ids, attention_mask=attn).logits[:, :-1, :]
        logits_perf = perf_model(input_ids=input_ids, attention_mask=attn).logits[:, :-1, :]
        targets = input_ids[:, 1:]
        mask = attn[:, 1:].float()
        tok_cnt = mask.sum(dim=-1).clamp(min=1.0)

        B_cur, T_cur, _ = logits_obs.shape
        diff_sum = torch.zeros(B_cur, device="cuda", dtype=torch.float32)
        var_sum = torch.zeros(B_cur, device="cuda", dtype=torch.float32)
        nll_perf_sum = torch.zeros(B_cur, device="cuda", dtype=torch.float32)
        xce_sum = torch.zeros(B_cur, device="cuda", dtype=torch.float32)

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
        fd_scores.extend((diff_sum / torch.sqrt(var_sum.clamp(min=1e-8))).cpu().tolist())
        bino_scores.extend((-(nll_perf_sum / tok_cnt) / (xce_sum / tok_cnt).clamp(min=1e-8)).cpu().tolist())

    del obs_model, perf_model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return fd_scores, bino_scores


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
def score_radar_and_desklib_on_split(rows, radar_sha, desk_sha, batch_size=32):
    # RADAR
    r_id = "TrustSafeAI/RADAR-Vicuna-7B"
    tok_r = AutoTokenizer.from_pretrained(r_id, revision=radar_sha)
    mod_r = AutoModelForSequenceClassification.from_pretrained(r_id, revision=radar_sha, dtype=torch.bfloat16).to("cuda").eval()
    txts = [r["text"] for r in rows]
    radar_s = []
    for st in range(0, len(txts), batch_size):
        enc = tok_r(txts[st:st + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        p = F.softmax(mod_r(**enc).logits.float(), dim=-1)[:, 0]
        radar_s.extend(p.cpu().tolist())
    del mod_r, tok_r
    gc.collect()
    torch.cuda.empty_cache()

    # Desklib
    d_id = "desklib/ai-text-detector-v1.01"
    cfg = AutoConfig.from_pretrained(d_id, revision=desk_sha)
    tok_d = AutoTokenizer.from_pretrained(d_id, revision=desk_sha)
    net = DesklibDetector(cfg)
    ckpt = hf_hub_download(repo_id=d_id, filename="model.safetensors", revision=desk_sha)
    sd = load_file(ckpt)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert len(set(missing) - {"model.embeddings.position_ids"}) == 0 and len(unexpected) == 0
    net = net.to("cuda").eval()
    desk_s = []
    for st in range(0, len(txts), batch_size):
        enc = tok_d(txts[st:st + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        desk_s.extend(torch.sigmoid(net(enc["input_ids"], enc["attention_mask"])).cpu().tolist())
    del net, tok_d
    gc.collect()
    torch.cuda.empty_cache()
    return radar_s, desk_s


@torch.inference_mode()
def score_hc3_roberta_all(datasets, prov, batch_size=64):
    """
    Classic supervised literature anchor: Hello-SimpleAI/chatgpt-detector-roberta (Guo et al., HC3).
    Resolves exact commit SHA and uses documented id2label['1'] == 'ChatGPT'.
    """
    m_id = "Hello-SimpleAI/chatgpt-detector-roberta"
    api = HfApi()
    sha = api.model_info(m_id).sha
    prov["hf_model_commit_shas"][m_id] = sha
    tok = AutoTokenizer.from_pretrained(m_id, revision=sha)
    mod = AutoModelForSequenceClassification.from_pretrained(m_id, revision=sha, dtype=torch.bfloat16).to("cuda").eval()
    id2l = {int(k): str(v).lower() for k, v in getattr(mod.config, "id2label", {0: "human", 1: "chatgpt"}).items()}
    ai_col = 1 if ("chatgpt" in id2l.get(1, "") or "ai" in id2l.get(1, "") or "machine" in id2l.get(1, "")) else 0

    out = {}
    for sp_name, rows in datasets.items():
        if sp_name == "train_v2":
            continue
        txts = [r["text"] for r in rows]
        sc = []
        for st in range(0, len(txts), batch_size):
            enc = tok(txts[st:st + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            p = F.softmax(mod(**enc).logits.float(), dim=-1)[:, ai_col]
            sc.extend(p.cpu().tolist())
        out[sp_name] = sc
    del mod, tok
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    t0 = time.time()
    datasets = load_all_splits_v3()
    prov = json.load(open(os.path.join(V2_DIR, "provenance_lock.json")))
    m_shas = prov["hf_model_commit_shas"]

    old_p_path = os.path.join(V1_DIR, "preds_pretrained_and_probes.json")
    assert os.path.exists(old_p_path), f"Missing {old_p_path}! Upload results/preds_pretrained_and_probes.json first."
    merged = json.load(open(old_p_path))

    # 1. Re-fit shortcut probes on 800-parent bidirectional train_v2 and score all splits + test_confirm
    probes_v2 = run_probes_v2(datasets)
    merged.update(probes_v2)

    # 2. Score Fast-DetectGPT, Binoculars, RADAR, Desklib on pristine test_confirm (reusing verified v1 scores on fixed splits)
    conf_rows = datasets["test_confirm"]
    print("\n=== [2/4] Scoring Fast-DetectGPT & Binoculars on Pristine test_confirm (1,280 rows) ===")
    fd_c, bi_c = score_fd_bino_on_split(
        conf_rows,
        obs_sha=m_shas["Qwen/Qwen2.5-3B"],
        perf_sha=m_shas["Qwen/Qwen2.5-3B-Instruct"],
    )
    merged.setdefault("fast_detectgpt_qwen3b", {})["test_confirm"] = fd_c
    merged.setdefault("binoculars_qwen3b", {})["test_confirm"] = bi_c

    print("\n=== [3/4] Scoring RADAR-Vicuna-7B & Desklib-RAID-v1 on Pristine test_confirm (1,280 rows) ===")
    rad_c, dsk_c = score_radar_and_desklib_on_split(
        conf_rows,
        radar_sha=m_shas["TrustSafeAI/RADAR-Vicuna-7B"],
        desk_sha=m_shas["desklib/ai-text-detector-v1.01"],
    )
    merged.setdefault("radar_vicuna7b", {})["test_confirm"] = rad_c
    merged.setdefault("desklib_raid_v1", {})["test_confirm"] = dsk_c

    print("\n=== [4/4] Scoring HC3-RoBERTa Literature Anchor Across All Evaluation Splits ===")
    merged["hc3_roberta_detector"] = score_hc3_roberta_all(datasets, prov)

    y_conf = [r["origin_label"] for r in conf_rows]
    for det_k in ["fast_detectgpt_qwen3b", "binoculars_qwen3b", "radar_vicuna7b", "desklib_raid_v1", "hc3_roberta_detector"]:
        print(f"  [test_confirm] {det_k} AUROC={safe_auroc(y_conf, merged[det_k]['test_confirm']):.4f}")

    prov["stage3_pretrained_eval_seconds"] = round(time.time() - t0, 2)
    with open(os.path.join(V2_DIR, "provenance_lock.json"), "w") as f:
        json.dump(prov, f, indent=2)

    out_path = os.path.join(OUT_DIR, "preds_pretrained_and_probes_v3.json")
    with open(out_path, "w") as f:
        json.dump(merged, f)
    print(f"\n=== SAVED {out_path} ({prov['stage3_pretrained_eval_seconds']}s) ===")


if __name__ == "__main__":
    main()

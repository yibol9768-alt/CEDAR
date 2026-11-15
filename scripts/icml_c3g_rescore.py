#!/usr/bin/env python3
"""Confirm-3G (docs/CONFIRM3G_SPEC_2026-11-15.md), Colab: re-score saved final-run weights (LoRA + head, bf16) on
val_calib, Confirm-3 (re-scoring check) and Confirm-3G, then score the five pretrained detectors on Confirm-3G.
Inputs: /content/weights/f_*_weights.pt, /content/cedar_data*/..., /content/scripts/v3_eval_pretrained_on_confirm.py
Output: /content/p1_out/<P1_TAG>_seed<P1_SEED>.json, rewritten after every run (partial results survive)."""
import glob
import json
import os
import subprocess
import sys
import time

subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
try:
    import peft  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

OUT = "/content/p1_out"; os.makedirs(OUT, exist_ok=True)
FN = os.path.join(OUT, f"{os.environ.get('P1_TAG', 'c3g_rescore')}_seed{os.environ.get('P1_SEED', '0')}.json")
MAXLEN, LORA_R = 384, 16
BACKBONE = {"8b": "Qwen/Qwen3-8B-Base", "1p7b": "Qwen/Qwen3-1.7B-Base"}


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Det(nn.Module):  # identical to scripts/icml_final_train.py, without gradient checkpointing
    def __init__(self, mid):
        super().__init__()
        base = AutoModel.from_pretrained(mid, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        cfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.lm = get_peft_model(base, cfg)
        h = base.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(2 * h, 512), nn.GELU(), nn.Linear(512, 128))
        self.cls = nn.Linear(128, 1)

    def forward(self, ids, att):
        hs = self.lm(input_ids=ids, attention_mask=att).last_hidden_state.float()
        m = att.unsqueeze(-1).float()
        mean = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        last = hs[torch.arange(hs.size(0), device=hs.device), att.sum(1) - 1]
        z = F.normalize(self.proj(torch.cat([mean, last], -1)), dim=-1)
        return self.cls(z).squeeze(-1), z


@torch.no_grad()
def score(model, tok, rows, bs=32):
    model.eval(); out = []
    for i in range(0, len(rows), bs):
        enc = tok([r["text"] for r in rows[i:i + bs]], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model(enc["input_ids"], enc["attention_mask"])
        out += torch.sigmoid(lg.float()).cpu().tolist()
    return out


def main():
    t0 = time.time()
    val = json.load(open("/content/cedar_data/split_val.json"))
    cal = set(json.load(open("/content/cedar_data_v2/val_manifest.json"))["val_calib_parent_ids"])
    splits = {"val_calib": [r for r in val if r["parent_id"] in cal],
              "test_confirm3": json.load(open("/content/cedar_data_c3/split_test_confirm3.json")),
              "test_confirm3g": json.load(open("/content/cedar_data_c3g/split_test_confirm3g.json"))}
    res = json.load(open(FN)) if os.path.exists(FN) else {"row_ids": {sp: [r["id"] for r in rows] for sp, rows in splits.items()}, "runs": {}, "pretrained": {}}
    files = sorted(glob.glob("/content/weights/f_*_weights.pt"))
    for size in ("8b", "1p7b"):
        todo = [f for f in files if f"/f_{size}_" in f and os.path.basename(f).replace("_weights.pt", "") not in res["runs"]]
        if not todo:
            continue
        tok = AutoTokenizer.from_pretrained(BACKBONE[size]); tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = Det(BACKBONE[size]).to("cuda")
        trainable = {k for k, v in model.named_parameters() if v.requires_grad}
        for f in todo:
            run = os.path.basename(f).replace("_weights.pt", "")
            sd = torch.load(f, map_location="cuda")
            assert set(sd) == trainable, (run, len(set(sd) ^ trainable))
            with torch.no_grad():
                for k, p in model.named_parameters():
                    if k in sd:
                        p.copy_(sd[k].to(p.dtype))
            res["runs"][run] = {sp: score(model, tok, rows) for sp, rows in splits.items()}
            json.dump(res, open(FN + ".tmp", "w")); os.replace(FN + ".tmp", FN)
            say(f"RESCORED {run} ({time.time() - t0:.0f}s)")
        del model; torch.cuda.empty_cache()
    if not res["pretrained"]:
        sys.path.insert(0, "/content/scripts")
        import v3_eval_pretrained_on_confirm as v3p
        shas = json.load(open(os.path.join(v3p.V2_DIR, "provenance_lock.json")))["hf_model_commit_shas"]
        rows, sp = splits["test_confirm3g"], "test_confirm3g"
        pre = {}
        fd, bi = v3p.score_fd_bino_on_split(rows, obs_sha=shas["Qwen/Qwen2.5-3B"], perf_sha=shas["Qwen/Qwen2.5-3B-Instruct"])
        pre["fast_detectgpt_qwen3b"], pre["binoculars_qwen3b"] = {sp: fd}, {sp: bi}
        rd, dk = v3p.score_radar_and_desklib_on_split(rows, radar_sha=shas["TrustSafeAI/RADAR-Vicuna-7B"], desk_sha=shas["desklib/ai-text-detector-v1.01"])
        pre["radar_vicuna7b"], pre["desklib_raid_v1"] = {sp: rd}, {sp: dk}
        pre["hc3_roberta_detector"] = v3p.score_hc3_roberta_all({sp: rows}, json.load(open(os.path.join(v3p.V2_DIR, "provenance_lock.json"))))
        res["pretrained"] = pre
        json.dump(res, open(FN + ".tmp", "w")); os.replace(FN + ".tmp", FN)
    say(f"C3G_RESCORE_DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

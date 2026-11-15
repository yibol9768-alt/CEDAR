#!/usr/bin/env python3
"""ICML pilot P1 (docs/ICML_PLAN_2026-11-15.md): does an LLM backbone raise low-FPR detection under shift?
ERM + retention-weighted pair consistency (the recipe that replicated), on a decoder LLM with LoRA.
Env: P1_MODEL (HF id), P1_TAG, P1_SEED (42), P1_EPOCHS (2), P1_LORA_R (16), P1_LR (1e-4).
Runs on Colab; data under /content/cedar_data*, writes /content/p1_out/<tag>_seed<seed>.json
in the same format as the v3/E2 prediction files (row_ids, preds as probabilities, locked_thresholds)."""
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict

# Colab ships torchao 0.10, which current peft rejects at LoRA creation; we do not use torchao.
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
try:
    import peft  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

MODEL = os.environ.get("P1_MODEL", "Qwen/Qwen3-1.7B-Base")
TAG = os.environ.get("P1_TAG", "qwen3_1p7b_pair")
SEED = int(os.environ.get("P1_SEED", "42"))
EPOCHS = int(os.environ.get("P1_EPOCHS", "2"))
LORA_R = int(os.environ.get("P1_LORA_R", "16"))
LR = float(os.environ.get("P1_LR", "1e-4"))
PAIR_W = float(os.environ.get("P1_PAIR_W", "0.5"))  # 0 = augmented ERM on the same backbone
MAXLEN, PARENTS_PER_STEP, ACCUM = 384, 4, 2
LV = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4}
OUT = "/content/p1_out"
os.makedirs(OUT, exist_ok=True)


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(p):
    return json.load(open(p))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Det(nn.Module):
    def __init__(self):
        super().__init__()
        base = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
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
        last = hs[torch.arange(hs.size(0), device=hs.device), att.sum(1) - 1]  # right padding
        z = F.normalize(self.proj(torch.cat([mean, last], -1)), dim=-1)
        return self.cls(z).squeeze(-1), z


def pair_loss(z, pid, y, e, ld):
    d, w = [], []
    for p in pid.unique():
        for o in (0, 1):
            m = (pid == p) & (y == o)
            i0 = torch.nonzero(m & (e == 0)).flatten()
            if i0.numel() == 0:
                continue
            for i in torch.nonzero(m & (e > 0)).flatten():
                d.append(((z[i] - z[i0[0]]) ** 2).sum()); w.append((1 - ld[i]).clamp(0.25, 1.0))
    if not d:
        return z.sum() * 0
    w = torch.stack(w); return (w * torch.stack(d)).sum() / w.sum()


def conformal(s, pids, a=0.05):
    c = Counter(pids); w = np.array([1 / c[p] for p in pids])
    for t in np.unique(s):
        if (1 + w[s > t].sum()) / (len(c) + 1) <= a + 1e-12:
            return float(t)
    return float("inf")


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
    set_seed(SEED); t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train = [r for r in load("/content/cedar_data_v2/split_train_v2.json") if r["edit_level"] in ("E0", "E1", "E2", "E3")]
    val = load("/content/cedar_data/split_val.json")
    man = load("/content/cedar_data_v2/val_manifest.json")
    cal_p, sel_p = set(man["val_calib_parent_ids"]), set(man.get("val_select_parent_ids", []))
    splits = {"val_calib": [r for r in val if r["parent_id"] in cal_p],
              "val_select": [r for r in val if r["parent_id"] not in cal_p],
              "test_id": load("/content/cedar_data/split_test_id.json"),
              "test_confirm": load("/content/cedar_data_v2/split_test_confirm.json"),
              "val_shift": load("/content/cedar_data_e2/split_val_shift.json"),
              "test_confirm2": load("/content/cedar_data_e2/split_test_confirm2.json")}
    for ext in ["split_test_ood_ar", "split_test_ood_llada", "external_esl_full", "external_mixset", "external_beemo", "external_mage_wild"]:
        fp = f"/content/cedar_data/{ext}.json"
        if os.path.exists(fp):
            splits[ext.replace("split_", "")] = load(fp)
    model = Det().to("cuda")
    say(f"model {MODEL} loaded; trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")
    params = [{"params": [p for p in model.lm.parameters() if p.requires_grad], "lr": LR},
              {"params": list(model.proj.parameters()) + list(model.cls.parameters()), "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    by_p = defaultdict(list)
    for i, r in enumerate(train):
        by_p[r["parent_id"]].append(i)
    parents = sorted(by_p)
    steps = EPOCHS * math.ceil(len(parents) / PARENTS_PER_STEP)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    step = 0
    for ep in range(EPOCHS):
        model.train(); random.shuffle(parents)
        for b in range(0, len(parents), PARENTS_PER_STEP):
            idx = [i for p in parents[b:b + PARENTS_PER_STEP] for i in by_p[p]]
            rows = [train[i] for i in idx]
            enc = tok([r["text"] for r in rows], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
            y = torch.tensor([int(r["origin_label"]) for r in rows], device="cuda")
            e = torch.tensor([LV[r["edit_level"]] for r in rows], device="cuda")
            pid = torch.tensor([parents.index(r["parent_id"]) for r in rows], device="cuda")
            ld = torch.tensor([float(r.get("measured_lex_dist", 0.0)) for r in rows], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, z = model(enc["input_ids"], enc["attention_mask"])
            loss = F.binary_cross_entropy_with_logits(lg.float(), y.float()) + (PAIR_W * pair_loss(z, pid, y, e, ld) if PAIR_W > 0 else 0.0)
            (loss / ACCUM).backward()
            if (b // PARENTS_PER_STEP + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            sched.step(); step += 1
            if step % 50 == 0:
                say(f"ep{ep} step {step}/{steps} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")
    preds = {sp: score(model, tok, rows) for sp, rows in splits.items()}
    cal = splits["val_calib"]
    hum = [i for i, r in enumerate(cal) if int(r["origin_label"]) == 0 and r["edit_level"] in ("E0", "E1", "E2", "E3")]
    tau = conformal(np.array(preds["val_calib"])[hum], [cal[i]["parent_id"] for i in hum])
    rec = {"run_name": TAG, "seed": SEED, "backbone": MODEL, "epochs": EPOCHS, "lora_r": LORA_R, "lr": LR, "pair_w": PAIR_W,
           "row_ids": {sp: [r["id"] for r in rows] for sp, rows in splits.items()}, "preds": preds,
           "locked_thresholds": {"tau_cal_all5": tau}, "train_seconds": time.time() - t0}
    json.dump(rec, open(os.path.join(OUT, f"{TAG}_seed{SEED}.json"), "w"))
    say(f"P1_DONE {TAG} tau={tau:.4f} total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

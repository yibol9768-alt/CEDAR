#!/usr/bin/env python3
"""Confirm-3 (sealed): score the five pretrained detectors (same models/commits as v3/E2) on test_confirm3, and the
pre/post exposure diagnostic: Qwen3-8B-Base mean token NLL of every Confirm-3 text (spec: diagnostic, not a claim).
Thresholds are not set here; the final analysis locks them on val_calib from the v3 predictions, as in v3/E2.
Runs on Colab (bundle: /content/scripts, /content/cedar_data_v2/provenance_lock.json, /content/cedar_data_c3).
Output: /content/p1_out/<P1_TAG>_seed<P1_SEED>.json (run through scripts/icml_run_jobs.py with TRAINER set to this file)
  {detector: {"test_confirm3": [score per row]}, "row_ids": ...}"""
import json
import os
import sys
import time

sys.path.insert(0, "/content/scripts")
import torch  # noqa: E402
import v3_eval_pretrained_on_confirm as v3p  # noqa: E402

OUT = "/content/p1_out"
os.makedirs(OUT, exist_ok=True)


@torch.no_grad()
def mean_nll(rows, mid="Qwen/Qwen3-8B-Base", bs=16, maxlen=512):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
    out = []
    for i in range(0, len(rows), bs):
        enc = tok([r["text"] for r in rows[i:i + bs]], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to("cuda")
        lg = lm(**enc).logits[:, :-1].float()
        tgt = enc["input_ids"][:, 1:]; m = enc["attention_mask"][:, 1:].float()
        nll = torch.nn.functional.cross_entropy(lg.transpose(1, 2), tgt, reduction="none")
        out += ((nll * m).sum(1) / m.sum(1)).cpu().tolist()
    del lm; torch.cuda.empty_cache()
    return out


def main():
    t0 = time.time()
    shas = json.load(open(os.path.join(v3p.V2_DIR, "provenance_lock.json")))["hf_model_commit_shas"]
    rows = json.load(open("/content/cedar_data_c3/split_test_confirm3.json"))
    sp = "test_confirm3"
    out = {k: {} for k in ["fast_detectgpt_qwen3b", "binoculars_qwen3b", "radar_vicuna7b", "desklib_raid_v1"]}
    fd, bi = v3p.score_fd_bino_on_split(rows, obs_sha=shas["Qwen/Qwen2.5-3B"], perf_sha=shas["Qwen/Qwen2.5-3B-Instruct"])
    out["fast_detectgpt_qwen3b"][sp], out["binoculars_qwen3b"][sp] = fd, bi
    rd, dk = v3p.score_radar_and_desklib_on_split(rows, radar_sha=shas["TrustSafeAI/RADAR-Vicuna-7B"],
                                                   desk_sha=shas["desklib/ai-text-detector-v1.01"])
    out["radar_vicuna7b"][sp], out["desklib_raid_v1"][sp] = rd, dk
    out["hc3_roberta_detector"] = v3p.score_hc3_roberta_all({sp: rows}, json.load(open(os.path.join(v3p.V2_DIR, "provenance_lock.json"))))
    out["qwen3_8b_base_mean_nll"] = {sp: mean_nll(rows)}
    out["row_ids"] = {sp: [r["id"] for r in rows]}
    fn = f"{os.environ.get('P1_TAG', 'preds_pretrained_c3')}_seed{os.environ.get('P1_SEED', '0')}.json"
    json.dump(out, open(os.path.join(OUT, fn), "w"))
    print(f"C3_PRETRAINED_DONE ({time.time() - t0:.0f}s)", flush=True)  # no metrics printed: the split is sealed


if __name__ == "__main__":
    main()

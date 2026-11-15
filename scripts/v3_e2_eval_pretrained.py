#!/usr/bin/env python3
"""E2: score the five pretrained detectors on val_shift and test_confirm2 (same models/commits as v3).

Thresholds are NOT set here; the analysis locks them on val_calib from the v3 predictions, as in v3.
Output: /content/cedar_results_e2/preds_pretrained_e2.json  {detector: {split: [score per row]}}
"""
import json
import os
import sys
import time

sys.path.insert(0, "/content/scripts")
import v3_eval_pretrained_on_confirm as v3p  # noqa: E402

E2_DIR = "/content/cedar_data_e2"
OUT = "/content/cedar_results_e2"
os.makedirs(OUT, exist_ok=True)


def main():
    t0 = time.time()
    prov = json.load(open(os.path.join(v3p.V2_DIR, "provenance_lock.json")))
    shas = prov["hf_model_commit_shas"]
    splits = {sp: json.load(open(os.path.join(E2_DIR, f"split_{sp}.json"))) for sp in ("val_shift", "test_confirm2")}
    out = {k: {} for k in ["fast_detectgpt_qwen3b", "binoculars_qwen3b", "radar_vicuna7b", "desklib_raid_v1"]}
    for sp, rows in splits.items():
        fd, bi = v3p.score_fd_bino_on_split(rows, obs_sha=shas["Qwen/Qwen2.5-3B"], perf_sha=shas["Qwen/Qwen2.5-3B-Instruct"])
        out["fast_detectgpt_qwen3b"][sp], out["binoculars_qwen3b"][sp] = fd, bi
        rd, dk = v3p.score_radar_and_desklib_on_split(rows, radar_sha=shas["TrustSafeAI/RADAR-Vicuna-7B"],
                                                       desk_sha=shas["desklib/ai-text-detector-v1.01"])
        out["radar_vicuna7b"][sp], out["desklib_raid_v1"][sp] = rd, dk
    out["hc3_roberta_detector"] = v3p.score_hc3_roberta_all(dict(splits), prov)
    out["row_ids"] = {sp: [r["id"] for r in rows] for sp, rows in splits.items()}
    for k in ["fast_detectgpt_qwen3b", "binoculars_qwen3b", "radar_vicuna7b", "desklib_raid_v1", "hc3_roberta_detector"]:
        for sp, rows in splits.items():
            print(f"[{sp}] {k} AUROC={v3p.safe_auroc([r['origin_label'] for r in rows], out[k][sp]):.4f}", flush=True)
    json.dump(out, open(os.path.join(OUT, "preds_pretrained_e2.json"), "w"))
    print(f"saved ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

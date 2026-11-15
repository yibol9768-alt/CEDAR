import os
import gc
import json
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
sys.path.insert(0, "/content")
import src.cedar_v2_model as cedar_v2_mod
from scripts.v3_train_cedar_v2_suite import (
    V2_DIR,
    load_and_verify_all_splits,
    compute_full_split_fingerprint,
    build_job_queues,
    pretokenize_split,
    train_and_eval_job,
)

# On 40GB GPUs (e.g. A100-40GB), DeBERTa-v3-large batch=64 L=384 uses ~39.8GB uncheckpointed;
# enable activation checkpointing on DeBERTa without modifying src/cedar_v2_model.py (preserving config_hash & exact math).
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 55 * (1024 ** 3):
    _orig_init = cedar_v2_mod.CEDARv2Detector.__init__

    def _lowmem_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        bname = kwargs.get("backbone_name", args[0] if args else "")
        if "deberta" in str(bname).lower() and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    cedar_v2_mod.CEDARv2Detector.__init__ = _lowmem_init


def main():
    worker_id = int(os.environ["CEDAR_WORKER_ID"])
    num_workers = int(os.environ.get("CEDAR_NUM_WORKERS", "2"))
    mb_tail_count = int(os.environ.get("CEDAR_MB_TAIL_COUNT", "23"))
    print(
        f"=== v3_bottom_up_tail_worker | mirroring tail of worker={worker_id}/{num_workers} "
        f"(all 15 DB + last {mb_tail_count} MB | vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB) ===",
        flush=True,
    )

    prov = json.load(open(os.path.join(V2_DIR, "provenance_lock.json")))
    m_shas = prov["hf_model_commit_shas"]
    raw_splits, val_manifest = load_and_verify_all_splits()
    data_fp = compute_full_split_fingerprint(raw_splits, val_manifest)
    mb_jobs_all, db_jobs_all = build_job_queues()

    my_mb_jobs = [jb for i, jb in enumerate(mb_jobs_all) if (i % num_workers) == worker_id]
    my_db_jobs = [jb for i, jb in enumerate(db_jobs_all) if (i % num_workers) == worker_id]

    mb_id = "answerdotai/ModernBERT-large"
    mb_sha = m_shas[mb_id]
    db_id = "microsoft/deberta-v3-large"
    db_sha = m_shas[db_id]

    snapshot_download(repo_id=mb_id)
    snapshot_download(repo_id=mb_id, revision=mb_sha)
    snapshot_download(repo_id=db_id)
    snapshot_download(repo_id=db_id, revision=db_sha)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 1. Run DeBERTa-v3-large jobs FIRST (bottom-up: indices 14 -> 0) since primary workers run DB LAST!
    print(f"\n>>> [BOTTOM-UP PHASE 1] Pre-tokenizing splits for {db_id} @ {db_sha[:10]} (15 runs) ...", flush=True)
    db_tok = AutoTokenizer.from_pretrained(db_id, revision=db_sha, local_files_only=True)
    db_tok_splits = {k: pretokenize_split(v, db_tok, max_len=384) for k, v in raw_splits.items()}

    for jb in reversed(my_db_jobs):
        train_and_eval_job(jb, db_id, db_sha, db_tok_splits, data_fp, lr_backbone=1.5e-5, lr_head=2.0e-4, epochs=4)

    del db_tok_splits, db_tok
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Run ModernBERT-large tail jobs (bottom-up: indices 42 -> 43 - mb_tail_count)
    tail_mb = list(reversed(my_mb_jobs))[:mb_tail_count]
    print(f"\n>>> [BOTTOM-UP PHASE 2] Pre-tokenizing splits for {mb_id} @ {mb_sha[:10]} ({len(tail_mb)} tail runs) ...", flush=True)
    mb_tok = AutoTokenizer.from_pretrained(mb_id, revision=mb_sha, local_files_only=True)
    mb_tok_splits = {k: pretokenize_split(v, mb_tok, max_len=384) for k, v in raw_splits.items()}

    for jb in tail_mb:
        train_and_eval_job(jb, mb_id, mb_sha, mb_tok_splits, data_fp, lr_backbone=2.5e-5, lr_head=2.5e-4, epochs=4)

    print(f"\n=== BOTTOM-UP TAIL WORKER FOR SHARD {worker_id}/{num_workers} FINISHED! ===", flush=True)


if __name__ == "__main__":
    main()

import os
import gc
import sys
import json
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "/content/scripts/v3_4way_worker.py"))))
sys.path.insert(0, "/content" if os.path.exists("/content/src") else _REPO_ROOT)
import src.cedar_v2_model as cedar_v2_mod
from scripts.v3_train_cedar_v2_suite import (
    V2_DIR,
    PRED_DIR,
    CKPT_DIR,
    load_and_verify_all_splits,
    compute_full_split_fingerprint,
    build_job_queues,
    pretokenize_split,
    train_and_eval_job,
)

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 55 * (1024 ** 3):
    _orig_init = cedar_v2_mod.CEDARv2Detector.__init__

    def _lowmem_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        bname = kwargs.get("backbone_name", args[0] if args else "")
        if "deberta" in str(bname).lower() and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    cedar_v2_mod.CEDARv2Detector.__init__ = _lowmem_init


def job_is_done(jb: dict, data_fp: str) -> bool:
    stem = f"{jb['run_name']}_seed{jb['seed']}"
    jf = os.path.join(PRED_DIR, f"{stem}.json")
    cf = os.path.join(CKPT_DIR, f"{stem}.pt")
    if not os.path.exists(jf):
        return False
    if jb.get("save_ckpt", False) and not os.path.exists(cf):
        return False
    try:
        old = json.load(open(jf))
        return old.get("data_fingerprint") == data_fp
    except Exception:
        return False


def main():
    worker_id = int(os.environ["CEDAR_WORKER_ID"])
    num_workers = int(os.environ.get("CEDAR_NUM_WORKERS", "4"))

    prov = json.load(open(os.path.join(V2_DIR, "provenance_lock.json")))
    m_shas = prov["hf_model_commit_shas"]
    raw_splits, val_manifest = load_and_verify_all_splits()
    data_fp = compute_full_split_fingerprint(raw_splits, val_manifest)
    mb_jobs_all, db_jobs_all = build_job_queues()

    rem_mb = [jb for jb in mb_jobs_all if not job_is_done(jb, data_fp)]
    rem_db = [jb for jb in db_jobs_all if not job_is_done(jb, data_fp)]

    my_mb = [jb for i, jb in enumerate(rem_mb) if (i % num_workers) == worker_id]
    my_db = [jb for i, jb in enumerate(rem_db) if (i % num_workers) == worker_id]
    print(
        f"=== v3_4way_worker {worker_id}/{num_workers} | vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB "
        f"| assigned {len(my_mb)}/{len(rem_mb)} MB + {len(my_db)}/{len(rem_db)} DB ===",
        flush=True,
    )

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

    if my_mb:
        mb_tok = AutoTokenizer.from_pretrained(mb_id, revision=mb_sha, local_files_only=True)
        mb_tok_splits = {k: pretokenize_split(v, mb_tok, max_len=384) for k, v in raw_splits.items()}
        for jb in my_mb:
            train_and_eval_job(jb, mb_id, mb_sha, mb_tok_splits, data_fp, lr_backbone=2.5e-5, lr_head=2.5e-4, epochs=4)
        del mb_tok_splits, mb_tok
        gc.collect()
        torch.cuda.empty_cache()

    if my_db:
        db_tok = AutoTokenizer.from_pretrained(db_id, revision=db_sha, local_files_only=True)
        db_tok_splits = {k: pretokenize_split(v, db_tok, max_len=384) for k, v in raw_splits.items()}
        for jb in my_db:
            train_and_eval_job(jb, db_id, db_sha, db_tok_splits, data_fp, lr_backbone=1.5e-5, lr_head=2.0e-4, epochs=4)
        del db_tok_splits, db_tok
        gc.collect()
        torch.cuda.empty_cache()

    print(f"=== v3_4way_worker {worker_id}/{num_workers} COMPLETED ALL ASSIGNED RUNS! ===", flush=True)


if __name__ == "__main__":
    main()

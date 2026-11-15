"""Run an explicit list of suite jobs (CEDAR_JOBS="run_name_seedN,...") on one GPU.
CEDAR_FORCE_GC=1 enables DeBERTa gradient checkpointing regardless of VRAM (needed on 80GB A100)."""
import os
import gc
import sys
import json
import torch
from transformers import AutoTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "/content/scripts/v3_run_named_jobs.py"))))
sys.path.insert(0, "/content" if os.path.exists("/content/src") else _REPO_ROOT)
import src.cedar_v2_model as cedar_v2_mod
from scripts.v3_train_cedar_v2_suite import (
    V2_DIR, load_and_verify_all_splits, compute_full_split_fingerprint,
    build_job_queues, build_extension_jobs_e1, build_extension_jobs_e2, pretokenize_split, train_and_eval_job,
)

if os.environ.get("CEDAR_FORCE_GC") == "1" or torch.cuda.get_device_properties(0).total_memory < 55 * (1024 ** 3):
    _orig_init = cedar_v2_mod.CEDARv2Detector.__init__

    def _gc_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        bname = kwargs.get("backbone_name", args[0] if args else "")
        if "deberta" in str(bname).lower() and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    cedar_v2_mod.CEDARv2Detector.__init__ = _gc_init


def is_db(name):
    return name.startswith("db_") or name.startswith("e2_db_")


def main():
    wanted = [s.strip() for s in os.environ["CEDAR_JOBS"].split(",") if s.strip()]
    prov = json.load(open(os.path.join(V2_DIR, "provenance_lock.json")))
    m_shas = prov["hf_model_commit_shas"]
    raw_splits, val_manifest = load_and_verify_all_splits()
    data_fp = compute_full_split_fingerprint(raw_splits, val_manifest)
    mb_all, db_all = build_job_queues()
    e_mb, e_db = build_extension_jobs_e1()
    f_mb, f_db = build_extension_jobs_e2()
    mb_all, db_all = mb_all + e_mb + f_mb, db_all + e_db + f_db
    idx = {f"{j['run_name']}_seed{j['seed']}": j for j in mb_all + db_all}
    missing = [w for w in wanted if w not in idx]
    assert not missing, f"unknown jobs: {missing}"
    from huggingface_hub import snapshot_download
    for _mid in {("microsoft/deberta-v3-large" if is_db(w) else "answerdotai/ModernBERT-large") for w in wanted}:
        snapshot_download(repo_id=_mid)
        snapshot_download(repo_id=_mid, revision=m_shas[_mid])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    groups = [
        ("answerdotai/ModernBERT-large", dict(lr_backbone=2.5e-5, lr_head=2.5e-4, epochs=4), [w for w in wanted if not is_db(w)]),
        ("microsoft/deberta-v3-large", dict(lr_backbone=1.5e-5, lr_head=2.0e-4, epochs=4), [w for w in wanted if is_db(w)]),
    ]
    print(f"=== named-jobs worker | vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB | jobs={wanted} ===", flush=True)
    for mid, hp, names in groups:
        if not names:
            continue
        sha = m_shas[mid]
        tok = AutoTokenizer.from_pretrained(mid, revision=sha, local_files_only=True)
        toks = {k: pretokenize_split(v, tok, max_len=384) for k, v in raw_splits.items()}
        for n in names:
            train_and_eval_job(idx[n], mid, sha, toks, data_fp, **hp)
        del toks, tok
        gc.collect()
        torch.cuda.empty_cache()
    print("=== named-jobs worker COMPLETED ===", flush=True)


if __name__ == "__main__":
    main()

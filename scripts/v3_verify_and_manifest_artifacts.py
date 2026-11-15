import os
import re
import glob
import json
import hashlib
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_V1_DIR = os.path.join(ROOT, "data")
DATA_V2_DIR = os.path.join(ROOT, "data_v2")
RES_V3_DIR = os.path.join(ROOT, "results_v3")
SCRIPT_DIR = os.path.join(ROOT, "scripts")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(t: str) -> str:
    norm = re.sub(r"\W+", "", t.lower())[:300]
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def verify_mixset_clustering():
    ms_path = os.path.join(DATA_V1_DIR, "external_mixset.json")
    with open(ms_path) as f:
        ms = json.load(f)
    assert len(ms) == 720, f"Expected 720 MixSet records, got {len(ms)}"
    fams_all = {r.get("parent_id") for r in ms}
    fams_pure = {r.get("parent_id") for r in ms if not r.get("is_completion", False)}
    assert len(fams_all) == 120 and len(fams_pure) == 120

    orig_records = [r for r in ms if r["id"].startswith("mixset_orig_")]
    assert len(orig_records) == 120
    for r in orig_records:
        expected_pid = f"ms_orig_{text_hash(r['text'])}"
        assert r["parent_id"] == expected_pid

    fam_sizes = Counter(Counter(r["parent_id"] for r in ms).values())
    pure_rows = [r for r in ms if not r.get("is_completion", False)]
    return {
        "n_all": len(ms),
        "n_all_human": sum(1 for r in ms if r["origin_label"] == 0),
        "n_all_machine": sum(1 for r in ms if r["origin_label"] == 1),
        "n_pure": len(pure_rows),
        "n_pure_human": sum(1 for r in pure_rows if r["origin_label"] == 0),
        "n_pure_machine": sum(1 for r in pure_rows if r["origin_label"] == 1),
        "n_hwt_llm_polished": sum(1 for r in ms if ("polish" in r["subset"]) and (r["origin_label"] == 0)),
        "n_mgt_humanized": sum(1 for r in ms if ("humanize" in r["subset"]) and (r["origin_label"] == 1)),
        "n_families_all": len(fams_all),
        "n_families_pure": len(fams_pure),
        "family_size_histogram": {str(k): int(v) for k, v in sorted(fam_sizes.items())},
    }


def fmt_pct(x: float, digits: int = 1) -> str:
    q = Decimal("10") ** (-digits)
    return str((Decimal(str(x)) * Decimal("100")).quantize(q, rounding=ROUND_HALF_UP))


def main():
    ms_info = verify_mixset_clustering()

    scripts_manifest = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "src", "*.py")) + glob.glob(os.path.join(SCRIPT_DIR, "*.py"))):
        scripts_manifest[os.path.relpath(p, ROOT)] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}

    data_manifest = {}
    for d_dir in [DATA_V1_DIR, DATA_V2_DIR]:
        for p in sorted(glob.glob(os.path.join(d_dir, "*.json")) + glob.glob(os.path.join(d_dir, "*.txt"))):
            entry = {"sha256": sha256_file(p), "sha256_prefix12": sha256_file(p)[:12], "bytes": os.path.getsize(p)}
            if p.endswith(".json"):
                with open(p) as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    entry["n_records"] = len(obj)
                    def _cid(r):
                        if r.get("parent_id"):
                            return str(r["parent_id"])
                        if r.get("prompt_id"):
                            return str(r["prompt_id"])
                        rid = str(r.get("id", ""))
                        return (rid[:-3] if rid.endswith("_E1") else rid) if rid.startswith("esl_") else rid
                    entry["n_clusters"] = len({_cid(r) for r in obj})
            data_manifest[os.path.relpath(p, ROOT)] = entry

    pred_files = sorted(glob.glob(os.path.join(RES_V3_DIR, "model_preds_v3", "*.json")))
    assert len(pred_files) == 116, f"Expected 116 prediction JSONs, found {len(pred_files)}"

    mb_secs, db_secs = [], []
    preds_manifest = {}
    for p in pred_files:
        with open(p) as f:
            pl = json.load(f)
        sec = float(pl["stage_timings"]["total_seconds"])
        if pl["backbone"] == "answerdotai/ModernBERT-large":
            mb_secs.append(sec)
        elif pl["backbone"] == "microsoft/deberta-v3-large":
            db_secs.append(sec)
        preds_manifest[os.path.relpath(p, ROOT)] = {
            "sha256": sha256_file(p),
            "run_name": pl["run_name"],
            "seed": int(pl["seed"]),
            "backbone": pl["backbone"],
            "backbone_commit_sha": pl.get("backbone_commit_sha"),
            "total_seconds": sec,
            "trainable_params": pl.get("trainable_params"),
            "data_fingerprint": pl.get("data_fingerprint"),
            "config_hash": pl.get("config_hash"),
        }

    ckpt_manifest = {}
    for p in sorted(glob.glob(os.path.join(RES_V3_DIR, "checkpoints", "*.pt"))):
        ckpt_manifest[os.path.relpath(p, ROOT)] = {
            "sha256": sha256_file(p),
            "sha256_prefix12": sha256_file(p)[:12],
            "bytes": os.path.getsize(p),
        }

    prov = json.load(open(os.path.join(DATA_V2_DIR, "provenance_lock.json")))
    total_secs = sum(mb_secs) + sum(db_secs)
    compute_accounting = {
        "n_saved_neural_runs": len(pred_files),
        "n_modernbert_runs": len(mb_secs),
        "n_deberta_runs": len(db_secs),
        "total_logged_train_eval_seconds": round(total_secs, 2),
        "total_logged_train_eval_gpu_hours": round(total_secs / 3600.0, 4),
        "modernbert_seconds_mean": round(sum(mb_secs) / len(mb_secs), 2),
        "modernbert_seconds_min_max": [round(min(mb_secs), 2), round(max(mb_secs), 2)],
        "deberta_seconds_mean": round(sum(db_secs) / len(db_secs), 2),
        "deberta_seconds_min_max": [round(min(db_secs), 2), round(max(db_secs), 2)],
        "stage2_bidirectional_generation_timings_seconds": prov.get("generation_stage_timings_seconds", {}),
        "archived_canonical_checkpoints": list(ckpt_manifest.keys()),
    }

    result_files = {}
    for fn in [
        "preds_pretrained_and_probes_v3.json",
        "nsd_v3_theorem2_and_geometry.json",
        "master_evaluation_summary_v3.json",
    ]:
        p = os.path.join(RES_V3_DIR, fn)
        assert os.path.exists(p), f"Missing {p}"
        result_files[os.path.relpath(p, ROOT)] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}

    manifest = {
        "schema_version": "3.0",
        "code_version": "cedar_v3_spotlight",
        "provenance_lock_summary": {
            "n_pinned_hf_models": len(prov.get("hf_model_commit_shas", {})),
            "n_pinned_hf_datasets": len(prov.get("hf_dataset_commit_shas", {})),
            "hf_model_commit_shas": prov.get("hf_model_commit_shas", {}),
            "hf_dataset_commit_shas": prov.get("hf_dataset_commit_shas", {}),
            "requirements_lock_packages": len(prov.get("pip_freeze_packages", [])),
        },
        "mixset_clustering_verification": ms_info,
        "compute_accounting": compute_accounting,
        "scripts": scripts_manifest,
        "datasets_and_locks": data_manifest,
        "archived_checkpoints": ckpt_manifest,
        "result_artifacts": result_files,
        "neural_prediction_runs": preds_manifest,
    }
    out_path = os.path.join(RES_V3_DIR, "artifact_manifest_v3.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote", out_path)
    print("Compute accounting:", json.dumps(compute_accounting, indent=2))


if __name__ == "__main__":
    main()

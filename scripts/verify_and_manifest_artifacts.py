import os
import re
import glob
import json
import hashlib
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
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
    ms_path = os.path.join(DATA_DIR, "external_mixset.json")
    with open(ms_path) as f:
        ms = json.load(f)
    assert len(ms) == 720, f"Expected 720 MixSet records, got {len(ms)}"
    fams_all = {r.get("parent_id") for r in ms}
    fams_pure = {r.get("parent_id") for r in ms if not r.get("is_completion", False)}
    assert len(fams_all) == 120, f"Expected 120 families in All 720, got {len(fams_all)}"
    assert len(fams_pure) == 120, f"Expected 120 families in Pure 620, got {len(fams_pure)}"

    orig_records = [r for r in ms if r["id"].startswith("mixset_orig_")]
    assert len(orig_records) == 120, f"Expected 120 deduplicated originals, got {len(orig_records)}"
    for r in orig_records:
        expected_pid = f"ms_orig_{text_hash(r['text'])}"
        assert r["parent_id"] == expected_pid, f"Mismatch on {r['id']}: {r['parent_id']} != {expected_pid}"

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
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
        with open(p) as f:
            obj = json.load(f)
        n_rows = len(obj) if isinstance(obj, list) else None
        n_clusters = (
            len({r.get("parent_id") or r.get("prompt_id") or r.get("id") for r in obj})
            if isinstance(obj, list)
            else None
        )
        data_manifest[os.path.relpath(p, ROOT)] = {
            "sha256": sha256_file(p),
            "sha256_prefix12": sha256_file(p)[:12],
            "n_records": n_rows,
            "n_clusters": n_clusters,
        }

    pred_files = sorted(glob.glob(os.path.join(RES_DIR, "model_preds_v2", "*.json")))
    mb_secs, db_secs = [], []
    preds_manifest = {}
    for p in pred_files:
        with open(p) as f:
            pl = json.load(f)
        sec = float(pl["train_seconds"])
        if pl["run_name"].startswith("mb_"):
            mb_secs.append(sec)
        elif pl["run_name"].startswith("db_"):
            db_secs.append(sec)
        preds_manifest[os.path.relpath(p, ROOT)] = {
            "sha256": sha256_file(p),
            "run_name": pl["run_name"],
            "seed": int(pl["seed"]),
            "train_seconds": sec,
            "trainable_params": pl.get("trainable_params"),
            "data_fingerprint": pl.get("data_fingerprint"),
        }

    total_secs = sum(mb_secs) + sum(db_secs)
    compute_accounting = {
        "n_saved_neural_runs": len(pred_files),
        "n_modernbert_runs": len(mb_secs),
        "n_deberta_runs": len(db_secs),
        "total_logged_train_eval_seconds": round(total_secs, 2),
        "total_logged_train_eval_hours": round(total_secs / 3600.0, 4),
        "modernbert_seconds_mean": round(sum(mb_secs) / len(mb_secs), 2),
        "modernbert_seconds_min_max": [round(min(mb_secs), 2), round(max(mb_secs), 2)],
        "deberta_seconds_mean": round(sum(db_secs) / len(db_secs), 2),
        "deberta_seconds_min_max": [round(min(db_secs), 2), round(max(db_secs), 2)],
        "checkpoint_persistence_note": "In-memory best val_select model states were evaluated immediately to produce persisted per-record prediction JSONs (results/model_preds_v2/*.json); raw model weight checkpoints were not archived.",
    }

    result_files = {}
    for fn in ["preds_pretrained_and_probes.json", "nsd_spectrum_and_geometry.json", "master_evaluation_summary_v2.json"]:
        p = os.path.join(RES_DIR, fn)
        if os.path.exists(p):
            result_files[os.path.relpath(p, ROOT)] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}

    pipeline_stages = [
        {
            "stage": "step1_prepare_sources_and_external",
            "script": "scripts/step1_prepare_sources_and_external.py",
            "upstream_datasets": [
                "HuggingFaceH4/no_robots",
                "yaful/MAGE",
                "liamdugan/raid",
                "chillies/IELTS-writing-task-2-evaluation",
                "ONE-Lab/MixSet",
                "toloka/beemo",
            ],
            "outputs": [
                "data/raw_parents.json",
                "data/external_esl_raw.json",
                "data/external_mixset.json",
                "data/external_beemo.json",
                "data/external_mage_wild.json",
            ],
        },
        {
            "stage": "step2_generate_symmetric_matrix",
            "script": "scripts/step2_generate_symmetric_matrix.py",
            "upstream_models": [
                "Qwen/Qwen2.5-7B-Instruct",
                "HuggingFaceH4/zephyr-7b-beta",
                "tiiuae/Falcon3-7B-Instruct",
                "allenai/OLMo-2-1124-7B-Instruct",
                "GSAI-ML/LLaDA-8B-Instruct",
                "microsoft/Phi-4-mini-instruct",
            ],
            "outputs": [
                "data/split_train.json",
                "data/split_val.json",
                "data/split_test_id.json",
                "data/split_test_ood_ar.json",
                "data/split_test_ood_llada.json",
                "data/external_esl_full.json",
            ],
        },
        {
            "stage": "step3_eval_pretrained_and_probes",
            "script": "scripts/step3_eval_pretrained_and_probes.py",
            "upstream_models": [
                "Qwen/Qwen2.5-3B",
                "Qwen/Qwen2.5-3B-Instruct",
                "TrustSafeAI/RADAR-Vicuna-7B",
                "desklib/ai-text-detector-academic-v1.01",
            ],
            "outputs": ["results/preds_pretrained_and_probes.json"],
        },
        {
            "stage": "step4_train_cedar_and_baselines",
            "scripts": [
                "src/cedar_model.py",
                "scripts/step4_train_cedar_and_baselines.py",
                "scripts/step4_worker2_bottom_up.py",
            ],
            "upstream_models": ["answerdotai/ModernBERT-large", "microsoft/deberta-v3-large"],
            "outputs": ["results/model_preds_v2/*.json (51 ModernBERT + 12 DeBERTa runs)"],
        },
        {
            "stage": "step4b_nsd_rank_and_spectrum",
            "scripts": ["src/cedar_model.py", "scripts/step4b_nsd_rank_and_spectrum.py"],
            "outputs": [
                "results/nsd_spectrum_and_geometry.json",
                "results/model_preds_v2/mb_cedar_r{8,16,64}_seed{42,1042,2042}.json",
            ],
        },
        {
            "stage": "step5_to_step7_local_analysis_figures_tables",
            "scripts": [
                "scripts/step5_statistical_analysis_and_figures.py",
                "scripts/step6_generate_iclr_figures.py",
                "scripts/step7_generate_latex_tables.py",
            ],
            "outputs": [
                "results/master_evaluation_summary_v2.json",
                "paper/figures/fig1..fig4.pdf",
                "paper/tables/tab1..tab5.tex",
            ],
        },
    ]

    environment_notes = {
        "gpu_hardware": "NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, Google Colab Linux x86_64)",
        "cloud_software_stack_unpinned": ["torch (CUDA bfloat16/float32)", "transformers", "datasets", "accelerate", "scikit-learn", "scipy", "numpy"],
        "local_replay_stack": {
            "os": "macOS",
            "python_stdlib_only_manifest_verifier": True,
            "analysis_figure_packages": ["numpy", "scipy", "scikit-learn", "matplotlib"],
        },
        "reproducibility_scope_and_limitations": "Exact deterministic replay is guaranteed from persisted split JSONs (data/*.json) and per-record model prediction JSONs (results/model_preds_v2/*.json, results/preds_pretrained_and_probes.json) through statistical summaries, figures, and LaTeX tables. Resolved cloud package lockfiles, upstream Hugging Face commit SHAs (pulled from default main revisions on 2026-11-14), per-stage generation logs, and trained model weight checkpoints were not archived, so fresh upstream generation or retraining is not bitwise-guaranteed.",
        "path_note": "Cloud GPU scripts (step1-step4b) default to /content/cedar_data; local analysis/figure/table scripts (step5-step7) resolve relative repository paths data/, results/, paper/.",
    }

    manifest = {
        "schema_version": "1.1",
        "environment_and_provenance": environment_notes,
        "pipeline_stages": pipeline_stages,
        "mixset_clustering_verification": ms_info,
        "compute_accounting": compute_accounting,
        "scripts": scripts_manifest,
        "datasets": data_manifest,
        "result_artifacts": result_files,
        "neural_prediction_runs": preds_manifest,
    }
    out_path = os.path.join(RES_DIR, "artifact_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote", out_path)
    print("MixSet verification:", json.dumps(ms_info))
    print("Compute accounting:", json.dumps(compute_accounting))

    with open(os.path.join(RES_DIR, "master_evaluation_summary_v2.json")) as f:
        S = json.load(f)
    agg = S["trained_multi_seed_agg"]
    pre = S["pretrained_and_probes"]
    sig = S["paired_significance_tests"]
    print("=== ONE-STEP ROUND_HALF_UP CHECK ===")
    c = agg["mb_cedar_full"]
    a = agg["mb_augmented_erm"]
    u = agg["mb_unedited_erm"]
    dc = agg["db_cedar_full"]
    print("CEDAR ID WG:", fmt_pct(c["id_wg_E0_E3"]["mean"]), "±", fmt_pct(c["id_wg_E0_E3"]["std"]))
    print("CEDAR E2 E0-FPR:", fmt_pct(c["id_recoverability"]["E2"]["fpr_e0cal"]["mean"]), "±", fmt_pct(c["id_recoverability"]["E2"]["fpr_e0cal"]["std"]))
    print("Unedited ERM E2 E0-FPR:", fmt_pct(u["id_recoverability"]["E2"]["fpr_e0cal"]["mean"]), "±", fmt_pct(u["id_recoverability"]["E2"]["fpr_e0cal"]["std"]))
    print("CEDAR MixSet Pure:", fmt_pct(c["mixset_pure_auroc"]["mean"]), "±", fmt_pct(c["mixset_pure_auroc"]["std"]), "| 2dec:", fmt_pct(c["mixset_pure_auroc"]["mean"], 2))
    print("CEDAR MixSet All:", fmt_pct(c["mixset_all_auroc"]["mean"]), "±", fmt_pct(c["mixset_all_auroc"]["std"]))
    print("Desklib MixSet Pure/All:", fmt_pct(pre["desklib_raid_v1"]["external_mixset"]["pure_origin_auroc"]), "/", fmt_pct(pre["desklib_raid_v1"]["external_mixset"]["all_720_auroc"]))
    print("CEDAR vs AugERM Beemo:", fmt_pct(c["beemo_auroc"]["mean"]), "vs", fmt_pct(a["beemo_auroc"]["mean"]))
    print("CEDAR ESL E1 FPR (allcal / e0cal):", fmt_pct(c["esl_E1_fpr"]["mean"]), "/", fmt_pct(c["esl_E1_fpr_e0cal"]["mean"]))
    print("CEDAR E4 share at cov 50/60/70/80 (margin vs intens):",
          [(k, fmt_pct(c["rc_margin_e4_share"][k]["mean"]), fmt_pct(c["rc_intens_e4_share"][k]["mean"])) for k in ["cov_50","cov_60","cov_70","cov_80"]])
    print("Headline WG boot:", sig["mb_cedar_full_vs_mb_augmented_erm"]["test_id_worst_group_E0_E3"])
    print("MixSet Pure boot:", sig["mb_cedar_full_vs_mb_augmented_erm"]["external_mixset_pure_auroc"])
    print("MixSet All boot:", sig["mb_cedar_full_vs_mb_augmented_erm"]["external_mixset_all_auroc"])


if __name__ == "__main__":
    main()

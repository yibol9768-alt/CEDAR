#!/usr/bin/env bash
set -euo pipefail

COLAB="$HOME/.local/bin/colab"
ROOT="/Users/liuyibo/Documents/antigravity/quirky-babbage/aigt_detection_iclr"
mkdir -p "$ROOT/results/model_preds_v2"

cleanup_w2() {
    echo "=== Ensuring Worker 2 (aigt_iclr_w2) is stopped cleanly ==="
    "$COLAB" stop -s aigt_iclr_w2 || true
}
trap cleanup_w2 EXIT

echo "=== [W2-1] Provisioning Worker 2 (aigt_iclr_w2, G4 102GB Blackwell) ==="
"$COLAB" new -s aigt_iclr_w2 --gpu G4
echo 'import subprocess, os; subprocess.run(["uv", "pip", "install", "--system", "-q", "scikit-learn", "scipy", "accelerate", "safetensors", "sentencepiece"], check=True); os.makedirs("/content/cedar_data/model_preds_v2", exist_ok=True); os.makedirs("/content/src", exist_ok=True); os.makedirs("/content/scripts", exist_ok=True); open("/content/src/__init__.py","w").close(); open("/content/scripts/__init__.py","w").close(); print("W2 READY")' | "$COLAB" exec -s aigt_iclr_w2 --timeout 300

for f in split_train.json split_val.json split_test_id.json split_test_ood_ar.json split_test_ood_llada.json external_esl_full.json external_mixset.json external_beemo.json external_mage_wild.json; do
    "$COLAB" upload -s aigt_iclr_w2 "$ROOT/data/$f" "/content/cedar_data/$f"
done

"$COLAB" upload -s aigt_iclr_w2 "$ROOT/src/cedar_model.py" "/content/src/cedar_model.py"
"$COLAB" upload -s aigt_iclr_w2 "$ROOT/scripts/step4_train_cedar_and_baselines.py" "/content/scripts/step4_train_cedar_and_baselines.py"
"$COLAB" upload -s aigt_iclr_w2 "$ROOT/scripts/step4b_nsd_rank_and_spectrum.py" "/content/scripts/step4b_nsd_rank_and_spectrum.py"
"$COLAB" upload -s aigt_iclr_w2 "$ROOT/scripts/step4_worker2_bottom_up.py" "/content/scripts/step4_worker2_bottom_up.py"

sync_run_group() {
    for rname in "$@"; do
        for sd in 42 1042 2042; do
            fn="${rname}_seed${sd}.json"
            "$COLAB" download -s aigt_iclr_w2 "/content/cedar_data/model_preds_v2/$fn" "$ROOT/results/model_preds_v2/$fn"
            "$COLAB" upload -s aigt_iclr_g4 "$ROOT/results/model_preds_v2/$fn" "/content/cedar_data/model_preds_v2/$fn"
        done
    done
}

echo "=== [W2-2] Step 4b already completed & synced locally — proceeding straight to DeBERTa-v3-large! ==="

echo "=== [W2-3] Running DeBERTa-v3-large Suite (12 runs) ==="
"$COLAB" restart-kernel -s aigt_iclr_w2
"$COLAB" exec -s aigt_iclr_w2 --env W2_CHUNK=deberta -f "$ROOT/scripts/step4_worker2_bottom_up.py" --timeout 3600
sync_run_group db_unedited_erm db_augmented_erm db_supcon_orth_disentangle db_cedar_full

echo "=== [W2-4] Running Bottom 3 ModernBERT Ablations (9 runs) ==="
"$COLAB" restart-kernel -s aigt_iclr_w2
"$COLAB" exec -s aigt_iclr_w2 --env W2_CHUNK=abl_bottom3 -f "$ROOT/scripts/step4_worker2_bottom_up.py" --timeout 3600
sync_run_group mb_ablation_unprotected_nsd mb_ablation_no_gdro mb_ablation_no_pair

echo "=== [W2-5] Running Top 3 ModernBERT Ablations (9 runs) ==="
"$COLAB" restart-kernel -s aigt_iclr_w2
"$COLAB" exec -s aigt_iclr_w2 --env W2_CHUNK=abl_top3 -f "$ROOT/scripts/step4_worker2_bottom_up.py" --timeout 3600
sync_run_group mb_ablation_param_matched_mlp mb_ablation_no_dte mb_ablation_no_nsd

echo "=== WORKER 2 COMPLETE — ALL BOTTOM-UP RUNS SYNCED LOCALLY & STAGED TO G4! ==="

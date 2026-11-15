#!/usr/bin/env bash
set -euo pipefail

COLAB="$HOME/.local/bin/colab"
ROOT="/Users/liuyibo/Documents/antigravity/quirky-babbage/aigt_detection_iclr"
SCRIPT="$ROOT/scripts/step2_generate_symmetric_matrix.py"
mkdir -p "$ROOT/data" "$ROOT/results"

echo "=== [Stage 2A] Parallel Draft Generation on G4 (Qwen2.5 + LLaDA-8B) & A100 (Zephyr + OLMo-2) ==="
"$COLAB" restart-kernel -s aigt_iclr_g4
"$COLAB" exec -s aigt_iclr_g4 --env CEDAR_STAGE=stage2a_g4 -f "$SCRIPT" --timeout 3600 &
P_G4=$!
"$COLAB" exec -s aigt_iclr_dev --env CEDAR_STAGE=stage2a_a100 -f "$SCRIPT" --timeout 3600 &
P_A100=$!
wait $P_G4 $P_A100

echo "=== [Checkpoint 2A] Downloading all 4 draft caches locally & cross-syncing ==="
"$COLAB" download -s aigt_iclr_g4 "/content/cedar_data/cache_drafts_qwen.json" "$ROOT/data/cache_drafts_qwen.json"
"$COLAB" download -s aigt_iclr_g4 "/content/cedar_data/cache_drafts_llada.json" "$ROOT/data/cache_drafts_llada.json"
"$COLAB" download -s aigt_iclr_dev "/content/cedar_data/cache_drafts_zephyr.json" "$ROOT/data/cache_drafts_zephyr.json"
"$COLAB" download -s aigt_iclr_dev "/content/cedar_data/cache_drafts_olmo2.json" "$ROOT/data/cache_drafts_olmo2.json"

"$COLAB" upload -s aigt_iclr_g4 "$ROOT/data/cache_drafts_zephyr.json" "/content/cedar_data/cache_drafts_zephyr.json"
"$COLAB" upload -s aigt_iclr_g4 "$ROOT/data/cache_drafts_olmo2.json" "/content/cedar_data/cache_drafts_olmo2.json"
"$COLAB" upload -s aigt_iclr_dev "$ROOT/data/cache_drafts_qwen.json" "/content/cedar_data/cache_drafts_qwen.json"
"$COLAB" upload -s aigt_iclr_dev "$ROOT/data/cache_drafts_llada.json" "/content/cedar_data/cache_drafts_llada.json"

echo "=== [Stage 2B] Parallel Symmetric Edits on G4 (Qwen2.5 + Phi-4-mini) & A100 (Falcon3) ==="
"$COLAB" restart-kernel -s aigt_iclr_g4
"$COLAB" restart-kernel -s aigt_iclr_dev
"$COLAB" exec -s aigt_iclr_g4 --env CEDAR_STAGE=stage2b_g4 -f "$SCRIPT" --timeout 5400 &
P_G4_B=$!
"$COLAB" exec -s aigt_iclr_dev --env CEDAR_STAGE=stage2b_a100 -f "$SCRIPT" --timeout 5400 &
P_A100_B=$!
wait $P_G4_B $P_A100_B

echo "=== [Checkpoint 2B] Backing up all edit caches locally & releasing helper A100 immediately ==="
"$COLAB" download -s aigt_iclr_g4 "/content/cedar_data/cache_edits_qwen.json" "$ROOT/data/cache_edits_qwen.json"
"$COLAB" download -s aigt_iclr_g4 "/content/cedar_data/cache_e4_regen_qwen.json" "$ROOT/data/cache_e4_regen_qwen.json"
"$COLAB" download -s aigt_iclr_g4 "/content/cedar_data/cache_edits_phi4_ood.json" "$ROOT/data/cache_edits_phi4_ood.json"
"$COLAB" download -s aigt_iclr_dev "/content/cedar_data/cache_edits_falcon3.json" "$ROOT/data/cache_edits_falcon3.json"
"$COLAB" download -s aigt_iclr_dev "/content/cedar_data/cache_e4_regen_falcon3.json" "$ROOT/data/cache_e4_regen_falcon3.json"

echo "=== Stopping temporary helper session aigt_iclr_dev ==="
"$COLAB" stop -s aigt_iclr_dev

echo "=== [Stage 2C] Uploading Falcon3 caches to G4, assembling final splits & backing up locally ==="
"$COLAB" upload -s aigt_iclr_g4 "$ROOT/data/cache_edits_falcon3.json" "/content/cedar_data/cache_edits_falcon3.json"
"$COLAB" upload -s aigt_iclr_g4 "$ROOT/data/cache_e4_regen_falcon3.json" "/content/cedar_data/cache_e4_regen_falcon3.json"
"$COLAB" exec -s aigt_iclr_g4 --env CEDAR_STAGE=assemble -f "$SCRIPT" --timeout 600

for f in split_train.json split_val.json split_test_id.json split_test_ood_ar.json split_test_ood_llada.json external_esl_full.json; do
    "$COLAB" download -s aigt_iclr_g4 "/content/cedar_data/$f" "$ROOT/data/$f"
done

echo "=== Uploading v2 CEDAR source & scripts to aigt_iclr_g4 ==="
echo "import os; os.makedirs('/content/src', exist_ok=True); os.makedirs('/content/scripts', exist_ok=True); open('/content/src/__init__.py','w').close(); open('/content/scripts/__init__.py','w').close(); print('Dirs ready')" | "$COLAB" exec -s aigt_iclr_g4
"$COLAB" upload -s aigt_iclr_g4 "$ROOT/src/cedar_model.py" "/content/src/cedar_model.py"
"$COLAB" upload -s aigt_iclr_g4 "$ROOT/scripts/step3_eval_pretrained_and_probes.py" "/content/scripts/step3_eval_pretrained_and_probes.py"
"$COLAB" upload -s aigt_iclr_g4 "$ROOT/scripts/step4_train_cedar_and_baselines.py" "/content/scripts/step4_train_cedar_and_baselines.py"
"$COLAB" upload -s aigt_iclr_g4 "$ROOT/scripts/step4b_nsd_rank_and_spectrum.py" "/content/scripts/step4b_nsd_rank_and_spectrum.py"
"$COLAB" upload -s aigt_iclr_g4 "$ROOT/scripts/step5_statistical_analysis_and_figures.py" "/content/scripts/step5_statistical_analysis_and_figures.py"

echo "=== ALL DUAL-GPU GENERATION + LOCAL BACKUP + V2 CODE SYNC COMPLETE! ==="

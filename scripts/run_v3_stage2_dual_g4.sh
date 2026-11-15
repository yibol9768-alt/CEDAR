#!/usr/bin/env bash
set -euo pipefail

COLAB="$HOME/.local/bin/colab"
ROOT="/Users/liuyibo/Documents/antigravity/quirky-babbage/aigt_detection_iclr"
SCRIPT="$ROOT/scripts/v3_build_v2_benchmark.py"
mkdir -p "$ROOT/data_v2" "$ROOT/results_v3"

echo "=== [Step 0] Uploading updated v3 code to cedar_v3_g4a & cedar_v3_g4b (v1 data already unpacked) ==="
for SESS in cedar_v3_g4a cedar_v3_g4b; do
    echo "import os; [os.makedirs(p, exist_ok=True) for p in ('/content/src','/content/scripts','/content/cedar_data_v2','/content/cedar_results_v3')]; print('$SESS dirs ready')" | "$COLAB" exec -s "$SESS" --timeout 60
    "$COLAB" upload -s "$SESS" "$ROOT/src/cedar_v2_model.py" /content/src/cedar_v2_model.py
    "$COLAB" upload -s "$SESS" "$SCRIPT" /content/scripts/v3_build_v2_benchmark.py
done

echo "=== [Step 1] Curating disjoint 800-parent v2 train + 160-parent pristine confirm + HF commit SHAs on cedar_v3_g4a ==="
"$COLAB" exec -s cedar_v3_g4a --env CEDAR_V3_STAGE=curate -f "$SCRIPT" --timeout 900

for f in provenance_lock.json requirements_lock.txt val_manifest.json parents_and_schedule_v2.json; do
    "$COLAB" download -s cedar_v3_g4a "/content/cedar_data_v2/$f" "$ROOT/data_v2/$f"
done
"$COLAB" upload -s cedar_v3_g4b "$ROOT/data_v2/provenance_lock.json" /content/cedar_data_v2/provenance_lock.json
"$COLAB" upload -s cedar_v3_g4b "$ROOT/data_v2/parents_and_schedule_v2.json" /content/cedar_data_v2/parents_and_schedule_v2.json

echo "=== [Step 2A] Parallel Draft Generation: g4a (Qwen-7B + Granite-3.1-8B confirm) & g4b (Zephyr-7B) ==="
"$COLAB" restart-kernel -s cedar_v3_g4a
"$COLAB" restart-kernel -s cedar_v3_g4b
"$COLAB" exec -s cedar_v3_g4a --env CEDAR_V3_STAGE=drafts_a -f "$SCRIPT" --timeout 3600 &
PID_A=$!
"$COLAB" exec -s cedar_v3_g4b --env CEDAR_V3_STAGE=drafts_b -f "$SCRIPT" --timeout 3600 &
PID_B=$!
wait $PID_A $PID_B

echo "=== [Checkpoint 2A] Backing up draft caches locally & cross-syncing ==="
"$COLAB" download -s cedar_v3_g4a /content/cedar_data_v2/v2_cache_drafts_qwen.json "$ROOT/data_v2/v2_cache_drafts_qwen.json"
"$COLAB" download -s cedar_v3_g4a /content/cedar_data_v2/v2_cache_drafts_granite_confirm.json "$ROOT/data_v2/v2_cache_drafts_granite_confirm.json"
"$COLAB" download -s cedar_v3_g4b /content/cedar_data_v2/v2_cache_drafts_zephyr.json "$ROOT/data_v2/v2_cache_drafts_zephyr.json"

"$COLAB" upload -s cedar_v3_g4a "$ROOT/data_v2/v2_cache_drafts_zephyr.json" /content/cedar_data_v2/v2_cache_drafts_zephyr.json
"$COLAB" upload -s cedar_v3_g4b "$ROOT/data_v2/v2_cache_drafts_qwen.json" /content/cedar_data_v2/v2_cache_drafts_qwen.json
"$COLAB" upload -s cedar_v3_g4b "$ROOT/data_v2/v2_cache_drafts_granite_confirm.json" /content/cedar_data_v2/v2_cache_drafts_granite_confirm.json

echo "=== [Step 2B] Parallel Bidirectional Edits: g4a (Qwen-7B + Phi-4-mini confirm) & g4b (Falcon3-7B) ==="
"$COLAB" restart-kernel -s cedar_v3_g4a
"$COLAB" restart-kernel -s cedar_v3_g4b
"$COLAB" exec -s cedar_v3_g4a --env CEDAR_V3_STAGE=edits_a -f "$SCRIPT" --timeout 5400 &
PID_EA=$!
"$COLAB" exec -s cedar_v3_g4b --env CEDAR_V3_STAGE=edits_b -f "$SCRIPT" --timeout 5400 &
PID_EB=$!
wait $PID_EA $PID_EB

echo "=== [Checkpoint 2B] Backing up edit caches locally & assembling final v2 splits ==="
"$COLAB" download -s cedar_v3_g4a /content/cedar_data_v2/v2_cache_edits_qwen.json "$ROOT/data_v2/v2_cache_edits_qwen.json"
"$COLAB" download -s cedar_v3_g4a /content/cedar_data_v2/v2_cache_edits_phi4_confirm.json "$ROOT/data_v2/v2_cache_edits_phi4_confirm.json"
"$COLAB" download -s cedar_v3_g4b /content/cedar_data_v2/v2_cache_edits_falcon3.json "$ROOT/data_v2/v2_cache_edits_falcon3.json"

"$COLAB" upload -s cedar_v3_g4a "$ROOT/data_v2/v2_cache_edits_falcon3.json" /content/cedar_data_v2/v2_cache_edits_falcon3.json
"$COLAB" exec -s cedar_v3_g4a --env CEDAR_V3_STAGE=assemble -f "$SCRIPT" --timeout 600

for f in split_train_fit_v2.json split_train_op_v2.json split_train_v2.json split_test_confirm.json provenance_lock.json; do
    "$COLAB" download -s cedar_v3_g4a "/content/cedar_data_v2/$f" "$ROOT/data_v2/$f"
done

# Sync assembled splits to cedar_v3_g4b as well so both GPUs are ready for immediate 5-seed training!
tar -czf /tmp/cedar_v2_splits_bundle.tar.gz -C "$ROOT/data_v2" \
    val_manifest.json split_train_fit_v2.json split_train_op_v2.json split_train_v2.json split_test_confirm.json provenance_lock.json
"$COLAB" upload -s cedar_v3_g4b /tmp/cedar_v2_splits_bundle.tar.gz /content/cedar_v2_splits_bundle.tar.gz
echo "import tarfile; tarfile.open('/content/cedar_v2_splits_bundle.tar.gz').extractall('/content/cedar_data_v2'); print('cedar_v3_g4b synced!')" | "$COLAB" exec -s cedar_v3_g4b --timeout 60

echo "=== ALL V2 BIDIRECTIONAL SYMMETRIC DATA + PRISTINE CONFIRMATION BENCHMARK COMPLETE ON BOTH G4 GPUS! ==="

#!/usr/bin/env bash
set -euo pipefail

COLAB="$HOME/.local/bin/colab"
ROOT="/Users/liuyibo/Documents/antigravity/quirky-babbage/aigt_detection_iclr"
PRED_LOCAL="$ROOT/results_v3/model_preds_v3"
CKPT_LOCAL="$ROOT/results_v3/checkpoints"
mkdir -p "$PRED_LOCAL" "$CKPT_LOCAL"

echo "=== [1/4] Packing data + data_v2 + src + scripts bundle for cedar_v3_g4a2, cedar_v3_g4b2, cedar_v3_a100d2 ==="
tar -czf /tmp/cedar_full_v3_bundle.tar.gz -C "$ROOT" \
    src/cedar_model.py src/cedar_v2_model.py \
    scripts/v3_train_cedar_v2_suite.py scripts/v3_bottom_up_tail_worker.py scripts/v3_theorem2_convergence_and_geometry.py \
    data/split_train.json data/split_val.json data/split_test_id.json data/split_test_ood_ar.json data/split_test_ood_llada.json \
    data/external_esl_full.json data/external_mixset.json data/external_beemo.json data/external_mage_wild.json \
    data_v2/provenance_lock.json data_v2/val_manifest.json \
    data_v2/split_train_fit_v2.json data_v2/split_train_op_v2.json data_v2/split_train_v2.json data_v2/split_test_confirm.json

for SESS in cedar_v3_g4a2 cedar_v3_g4b2 cedar_v3_a100d2; do
    "$COLAB" upload -s "$SESS" /tmp/cedar_full_v3_bundle.tar.gz /content/cedar_full_v3_bundle.tar.gz
    echo "import os, tarfile, shutil
os.makedirs('/tmp/ext_v3', exist_ok=True)
tarfile.open('/content/cedar_full_v3_bundle.tar.gz').extractall('/tmp/ext_v3')
for src_d, dst_d in [('src','/content/src'),('scripts','/content/scripts'),('data','/content/cedar_data'),('data_v2','/content/cedar_data_v2')]:
    shutil.copytree(f'/tmp/ext_v3/{src_d}', dst_d, dirs_exist_ok=True)
os.makedirs('/content/cedar_results_v3/model_preds_v3', exist_ok=True)
os.makedirs('/content/cedar_results_v3/checkpoints', exist_ok=True)
print('$SESS unpacked & ready!')
" | "$COLAB" exec -s "$SESS" --timeout 120
done

echo "=== [2/4] Launching 3 workers (g4a2: Shard0 Top-Down, g4b2: Shard1 Top-Down, a100d2: Shard1 Bottom-Up; g4c is already running Shard0 Bottom-Up!) ==="
"$COLAB" exec -s cedar_v3_g4a2 --env CEDAR_WORKER_ID=0 --env CEDAR_NUM_WORKERS=2 -f "$ROOT/scripts/v3_train_cedar_v2_suite.py" --timeout 14400 >/tmp/v3_g4a2.log 2>&1 &
PID_A2=$!

"$COLAB" exec -s cedar_v3_g4b2 --env CEDAR_WORKER_ID=1 --env CEDAR_NUM_WORKERS=2 -f "$ROOT/scripts/v3_train_cedar_v2_suite.py" --timeout 14400 >/tmp/v3_g4b2.log 2>&1 &
PID_B2=$!

"$COLAB" exec -s cedar_v3_a100d2 --env CEDAR_WORKER_ID=1 --env CEDAR_NUM_WORKERS=2 --env CEDAR_MB_TAIL_COUNT=24 -f "$ROOT/scripts/v3_bottom_up_tail_worker.py" --timeout 14400 >/tmp/v3_a100d2.log 2>&1 &
PID_D2=$!

sync_from_sessions() {
    for SESS in cedar_v3_g4a2 cedar_v3_g4b2 cedar_v3_g4c cedar_v3_a100d2; do
        FILES=$("$COLAB" ls -s "$SESS" /content/cedar_results_v3/model_preds_v3 2>/dev/null | grep -E '^[a-zA-Z0-9_]+\.json$' || true)
        for FN in $FILES; do
            if [ ! -s "$PRED_LOCAL/$FN" ]; then
                "$COLAB" download -s "$SESS" "/content/cedar_results_v3/model_preds_v3/$FN" "$PRED_LOCAL/$FN.part" >/dev/null 2>&1 && mv "$PRED_LOCAL/$FN.part" "$PRED_LOCAL/$FN" || rm -f "$PRED_LOCAL/$FN.part"
            fi
        done
        CKPTS=$("$COLAB" ls -s "$SESS" /content/cedar_results_v3/checkpoints 2>/dev/null | grep -E '^[a-zA-Z0-9_]+\.pt$' || true)
        for CK in $CKPTS; do
            if [ ! -s "$CKPT_LOCAL/$CK" ]; then
                "$COLAB" download -s "$SESS" "/content/cedar_results_v3/checkpoints/$CK" "$CKPT_LOCAL/$CK.part" >/dev/null 2>&1 && mv "$CKPT_LOCAL/$CK.part" "$CKPT_LOCAL/$CK" || rm -f "$CKPT_LOCAL/$CK.part"
            fi
        done
    done
}

echo "=== [3/4] Entering 90-second Keepalive + Incremental Live-Sync Loop until 116/116 JSONs arrive ==="
while true; do
    sync_from_sessions
    CNT=$(ls -1 "$PRED_LOCAL"/*.json 2>/dev/null | wc -l | tr -d ' ')
    CKCNT=$(ls -1 "$CKPT_LOCAL"/*.pt 2>/dev/null | wc -l | tr -d ' ')
    echo "[LIVE SYNC $(date '+%H:%M:%S')] Local predictions: ${CNT}/116 | Local checkpoints: ${CKCNT}"
    if [ "$CNT" -ge 116 ] && [ -s "$CKPT_LOCAL/mb_v3_cedar_split_op_seed42.pt" ] && [ -s "$CKPT_LOCAL/mb_v3_strong_combined_no_nsd_seed42.pt" ] && [ -s "$CKPT_LOCAL/db_v3_cedar_split_op_seed42.pt" ]; then
        echo "=== ALL 116 PREDICTION JSONS & ALL CANONICAL SEED-42 CHECKPOINTS SYNCED LOCALLY! ==="
        break
    fi
    if ! kill -0 $PID_A2 2>/dev/null && ! kill -0 $PID_B2 2>/dev/null && ! kill -0 $PID_D2 2>/dev/null; then
        sync_from_sessions
        CNT=$(ls -1 "$PRED_LOCAL"/*.json 2>/dev/null | wc -l | tr -d ' ')
        echo "[WORKERS EXITED] Final local count: ${CNT}/116"
        break
    fi
    sleep 90
done

kill $PID_A2 $PID_B2 $PID_D2 2>/dev/null || true

echo "=== [4/4] Running Theorem 2 Disjoint-Pool Convergence & Geometry on cedar_v3_g4a2 ==="
"$COLAB" restart-kernel -s cedar_v3_g4a2
if [ -s "$CKPT_LOCAL/mb_v3_cedar_split_op_seed42.pt" ]; then
    "$COLAB" upload -s cedar_v3_g4a2 "$CKPT_LOCAL/mb_v3_cedar_split_op_seed42.pt" /content/cedar_results_v3/checkpoints/mb_v3_cedar_split_op_seed42.pt || true
fi
"$COLAB" exec -s cedar_v3_g4a2 -f "$ROOT/scripts/v3_theorem2_convergence_and_geometry.py" --timeout 1800
"$COLAB" download -s cedar_v3_g4a2 /content/cedar_results_v3/nsd_v3_theorem2_and_geometry.json "$ROOT/results_v3/nsd_v3_theorem2_and_geometry.json"

echo "=== Stopping ONLY our 4 sessions (cedar_v3_g4a2, cedar_v3_g4b2, cedar_v3_g4c, cedar_v3_a100d2) ==="
for SESS in cedar_v3_g4a2 cedar_v3_g4b2 cedar_v3_g4c cedar_v3_a100d2; do
    "$COLAB" stop -s "$SESS" || true
done
"$COLAB" usage
echo "=== STAGE 4 + THEOREM 2 + FULL HARVEST + GPU RELEASE 100% COMPLETE! ==="

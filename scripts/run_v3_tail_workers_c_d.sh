#!/usr/bin/env bash
set -euo pipefail

COLAB="$HOME/.local/bin/colab"
ROOT="/Users/liuyibo/Documents/antigravity/quirky-babbage/aigt_detection_iclr"

echo "=== Packing data + data_v2 + src + scripts bundle for cedar_v3_g4c & cedar_v3_a100d ==="
tar -czf /tmp/cedar_tail_bundle.tar.gz -C "$ROOT" \
    src/cedar_model.py src/cedar_v2_model.py \
    scripts/v3_train_cedar_v2_suite.py scripts/v3_bottom_up_tail_worker.py \
    data/split_train.json data/split_val.json data/split_test_id.json data/split_test_ood_ar.json data/split_test_ood_llada.json \
    data/external_esl_full.json data/external_mixset.json data/external_beemo.json data/external_mage_wild.json \
    data_v2/provenance_lock.json data_v2/val_manifest.json \
    data_v2/split_train_fit_v2.json data_v2/split_train_op_v2.json data_v2/split_train_v2.json data_v2/split_test_confirm.json

for SESS in cedar_v3_g4c cedar_v3_a100d; do
    "$COLAB" upload -s "$SESS" /tmp/cedar_tail_bundle.tar.gz /content/cedar_tail_bundle.tar.gz
    echo "import os, tarfile, shutil
os.makedirs('/tmp/ext_tail', exist_ok=True)
tarfile.open('/content/cedar_tail_bundle.tar.gz').extractall('/tmp/ext_tail')
for src_d, dst_d in [('src','/content/src'),('scripts','/content/scripts'),('data','/content/cedar_data'),('data_v2','/content/cedar_data_v2')]:
    shutil.copytree(f'/tmp/ext_tail/{src_d}', dst_d, dirs_exist_ok=True)
os.makedirs('/content/cedar_results_v3/model_preds_v3', exist_ok=True)
os.makedirs('/content/cedar_results_v3/checkpoints', exist_ok=True)
print('$SESS unpacked & ready!')
" | "$COLAB" exec -s "$SESS" --timeout 120
done

echo "=== Launching Bottom-Up Tail Workers on cedar_v3_g4c (Tail 0) & cedar_v3_a100d (Tail 1) ==="
"$COLAB" exec -s cedar_v3_g4c --env CEDAR_WORKER_ID=0 --env CEDAR_NUM_WORKERS=2 --env CEDAR_MB_TAIL_COUNT=20 -f "$ROOT/scripts/v3_bottom_up_tail_worker.py" --timeout 14400 &
PID_C=$!
"$COLAB" exec -s cedar_v3_a100d --env CEDAR_WORKER_ID=1 --env CEDAR_NUM_WORKERS=2 --env CEDAR_MB_TAIL_COUNT=20 -f "$ROOT/scripts/v3_bottom_up_tail_worker.py" --timeout 14400 &
PID_D=$!
wait $PID_C $PID_D
echo "=== BOTH BOTTOM-UP TAIL WORKERS COMPLETED! ==="

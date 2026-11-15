#!/usr/bin/env bash
set -euo pipefail

COLAB="$HOME/.local/bin/colab"
ROOT="/Users/liuyibo/Documents/antigravity/quirky-babbage/aigt_detection_iclr"
mkdir -p "$ROOT/results_v3/model_preds_v3" "$ROOT/results_v3/checkpoints"

echo "=== [Stage 4.0] Restarting kernels & uploading SDPA + offline-locked v3 training & Theorem 2 scripts ==="
for SESS in cedar_v3_g4a cedar_v3_g4b; do
    "$COLAB" restart-kernel -s "$SESS"
    "$COLAB" upload -s "$SESS" "$ROOT/data_v2/provenance_lock.json" /content/cedar_data_v2/provenance_lock.json
    "$COLAB" upload -s "$SESS" "$ROOT/src/cedar_model.py" /content/src/cedar_model.py
    "$COLAB" upload -s "$SESS" "$ROOT/src/cedar_v2_model.py" /content/src/cedar_v2_model.py
    "$COLAB" upload -s "$SESS" "$ROOT/scripts/v3_train_cedar_v2_suite.py" /content/scripts/v3_train_cedar_v2_suite.py
    "$COLAB" upload -s "$SESS" "$ROOT/scripts/v3_theorem2_convergence_and_geometry.py" /content/scripts/v3_theorem2_convergence_and_geometry.py
done

echo "=== [Stage 4.1] Launching Parallel Dual-G4 116-Run Suite + Theorem 2 ==="
(
    "$COLAB" exec -s cedar_v3_g4a --env CEDAR_WORKER_ID=0 --env CEDAR_NUM_WORKERS=2 -f "$ROOT/scripts/v3_train_cedar_v2_suite.py" --timeout 14400
    "$COLAB" restart-kernel -s cedar_v3_g4a
    "$COLAB" exec -s cedar_v3_g4a -f "$ROOT/scripts/v3_theorem2_convergence_and_geometry.py" --timeout 1800
    "$COLAB" download -s cedar_v3_g4a /content/cedar_results_v3/nsd_v3_theorem2_and_geometry.json "$ROOT/results_v3/nsd_v3_theorem2_and_geometry.json"
) &
PID_W0=$!

(
    "$COLAB" exec -s cedar_v3_g4b --env CEDAR_WORKER_ID=1 --env CEDAR_NUM_WORKERS=2 -f "$ROOT/scripts/v3_train_cedar_v2_suite.py" --timeout 14400
) &
PID_W1=$!

wait $PID_W0 $PID_W1

echo "=== [Stage 4.5] Downloading all 116 prediction JSONs & canonical seed-42 checkpoints ==="
echo "import tarfile, glob; tf=tarfile.open('/content/preds_w0.tar.gz','w:gz'); [tf.add(p, arcname=p.split('/')[-1]) for p in glob.glob('/content/cedar_results_v3/model_preds_v3/*.json')]; tf.close(); print('w0 tar ready')" | "$COLAB" exec -s cedar_v3_g4a --timeout 120
echo "import tarfile, glob; tf=tarfile.open('/content/preds_w1.tar.gz','w:gz'); [tf.add(p, arcname=p.split('/')[-1]) for p in glob.glob('/content/cedar_results_v3/model_preds_v3/*.json')]; tf.close(); print('w1 tar ready')" | "$COLAB" exec -s cedar_v3_g4b --timeout 120

"$COLAB" download -s cedar_v3_g4a /content/preds_w0.tar.gz /tmp/preds_w0.tar.gz
"$COLAB" download -s cedar_v3_g4b /content/preds_w1.tar.gz /tmp/preds_w1.tar.gz
tar -xzf /tmp/preds_w0.tar.gz -C "$ROOT/results_v3/model_preds_v3"
tar -xzf /tmp/preds_w1.tar.gz -C "$ROOT/results_v3/model_preds_v3"

echo "Downloaded JSON count: $(ls "$ROOT/results_v3/model_preds_v3"/*.json | wc -l)"

"$COLAB" download -s cedar_v3_g4a /content/cedar_results_v3/checkpoints/mb_v3_cedar_split_op_seed42.pt "$ROOT/results_v3/checkpoints/mb_v3_cedar_split_op_seed42.pt"
"$COLAB" download -s cedar_v3_g4b /content/cedar_results_v3/checkpoints/mb_v3_strong_combined_no_nsd_seed42.pt "$ROOT/results_v3/checkpoints/mb_v3_strong_combined_no_nsd_seed42.pt"
"$COLAB" download -s cedar_v3_g4b /content/cedar_results_v3/checkpoints/db_v3_cedar_split_op_seed42.pt "$ROOT/results_v3/checkpoints/db_v3_cedar_split_op_seed42.pt"

echo "=== [Stage 4.6] Stopping ONLY our two sessions (cedar_v3_g4a & cedar_v3_g4b) ==="
"$COLAB" stop -s cedar_v3_g4a
"$COLAB" stop -s cedar_v3_g4b
"$COLAB" sessions

echo "=== ALL V3 GPU EXPERIMENTS COMPLETED & G4 SESSIONS CLEANLY RELEASED! ==="

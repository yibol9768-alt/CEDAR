#!/usr/bin/env bash
set -euo pipefail

SESSION="aigt_iclr_dev"
COLAB="$HOME/.local/bin/colab"
ROOT="/Users/liuyibo/Documents/antigravity/quirky-babbage/aigt_detection_iclr"

echo "=== [1/3] Backing up generated splits & external datasets from Colab ==="
mkdir -p "$ROOT/data" "$ROOT/results"
for f in split_train.json split_val.json split_test_id.json split_test_ood_ar.json split_test_ood_llada.json external_esl_full.json external_mixset.json external_beemo.json external_mage_wild.json; do
    if [ ! -s "$ROOT/data/$f" ]; then
        "$COLAB" download -s "$SESSION" "/content/cedar_data/$f" "$ROOT/data/$f"
        echo "  Downloaded $f"
    else
        echo "  Already backed up $f"
    fi
done

echo "=== [2/3] Uploading v2 CEDAR source & scripts to Colab /content ==="
"$COLAB" exec -s "$SESSION" -c "import os; os.makedirs('/content/src', exist_ok=True); os.makedirs('/content/scripts', exist_ok=True); open('/content/src/__init__.py','w').close(); open('/content/scripts/__init__.py','w').close(); print('Dirs ready')"
"$COLAB" upload -s "$SESSION" "$ROOT/src/cedar_model.py" "/content/src/cedar_model.py"
"$COLAB" upload -s "$SESSION" "$ROOT/scripts/step3_eval_pretrained_and_probes.py" "/content/scripts/step3_eval_pretrained_and_probes.py"
"$COLAB" upload -s "$SESSION" "$ROOT/scripts/step4_train_cedar_and_baselines.py" "/content/scripts/step4_train_cedar_and_baselines.py"
"$COLAB" upload -s "$SESSION" "$ROOT/scripts/step4b_nsd_rank_and_spectrum.py" "/content/scripts/step4b_nsd_rank_and_spectrum.py"
"$COLAB" upload -s "$SESSION" "$ROOT/scripts/step5_statistical_analysis_and_figures.py" "/content/scripts/step5_statistical_analysis_and_figures.py"

echo "=== [3/3] Sync complete! ==="

#!/usr/bin/env bash
set -euo pipefail

COLAB="$HOME/.local/bin/colab"
ROOT="/Users/liuyibo/Documents/antigravity/quirky-babbage/aigt_detection_iclr"
mkdir -p "$ROOT/results/model_preds_v2"

echo "=== [1/4] Running Step 3: Pretrained Detectors (Fast-DetectGPT, Binoculars, RADAR, Desklib) & Shortcut Probes ==="
"$COLAB" restart-kernel -s aigt_iclr_g4
"$COLAB" exec -s aigt_iclr_g4 -f "$ROOT/scripts/step3_eval_pretrained_and_probes.py" --timeout 3600
"$COLAB" download -s aigt_iclr_g4 "/content/cedar_data/preds_pretrained_and_probes.json" "$ROOT/results/preds_pretrained_and_probes.json"
echo "  -> Backed up preds_pretrained_and_probes.json locally!"

echo "=== [2/4] Running Step 4: Multi-Seed CEDAR + Baselines + Ablations (ModernBERT-large & DeBERTa-v3-large) ==="
"$COLAB" restart-kernel -s aigt_iclr_g4
"$COLAB" exec -s aigt_iclr_g4 -f "$ROOT/scripts/step4_train_cedar_and_baselines.py" --timeout 7200

echo "=== [3/4] Running Step 4b: NSD Rank Sensitivity (r=8,16,64) & Spectral Geometry Diagnostics ==="
"$COLAB" restart-kernel -s aigt_iclr_g4
"$COLAB" exec -s aigt_iclr_g4 -f "$ROOT/scripts/step4b_nsd_rank_and_spectrum.py" --timeout 3600
"$COLAB" download -s aigt_iclr_g4 "/content/cedar_data/nsd_spectrum_and_geometry.json" "$ROOT/results/nsd_spectrum_and_geometry.json"
echo "  -> Backed up nsd_spectrum_and_geometry.json locally!"

echo "=== [4/4] Running Step 5: Parent-Cluster Split-Conformal & 1000x Paired Bootstrap Evaluation ==="
"$COLAB" exec -s aigt_iclr_g4 -f "$ROOT/scripts/step5_statistical_analysis_and_figures.py" --timeout 1800
"$COLAB" download -s aigt_iclr_g4 "/content/cedar_data/master_evaluation_summary_v2.json" "$ROOT/results/master_evaluation_summary_v2.json"
echo "=== ALL STEPS 3, 4, 4b, 5 COMPLETE AND SYNCED LOCALLY! ==="

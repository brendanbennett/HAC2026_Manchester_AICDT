#!/usr/bin/env bash
# Train the flow, check whether the operator contributes, reconstruct all ten models,
# and score the three public ones against their ground truth.
#
#   scripts/run_pipeline.sh [STEPS] [SAMPLES]
#
# Assumes scripts/fit_shapes.py has already written runs/corpus_codes.npz and the
# shared token decoder, and that a trained surrogate exists.
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-${FLOW_STEPS:-4000}}
SAMPLES=${2:-${RECON_SAMPLES:-6}}
PY=${PY:-.venv/bin/python}
BODIES=${BODIES:-40}
FLOW_PHASES=${FLOW_PHASES:-96}
FLOW_BATCH=${FLOW_BATCH:-2}
FLOW_OPERATOR_RES=${FLOW_OPERATOR_RES:-32}
FLOW_TRAIN_GEOMS=${FLOW_TRAIN_GEOMS:-28}
DECODER_FILE=${DECODER_FILE:-runs/token_decoder.pt}

echo "=== train $(date)"
$PY -u scripts/train_lpd.py --bodies "$BODIES" --steps "$STEPS" \
    --phases "$FLOW_PHASES" --batch "$FLOW_BATCH" \
    --operator-res "$FLOW_OPERATOR_RES" --train-geoms "$FLOW_TRAIN_GEOMS" \
    --decoder-file "$DECODER_FILE" --out runs/lpd_flow.pt

echo "=== ablation: does the operator contribute? $(date)"
$PY -u scripts/ablate_flow.py --draws 18 --phases "$FLOW_PHASES" \
    --operator-res "$FLOW_OPERATOR_RES" --decoder-file "$DECODER_FILE"

mkdir -p results/lpd
for M in 1 2 3 4 5 6 7 8 9 10; do
  P=$(printf "%02d" "$M")
  echo "=== reconstruct model $M"
  $PY -u scripts/reconstruct_lpd.py --model "$M" --samples "$SAMPLES" --res 48 \
      --decoder-file "$DECODER_FILE" --out "results/lpd/Asteroid$P.stl"
done

echo "=== score $(date)"
$PY hac26/scoring/voxel.py --stl results/lpd/Asteroid0{1,2,3}.stl \
    --models 1 2 3 --label "LPD flow"
$PY hac26/scoring/side_view.py --models 1 2 3 --recon-dir results/lpd
echo "=== done $(date)"

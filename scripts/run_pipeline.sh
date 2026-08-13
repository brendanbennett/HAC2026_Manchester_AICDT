#!/bin/bash
# Train the flow, check whether the operator contributes, reconstruct all ten models,
# and score the three public ones against their ground truth.
#
#   scripts/run_pipeline.sh [STEPS] [SAMPLES]
#
# Assumes scripts/fit_shapes.py has already written runs/corpus_codes.npz and the
# shared token decoder, and that a trained surrogate exists.
set -u
cd "$(dirname "$0")/.."
STEPS=${1:-4000}
SAMPLES=${2:-6}
PY=.venv/bin/python

echo "=== train $(date)"
$PY -u scripts/train_lpd.py --bodies 40 --steps "$STEPS" --phases 96 --batch 2 \
    --out runs/lpd_flow.pt

echo "=== ablation: does the operator contribute? $(date)"
$PY -u scripts/ablate_flow.py --draws 18 --corpus /tmp/lpd_corpus_96_g28_shared.npz

mkdir -p results/lpd
for M in 1 2 3 4 5 6 7 8 9 10; do
  P=$(printf "%02d" "$M")
  echo "=== reconstruct model $M"
  $PY -u scripts/reconstruct_lpd.py --model "$M" --samples "$SAMPLES" --res 48 \
      --out "results/lpd/Asteroid$P.stl"
done

echo "=== score $(date)"
$PY scoring/voxel.py --stl results/lpd/Asteroid0{1,2,3}.stl --label "LPD flow"
echo "=== done $(date)"

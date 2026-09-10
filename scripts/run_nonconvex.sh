#!/usr/bin/env bash
# Correct the convex answers by fitting the hull's reshaping and the carve together, then
# decide which corrections enter the submission.
#
#   scripts/run_nonconvex.sh
#
# The order is not arbitrary. The calibration comes first because everything downstream is a
# comparison of misfits and the residual it reports at the released shapes is the error those
# comparisons are made inside; notes/identifiability.md measures how small that residual has
# to be for a comparison to mean anything, and this script prints it before going on. The
# public model with a concavity comes next, because its shape is released and the gate that
# decides the scored models takes its threshold from that run. The scored models come last,
# and none of them reaches the submission except through scripts/select_answers.py.
#
# Every setting is a variable here and can be overridden from the environment.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CHANNEL=${CHANNEL:-blender}
DATA_DIR=${DATA_DIR:-dataset/raw}
OUT_DIR=${OUT_DIR:-results/gn}
CALIBRATION_MODEL=${CALIBRATION_MODEL:-3}   # the public body with a genuine concavity, whose
                                            # correction is what licenses the scored ones
SCORED=${SCORED:-"4 5 6 7 8 9 10"}
HOLD_OUT_GEOMS=${HOLD_OUT_GEOMS:-5}   # cameras kept out of each fit; their misfit is the
                                      # only honest test of the body that was written
PHASES=${PHASES:-48}
RESTARTS=${RESTARTS:-8}
CALIBRATE_STEPS=${CALIBRATE_STEPS:-600}
CALIBRATE_MODELS=${CALIBRATE_MODELS:-"1 3"}   # public bodies the instrument is fitted on. One
                                              # model error is shared across them, so a body
                                              # the chain cannot reproduce raises the error
                                              # admitted for every other body

mkdir -p "$OUT_DIR" logs

# the file the calibration writes for this channel, named by the code rather than here
INSTRUMENT=$($PY -c "import sys; sys.path.insert(0, 'scripts'); \
from calibrate import OUT_INSTRUMENT; print(OUT_INSTRUMENT['$CHANNEL'])")
if [ ! -f "$INSTRUMENT" ]; then
  echo "=== calibrate the $CHANNEL channel $(date)"
  $PY -u scripts/calibrate.py --channel "$CHANNEL" --steps "$CALIBRATE_STEPS" \
      --models $CALIBRATE_MODELS --data-dir "$DATA_DIR" \
      2>&1 | tee logs/calibrate_"$CHANNEL".log
else
  echo "=== calibrate: $INSTRUMENT is already there"
fi

echo "=== reconstruct the calibrating model $CALIBRATION_MODEL $(date)"
P=$(printf "%02d" "$CALIBRATION_MODEL")
$PY -u scripts/reconstruct_gn.py --model "$CALIBRATION_MODEL" --channel "$CHANNEL" \
    --data-dir "$DATA_DIR" --phases "$PHASES" --restarts "$RESTARTS" \
    --hold-out-geoms "$HOLD_OUT_GEOMS" --out "$OUT_DIR/Asteroid$P.stl" \
    2>&1 | tee logs/reconstruct_gn_"$P".log

for M in $SCORED; do
  P=$(printf "%02d" "$M")
  echo "=== reconstruct model $M $(date)"
  $PY -u scripts/reconstruct_gn.py --model "$M" --channel "$CHANNEL" \
      --data-dir "$DATA_DIR" --phases "$PHASES" --restarts "$RESTARTS" \
      --hold-out-geoms "$HOLD_OUT_GEOMS" --out "$OUT_DIR/Asteroid$P.stl" \
      2>&1 | tee logs/reconstruct_gn_"$P".log
done

echo "=== decide $(date)"
CP=$(printf "%02d" "$CALIBRATION_MODEL")
$PY scripts/select_answers.py --refined "$OUT_DIR" \
    --calibrate "$OUT_DIR/Asteroid$CP.json" --models $SCORED

echo "=== check the submission $(date)"
$PY scripts/check_submission.py results/submission
echo "=== done $(date)"

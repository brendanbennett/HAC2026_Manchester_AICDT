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
# public model with a concavity comes next, because its shape is released and a correction
# that moves it away from its truth is the failure the whole method has to be checked for.
# The scored models come last, and none of them reaches the submission except through
# scripts/select_answers.py.
#
# Two solvers search the same problem. reconstruct_gn.py sweeps a designed grid of starts and
# fits each by secant Gauss-Newton, which explores; reconstruct_map.py descends the same
# objective by the adjoint from the convex answer alone, which is a hundred times as much
# optimisation per render but only one basin. They minimise the same functional, measure the
# written body the same way and hold out the same cameras, so the two answers for a body are
# comparable and select_answers.py picks between them on the cameras neither fit saw. Each
# writes into its own directory, so neither can overwrite the other.
#
# Every body of every track is a unit of work. One already written, and newer than the
# instrument it was fitted against, is skipped, so a job that stops part way carries on where
# it stopped; and within a body both solvers checkpoint, so an interrupted fit resumes rather
# than restarting. TIME_BUDGET caps a single body, which is what keeps a queue of ten from
# being spent on the first of them.
#
# Every setting is a variable here and can be overridden from the environment.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CHANNEL=${CHANNEL:-blender}
DATA_DIR=${DATA_DIR:-dataset/raw}
TRACKS=${TRACKS:-"gn map"}            # the solvers to run, in order; either alone is valid
GN_DIR=${GN_DIR:-results/gn}
MAP_DIR=${MAP_DIR:-results/map}
CALIBRATION_MODEL=${CALIBRATION_MODEL:-3}   # the public body with a genuine concavity, whose
                                            # correction is the check that the method moves a
                                            # body toward its truth and not only its curves
SCORED=${SCORED:-"4 5 6 7 8 9 10"}
HOLD_OUT_GEOMS=${HOLD_OUT_GEOMS:-5}   # cameras kept out of each fit; their misfit is the
                                      # only honest test of the body that was written, and
                                      # both solvers take the same ones (data_io)
PHASES=${PHASES:-48}
RESTARTS=${RESTARTS:-9}                # the convex answer and one sweep of the designed
                                      # grid of caps; hac26/solvers/gauss_newton.py says why
MAP_STEPS=${MAP_STEPS:-300}            # descent steps of the gradient solver
TIME_BUDGET=${TIME_BUDGET:-0}          # seconds one body of one track may take; 0 is no cap
REDO=${REDO:-0}                        # 1 refits bodies that are already written
CALIBRATE_STEPS=${CALIBRATE_STEPS:-600}
CALIBRATE_MODELS=${CALIBRATE_MODELS:-"1 3"}   # public bodies the instrument is fitted on. One
                                              # model error is shared across them, so a body
                                              # the chain cannot reproduce raises the error
                                              # admitted for every other body

mkdir -p "$GN_DIR" "$MAP_DIR" logs

# Everything the run needs, checked while it is still cheap to fix. A missing wheel or an
# unbuilt extension otherwise surfaces hours in, and some of those failures do not look like
# what they are: a missing decimation package used to read as a body with no curves.
$PY scripts/preflight.py --data-dir "$DATA_DIR" || {
  echo "preflight failed; not starting a long run. See above." >&2; exit 1; }

# the file the calibration writes for this channel, named by the code rather than here
INSTRUMENT=$($PY -c "import sys; sys.path.insert(0, 'scripts'); \
from calibrate import OUT_INSTRUMENT; print(OUT_INSTRUMENT['$CHANNEL'])")
# Present is not enough: an instrument fitted against other cameras or another transfer loads
# into a different question, and every threshold downstream is measured in the model error it
# carries. Instrument.load refuses such a file, and refuses one that does not say what it was
# fitted against, so the test for reuse is that it loads rather than that it exists.
loads_here() {
  $PY -c "import sys; sys.path.insert(0, '.')
from hac26.forward.mesh.instrument import Instrument
Instrument.load('$INSTRUMENT')" >/dev/null 2>&1
}
if [ -f "$INSTRUMENT" ] && loads_here; then
  echo "=== calibrate: $INSTRUMENT is already there and was fitted against these cameras"
else
  if [ -f "$INSTRUMENT" ]; then
    echo "=== calibrate: $INSTRUMENT does not load against the cameras this tree renders,"
    echo "    so it is refitted. Commit the new one and later runs will skip this stage."
  fi
  echo "=== calibrate the $CHANNEL channel $(date)"
  $PY -u scripts/calibrate.py --channel "$CHANNEL" --steps "$CALIBRATE_STEPS" \
      --models $CALIBRATE_MODELS --data-dir "$DATA_DIR" \
      2>&1 | tee logs/calibrate_"$CHANNEL".log
  [ -f "$INSTRUMENT" ] || { echo "calibration wrote no instrument; stopping" >&2; exit 1; }
fi

dir_of() { case "$1" in gn) echo "$GN_DIR" ;; map) echo "$MAP_DIR" ;; esac; }

failed=""
# One body of one track. A body already written against this instrument is left alone; the
# solvers' own checkpoints handle an interruption inside a body. A body that fails does not
# stop the queue: the rest are still worth having, and select_answers.py keeps the convex
# answer wherever nothing was written.
fit_one() {
  local track="$1" M="$2" dir out js
  dir=$(dir_of "$track")
  out="$dir/Asteroid$(printf '%02d' "$M").stl"
  js="${out%.stl}.json"
  if [ "$REDO" != "1" ] && [ -s "$out" ] && [ -s "$js" ] && [ ! "$INSTRUMENT" -nt "$out" ]; then
    echo "  --- $track model $M: $out is current"
    return 0
  fi
  echo "=== $track model $M -> $out $(date)"
  local args=(--model "$M" --channel "$CHANNEL" --data-dir "$DATA_DIR"
              --calibration "$INSTRUMENT" --phases "$PHASES"
              --hold-out-geoms "$HOLD_OUT_GEOMS" --out "$out")
  case "$track" in
    gn)  args+=(--restarts "$RESTARTS")
         [ "$TIME_BUDGET" != "0" ] && args+=(--time-budget "$TIME_BUDGET")
         $PY -u scripts/reconstruct_gn.py "${args[@]}" \
           2>&1 | tee "logs/reconstruct_gn_$(printf '%02d' "$M").log" ;;
    map) args+=(--steps "$MAP_STEPS")
         [ "$TIME_BUDGET" != "0" ] && args+=(--time-budget "$TIME_BUDGET")
         $PY -u scripts/reconstruct_map.py "${args[@]}" \
           2>&1 | tee "logs/reconstruct_map_$(printf '%02d' "$M").log" ;;
  esac
  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "  --- $track model $M FAILED (exit $status)"
    failed="$failed $track/$M"
  fi
  return 0
}

for track in $TRACKS; do
  fit_one "$track" "$CALIBRATION_MODEL"
done
for M in $SCORED; do
  for track in $TRACKS; do
    fit_one "$track" "$M"
  done
done

echo "=== decide $(date)"
# Every track's directory is offered at once, so the decision between two solvers of one body
# is made on the cameras held out of both fits rather than by hand afterwards. Passing
# --calibrate here as well would additionally require every body to reproduce the public
# model's margin, which stands every convex answer in the submission when that one run falls
# short; a convex answer is not a safe default but a body known to be missing the concavities
# the challenge is about.
DIRS=""
for track in $TRACKS; do DIRS="$DIRS $(dir_of "$track")"; done
$PY scripts/select_answers.py --refined $DIRS --models $SCORED

echo "=== check the submission $(date)"
$PY scripts/check_submission.py results/submission
[ -n "$failed" ] && echo "=== bodies that failed and left the convex answer standing:$failed"
echo "=== done $(date)"

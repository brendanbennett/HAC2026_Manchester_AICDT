#!/usr/bin/env bash
# A small run of the pipeline, to check the wiring before a full run on a remote machine.
#
#   scripts/run_smoke_test.sh
#
# The first run creates and activates a .venv through scripts/_venv_setup.sh and installs the
# dependencies if they are missing; later runs skip straight to the work.
#
# Runs a few bodies at a coarse grid and a few optimiser steps per stage. It needs nvdiffrast
# on a GPU, like the pipeline itself, because every stage after the fit renders with the
# exact forward model; on a machine without one, `pytest tests` is the wiring test, since the
# tests run the same code on the pure-torch rasteriser. It does not need dataset/raw: without
# it the instrument is the uncalibrated default and reconstruction is skipped.
#
# Everything is written under runs/smoke/ and dataset/generated/shapes_smoke/, never to
# runs/corpus_codes.npz, runs/corpus.npz, runs/lpd_flow.pt, models/ or the real shape
# library, so this is safe to run alongside a real scripts/run_remote_pipeline.sh run.
#
# What this proves: the shape library builds valid, non-convex, single-component bodies;
# fit_shapes.py fits codes to them; build_corpus.py renders them with the exact operator and
# runs the convex stage on them; train_prior.py and train_lpd.py train on that corpus, in
# both runs (one expert, then branched and rolled out); decision_check.py reconstructs a
# held-out body; the resulting flow checkpoint loads and runs in reconstruct_lpd.py.
#
# What this does not prove: that the results are any good. A few bodies and a few steps
# cannot learn anything; only that every stage runs and the shapes of everything agree.
set -uo pipefail
cd "$(dirname "$0")/.."

N_BODIES=${N_BODIES:-16}
LIB_RES=${LIB_RES:-32}
LIB_WORKERS=${LIB_WORKERS:-$(nproc 2>/dev/null || echo 2)}
FIT_WORKERS=${FIT_WORKERS:-$LIB_WORKERS}
FIT_POINTS=${FIT_POINTS:-1500}
FLOW_STEPS=${FLOW_STEPS:-25}
FLOW_PHASES=${FLOW_PHASES:-16}     # few phases keep the run short
FLOW_OPERATOR_RES=${FLOW_OPERATOR_RES:-16}
DESIGN_N=${DESIGN_N:-4096}
CONVEX_CKPT=${CONVEX_CKPT:-models/lpd_convex.pt}

# shellcheck disable=SC1091
source scripts/_venv_setup.sh   # creates and activates the venv, installs deps, sets PY

LIB_DIR=dataset/generated/shapes_smoke
OUT=runs/smoke
mkdir -p "$OUT" logs

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
t_start=$(date +%s)

# Runs one stage and stops the script with a message if it fails. The command's own exit
# status is read from PIPESTATUS, since the output goes through tee.
run() {
  local desc="$1" logfile="$2"; shift 2
  "$@" > >(tee "$logfile") 2>&1
  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    log "FAILED: $desc (exit $status) -- see $logfile"
    log "command was: $*"
    exit "$status"
  fi
}

log "using interpreter: $($PY --version 2>&1) at $(command -v "$PY")"

if [ -z "${HAC26_SOFTWARE_RASTER:-}" ] && ! "$PY" -c "import nvdiffrast" >/dev/null 2>&1; then
  log "nvdiffrast is not importable, so the exact forward model cannot render here."
  log "Run scripts/setup_toolchain.sh on a GPU machine, or run 'pytest tests' for a wiring"
  log "test on the CPU."
  exit 1
fi

log "=== 1/10 shape library: $N_BODIES bodies at res=$LIB_RES"
run "shape library" logs/smoke_library.log \
  "$PY" scripts/build_shape_library.py \
    --n "$N_BODIES" --out "$LIB_DIR" --workers "$LIB_WORKERS" --res "$LIB_RES" \
    --report-sample "$N_BODIES"
tail -20 logs/smoke_library.log

log "=== 2/10 spherical design (no-op if hac26/design${DESIGN_N}.npy is already checked in)"
if [ -f "hac26/design${DESIGN_N}.npy" ]; then
  log "    hac26/design${DESIGN_N}.npy already exists" | tee logs/smoke_design.log
else
  run "spherical design" logs/smoke_design.log "$PY" scripts/make_design.py --n "$DESIGN_N"
fi
tail -5 logs/smoke_design.log

log "=== 3/10 fit_shapes: per-body fit over the smoke library"
run "fit_shapes" logs/smoke_fit.log \
  "$PY" scripts/fit_shapes.py \
    --bodies "$N_BODIES" --shapes-dir "$LIB_DIR" \
    --workers "$FIT_WORKERS" --points "$FIT_POINTS" \
    --out "$OUT/corpus_codes.npz"
tail -20 logs/smoke_fit.log

# The instrument: the calibrated one when it exists and loads, otherwise the uncalibrated
# default, which is enough to check the wiring and nothing else. A calibration written by an
# older Instrument does not load, and is treated as absent.
valid_instrument() {
  "$PY" -c "import sys; sys.path.insert(0, '.')
from hac26.forward.mesh.instrument import Instrument
Instrument.load('models/instrument_calibration.pt')" >/dev/null 2>&1
}
if [ -f models/instrument_calibration.pt ] && valid_instrument; then
  CAL=models/instrument_calibration.pt
else
  CAL=$OUT/instrument_default.pt
  "$PY" -c "import sys; sys.path.insert(0, '.')
from hac26.forward.mesh.instrument import Instrument
Instrument().save('$CAL')"
  if [ -f models/instrument_calibration.pt ]; then
    log "    NOTE: models/instrument_calibration.pt does not load with this Instrument; using the"
  else
    log "    NOTE: no models/instrument_calibration.pt; using the"
  fi
  log "    UNCALIBRATED default instrument ($CAL). Fine for a wiring test; the numbers below"
  log "    are meaningless."
fi

# The convex stage: the real checkpoint when it exists, otherwise an untrained one of the
# same kind, which is enough to check the wiring and nothing else.
if [ -f "$CONVEX_CKPT" ]; then
  CONVEX=$CONVEX_CKPT
else
  CONVEX=$OUT/convex_untrained.pt
  "$PY" -c "import sys; sys.path.insert(0, '.')
import torch
from dataclasses import asdict
from hac26.train import Preset, build_model
pr = Preset(ch=8, n_iter=2, n_primal=3, n_dual=3, n_theta=12, n_phi=24, r_cond=True)
torch.save({'preset': asdict(pr), 'model': build_model(pr, 'cpu')[0].state_dict()}, '$CONVEX')"
  log "    NOTE: no $CONVEX_CKPT; using an UNTRAINED convex stage ($CONVEX). Fine for a"
  log "    wiring test; the starts it makes are meaningless."
fi

log "=== 4/10 build_corpus: curves and convex starts of the smoke bodies"
rm -rf "$OUT/corpus.npz" "$OUT/corpus.npz.parts"
run "build_corpus" logs/smoke_corpus.log \
  "$PY" scripts/build_corpus.py \
    --bodies "$N_BODIES" --phases "$FLOW_PHASES" --operator-res "$FLOW_OPERATOR_RES" \
    --codes-file "$OUT/corpus_codes.npz" --calibration "$CAL" --convex "$CONVEX" \
    --out "$OUT/corpus.npz"
tail -5 logs/smoke_corpus.log

log "=== 5/10 train_prior: the prior part over the smoke corpus"
run "train_prior" logs/smoke_prior.log \
  "$PY" scripts/train_prior.py \
    --steps 200 --batch 8 --val-bodies 2 --val-every 50 \
    --log-every 50 --corpus "$OUT/corpus.npz" --out "$OUT/prior_flow.pt"
tail -5 logs/smoke_prior.log

log "=== 6/10 train_lpd: the data part over the smoke corpus, one expert"
run "train_lpd" logs/smoke_flow.log \
  "$PY" scripts/train_lpd.py \
    --steps "$FLOW_STEPS" --batch 1 --experts 1 \
    --val-bodies 2 --val-every 10 --patience 2 \
    --ckpt-every 10 --log-every 5 --no-resume \
    --corpus "$OUT/corpus.npz" \
    --calibration "$CAL" --prior "$OUT/prior_flow.pt" \
    --out "$OUT/lpd_flow.pt"
tail -20 logs/smoke_flow.log

log "=== 7/10 train_lpd: the same run continued, branched into its experts and rolled out"
run "train_lpd (rollout)" logs/smoke_flow_rollout.log \
  "$PY" scripts/train_lpd.py \
    --steps "$FLOW_STEPS" --extra-steps 4 --batch 1 \
    --val-bodies 2 --val-every 2 --patience 2 \
    --ckpt-every 2 --log-every 1 --rollout-frac 0.5 \
    --corpus "$OUT/corpus.npz" \
    --calibration "$CAL" --prior "$OUT/prior_flow.pt" \
    --out "$OUT/lpd_flow.pt"
tail -12 logs/smoke_flow_rollout.log
grep -q "branched from 1 to" logs/smoke_flow_rollout.log || { log "FAILED: the second run did not branch"; exit 1; }

log "=== 8/10 decision_check: a held-out smoke body, two draws"
run "decision_check" logs/smoke_decision.log \
  "$PY" scripts/decision_check.py --bodies 1 --samples 2 --polish-steps 2 --res 24 \
    --val-bodies 2 --side-points 20000 \
    --ckpt "$OUT/lpd_flow.pt" --corpus "$OUT/corpus.npz" --calibration "$CAL" \
    --out "$OUT/decision_check.json"
tail -12 logs/smoke_decision.log

if [ -d dataset/raw ]; then
  log "=== 9/10 convex: dataset/raw is present, the convex start of model 1"
  mkdir -p results/smoke
  run "reconstruct (convex)" logs/smoke_convex.log \
    "$PY" scripts/reconstruct.py --ckpt "$CONVEX" --model 1 \
      --out results/smoke/convex_Asteroid01.stl
  log "=== 10/10 reconstruct: model 1 from that start"
  run "reconstruct_lpd" logs/smoke_reconstruct.log \
    "$PY" scripts/reconstruct_lpd.py --model 1 --samples 2 --res 24 --polish-steps 2 \
      --hold-out-geoms 2 \
      --phases "$FLOW_PHASES" --operator-res "$FLOW_OPERATOR_RES" \
      --ckpt "$OUT/lpd_flow.pt" --calibration "$CAL" \
      --support-from results/smoke/convex_Asteroid01.stl \
      --medoid-volume-only --out results/smoke/Asteroid01.stl
  tail -20 logs/smoke_reconstruct.log
  log "    wrote results/smoke/Asteroid01.stl"
else
  log "=== 9/10 convex, 10/10 reconstruct: skipped (dataset/raw not present)"
  log "    Both need the real measured curves, so they can't run offline. Everything up to"
  log "    here (library, fit, corpus, prior, flow, decision check) is proven wired; download"
  log "    dataset/raw to also exercise the last two stages."
fi

dt=$(( $(date +%s) - t_start ))
log "=== smoke test complete in ${dt}s"
log "    library:  $LIB_DIR/report.md"
log "    codes:    $OUT/corpus_codes.npz"
log "    corpus:   $OUT/corpus.npz"
log "    prior:    $OUT/prior_flow.pt"
log "    flow:     $OUT/lpd_flow.pt"
log ""
log "If this all ran without error, scripts/run_remote_pipeline.sh should too. Nothing"
log "here touched runs/corpus_codes.npz, runs/corpus.npz, runs/lpd_flow.pt, models/ or"
log "dataset/generated/shapes -- delete runs/smoke/ and dataset/generated/shapes_smoke/"
log "whenever you like."

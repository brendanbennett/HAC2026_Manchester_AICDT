#!/usr/bin/env bash
# A tiny, fast, fully-offline run of the pipeline, to prove the wiring works before renting
# a big remote machine and waiting for 5000 bodies.
#
#   scripts/run_smoke_test.sh
#
# First run creates and activates a .venv (via scripts/_venv_setup.sh) and installs
# dependencies if they're missing, so the FIRST run is however long `pip install torch`
# takes on your connection; every run after that skips straight to the actual work.
#
# Runs in a couple of minutes on a laptop CPU: ~16 bodies at a coarse grid, ~20 optimiser
# steps per stage. It does NOT need dataset/raw or a GPU -- reconstruction and scoring need
# the real measured curves, which the smoke test doesn't have, so those two stages are run
# only if dataset/raw happens to be present, and skipped with a note otherwise.
#
# Everything writes under runs/smoke/ and dataset/generated/shapes_smoke/, never touching
# runs/corpus_codes.npz, runs/lpd_flow.pt, or the real shape library -- so this is safe to
# run before, after, or interleaved with a real scripts/run_remote_pipeline.sh run without
# clobbering it.
#
# What this DOES prove: the shape library builds valid, non-convex, single-component bodies;
# fit_shapes.py can fit an autodecoder against them; train_lpd.py can train against that
# corpus; the flow checkpoint that comes out loads and runs in reconstruct_lpd.py.
#
# What this does NOT prove: that the results are any good. 16 bodies, 20 steps, and (unless
# runs/surrogate.pt already exists from a real run) an UNTRAINED physical surrogate are
# nowhere near enough to learn anything -- `hac26.forward.learned_surrogate.Surrogate` runs
# fine uninitialised, it just outputs noise, which is exactly what a wiring test needs and
# nothing more.
set -uo pipefail
cd "$(dirname "$0")/.."

N_BODIES=${N_BODIES:-16}
LIB_RES=${LIB_RES:-32}
LIB_WORKERS=${LIB_WORKERS:-$(nproc 2>/dev/null || echo 2)}
FIT_STEPS=${FIT_STEPS:-25}
FIT_WORKERS=${FIT_WORKERS:-$LIB_WORKERS}
FIT_POINTS=${FIT_POINTS:-1500}
FLOW_STEPS=${FLOW_STEPS:-25}
FLOW_PHASES=${FLOW_PHASES:-16}     # != the real run's 96, so the /tmp curve cache can't
                                    # collide with a production run at the same phase count
FLOW_OPERATOR_RES=${FLOW_OPERATOR_RES:-16}
DESIGN_N=${DESIGN_N:-4096}

# shellcheck disable=SC1091
source scripts/_venv_setup.sh   # creates+activates .venv, installs deps if missing, sets PY

LIB_DIR=dataset/generated/shapes_smoke
OUT=runs/smoke
mkdir -p "$OUT" logs

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
t_start=$(date +%s)

# Runs a stage; aborts the whole script with a clear message the moment one fails, instead
# of silently continuing (which is what plain `set -o pipefail` without checking each
# pipeline's actual status would do, since `tee`/`tail` sit after the real command in the
# pipe and $? alone doesn't reflect the command's exit code).
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

log "=== 1/5 shape library: $N_BODIES bodies at res=$LIB_RES"
run "shape library" logs/smoke_library.log \
  "$PY" scripts/build_shape_library.py \
    --n "$N_BODIES" --out "$LIB_DIR" --workers "$LIB_WORKERS" --res "$LIB_RES" \
    --report-sample "$N_BODIES"
tail -20 logs/smoke_library.log

log "=== 2/5 spherical design (no-op if hac26/design${DESIGN_N}.npy is already checked in)"
if [ -f "hac26/design${DESIGN_N}.npy" ]; then
  log "    hac26/design${DESIGN_N}.npy already exists" | tee logs/smoke_design.log
else
  run "spherical design" logs/smoke_design.log "$PY" scripts/make_design.py --n "$DESIGN_N"
fi
tail -5 logs/smoke_design.log

log "=== 3/5 fit_shapes: autodecoder over the smoke library"
run "fit_shapes" logs/smoke_fit.log \
  "$PY" scripts/fit_shapes.py \
    --bodies "$N_BODIES" --shapes-dir "$LIB_DIR" \
    --steps "$FIT_STEPS" --batch 2 --workers "$FIT_WORKERS" --points "$FIT_POINTS" \
    --out "$OUT/corpus_codes.npz" --decoder "$OUT/token_decoder.pt"
tail -20 logs/smoke_fit.log

if [ ! -f runs/surrogate.pt ]; then
  log "    NOTE: runs/surrogate.pt not found -- train_lpd.py will use an UNTRAINED"
  log "    physical surrogate. That's fine for a wiring test; it means the numbers below"
  log "    are meaningless, only their SHAPES (tensor dimensions) matter here."
fi

log "=== 4/5 train_lpd: flow over the smoke corpus"
rm -f "/tmp/lpd_corpus_${FLOW_PHASES}_g28_res${FLOW_OPERATOR_RES}_n${DESIGN_N}_smoke.npz"
run "train_lpd" logs/smoke_flow.log \
  "$PY" scripts/train_lpd.py \
    --bodies "$N_BODIES" --steps "$FLOW_STEPS" --phases "$FLOW_PHASES" --batch 1 \
    --val-bodies 2 --val-every 10 --patience 2 \
    --ckpt-every 10 --log-every 5 --no-resume \
    --operator-res "$FLOW_OPERATOR_RES" \
    --codes-file "$OUT/corpus_codes.npz" --decoder-file "$OUT/token_decoder.pt" \
    --cache-tag smoke \
    --out "$OUT/lpd_flow.pt"
tail -20 logs/smoke_flow.log

if [ -d dataset/raw ] && [ -f runs/surrogate.pt ]; then
  log "=== 5/5 reconstruct: dataset/raw is present, reconstructing model 1 for real"
  mkdir -p results/smoke
  run "reconstruct_lpd" logs/smoke_reconstruct.log \
    "$PY" scripts/reconstruct_lpd.py --model 1 --samples 1 --res 24 \
      --ckpt "$OUT/lpd_flow.pt" --decoder-file "$OUT/token_decoder.pt" \
      --medoid-volume-only --out results/smoke/Asteroid01.stl
  tail -20 logs/smoke_reconstruct.log
  log "    wrote results/smoke/Asteroid01.stl"
elif [ -d dataset/raw ]; then
  log "=== 5/5 reconstruct: skipped (runs/surrogate.pt not present)"
  log "    train_lpd.py can use an untrained surrogate for wiring, but reconstruct_lpd.py"
  log "    intentionally loads a checkpoint. Copy or train runs/surrogate.pt to exercise"
  log "    this last stage."
else
  log "=== 5/5 reconstruct: skipped (dataset/raw not present)"
  log "    reconstruct_lpd.py needs the real measured curves to reconstruct against, so it"
  log "    can't run offline. Everything up to here (library, fit, flow) is proven wired;"
  log "    download dataset/raw to also exercise this last stage."
fi

dt=$(( $(date +%s) - t_start ))
log "=== smoke test complete in ${dt}s"
log "    library:  $LIB_DIR/report.md"
log "    codes:    $OUT/corpus_codes.npz"
log "    flow:     $OUT/lpd_flow.pt"
log ""
log "If this all ran without error, scripts/run_remote_pipeline.sh should too. Nothing"
log "here touched runs/corpus_codes.npz, runs/lpd_flow.pt, or dataset/generated/shapes --"
log "delete runs/smoke/ and dataset/generated/shapes_smoke/ whenever you like."

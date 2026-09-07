#!/usr/bin/env bash
# Build the shape library, then run the whole pipeline against it end to end, on a fresh
# or resumed remote machine.
#
#   git clone <repo> hac26 && cd hac26
#   scripts/run_remote_pipeline.sh
#
# or from a laptop, to a machine that already has the repo:
#
#   rsync -az --exclude runs --exclude dataset/generated . remote:hac26/
#   ssh remote 'cd hac26 && nohup scripts/run_remote_pipeline.sh > pipeline.out 2>&1 &'
#
# scripts/_venv_setup.sh creates a .venv if there is none, activates it, and installs the
# dependencies when they are not already importable, so a repeat run does not download torch
# again.
#
# Every stage writes a marker file under runs/.done/ when it finishes, recording the settings
# it ran with, and is skipped on the next invocation if the marker matches. A pre-empted or
# disconnected run can therefore be relaunched as is, and one stage can be redone with
# --force-stage.
#
#   ./scripts/run_remote_pipeline.sh                      # run everything, skip done stages
#   ./scripts/run_remote_pipeline.sh --force-stage fit     # redo `fit` and everything after
#   N_BODIES=2000 ./scripts/run_remote_pipeline.sh         # override any variable below
#
# Stages, in order, and what each needs:
#
#   0. models       scripts/fetch_shape_models.py    -- downloads public asteroid shape
#                   models into SHAPE_MODELS_DIR; skipped with FETCH_MODELS=0
#   0b. objects     scripts/fetch_objects.py         -- everyday printable objects from
#                   Thingi10K into SHAPE_MODELS_DIR/objects; only with FETCH_OBJECTS=1
#   1. library      scripts/build_shape_library.py  -- CPU only; draws on the shape models
#                   and objects when SHAPE_MODELS_DIR has any
#   2. design       scripts/make_design.py           -- GPU if available, else CPU
#   3. calibrate    scripts/calibrate.py             -- needs dataset/raw and nvdiffrast;
#                   skipped if models/instrument_calibration.pt is already present
#   4. fit          scripts/fit_shapes.py            -- fits codes to the library
#   4b. corpus      scripts/build_corpus.py          -- renders every body and runs the convex
#                   stage on it; needs nvdiffrast and the convex checkpoint CONVEX_CKPT
#   4c. prior       scripts/train_prior.py           -- the prior part of the flow; no operator
#   5. flow         scripts/train_lpd.py             -- the data part with one expert, on the
#                   straight line between noise and body; needs nvdiffrast on a GPU
#   5b. flow-rollout the same run continued: branched into its experts and trained on the
#                   sampler's own states; the main training phase (see train_lpd.py)
#   5c. decision    scripts/decision_check.py         -- reconstructs held-out corpus bodies
#                   and scores every rule for picking the answer against their truth
#   6. convex       scripts/reconstruct.py, all ten models -- the starts the flow corrects;
#                   needs dataset/raw
#   6b. reconstruct scripts/reconstruct_lpd.py, all ten models -- needs nvdiffrast
#   7. score        hac26/scoring/voxel.py and side_view.py on the public models --
#                   needs dataset/raw
#
# The exact forward model renders with nvdiffrast, which scripts/setup_toolchain.sh builds;
# the pipeline stops before the calibration if it cannot be imported.
#
# All stdout and stderr also go to logs/<stage>.log.
set -uo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------- configuration
N_BODIES=${N_BODIES:-1000}
LIB_SEED=${LIB_SEED:-0}
LIB_WORKERS=${LIB_WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}
LIB_RES=${LIB_RES:-64}
LIB_DIR=${LIB_DIR:-dataset/generated/shapes}
SHAPE_MODELS_DIR=${SHAPE_MODELS_DIR:-dataset/shape_models}
FETCH_MODELS=${FETCH_MODELS:-1}
FETCH_OBJECTS=${FETCH_OBJECTS:-0}   # needs `pip install thingi10k` and a few GB of download
N_OBJECTS=${N_OBJECTS:-600}

DESIGN_N=${DESIGN_N:-4096}
DESIGN_DEVICE=${DESIGN_DEVICE:-}

FIT_STEPS=${FIT_STEPS:-4000}
FIT_BATCH=${FIT_BATCH:-4}
FIT_WORKERS=${FIT_WORKERS:-$LIB_WORKERS}
FIT_POINTS=${FIT_POINTS:-6000}
CODES_FILE=runs/corpus_codes.npz

CONVEX_CKPT=${CONVEX_CKPT:-models/lpd_convex.pt}   # the convex stage, whose starts the flow corrects
CORPUS_FILE=runs/corpus.npz

PRIOR_STEPS=${PRIOR_STEPS:-20000}   # a cap: the prior stops early once the held-out loss plateaus
PRIOR_BATCH=${PRIOR_BATCH:-64}

FLOW_STEPS=${FLOW_STEPS:-1000}   # a cap: training stops early once the held-out loss plateaus
FLOW_PHASES=${FLOW_PHASES:-96}
FLOW_BATCH=${FLOW_BATCH:-2}
FLOW_VAL_BODIES=${FLOW_VAL_BODIES:-16}  # held out of training: early stopping scores them and
                                        # the decision check reconstructs them; 0 turns both off
FLOW_VAL_EVERY=${FLOW_VAL_EVERY:-200}
FLOW_PATIENCE=${FLOW_PATIENCE:-5}
FLOW_CKPT_EVERY=${FLOW_CKPT_EVERY:-100}   # steps between resumable checkpoints; 0 disables
FLOW_CKPT=${FLOW_CKPT:-runs/lpd_flow.pt.ckpt}   # under runs/, not /tmp: it has to outlive
                                                # the job that wrote it
FLOW_LOG_EVERY=${FLOW_LOG_EVERY:-10}
FLOW_OPERATOR_RES=${FLOW_OPERATOR_RES:-32}
FLOW_TRAIN_GEOMS=${FLOW_TRAIN_GEOMS:-28}   # geometries the operator renders per step; all of them
FLOW_ROLLOUT_STEPS=${FLOW_ROLLOUT_STEPS:-1000}  # cap on the second run's extra steps: branched into
                                                # experts, rolled out, the main phase; 0 skips it
FLOW_ROLLOUT_FRAC=${FLOW_ROLLOUT_FRAC:-0.5}     # share of its draws that come from the sampler

RECON_SAMPLES=${RECON_SAMPLES:-8}
RECON_POLISH_STEPS=${RECON_POLISH_STEPS:-30}   # most gradient steps of the polish per draw; 0 skips it
RECON_RES=${RECON_RES:-96}
RECON_SNAP=${RECON_SNAP:-0}
MEDOID_VOLUME_ONLY=${MEDOID_VOLUME_ONLY:-0}
MEDOID_SIDE_POINTS=${MEDOID_SIDE_POINTS:-200000}
MEDOID_SIDE_DIRS=${MEDOID_SIDE_DIRS:-36}
MEDOID_SIDE_RES=${MEDOID_SIDE_RES:-512}
MEDOID_SIDE_MODE=${MEDOID_SIDE_MODE:-side}

# shellcheck disable=SC1091
source scripts/_venv_setup.sh   # creates and activates the venv, installs deps, sets PY

DATA_DIR=${DATA_DIR:-dataset/raw}
FORCE_STAGE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --force-stage) FORCE_STAGE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p runs runs/.done logs results/lpd

# ---------------------------------------------------------------- helpers
STAGES_AFTER_FORCE=0

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

stage_signature() {
  case "$1" in
    models)
      printf 'stage=models\nSHAPE_MODELS_DIR=%s\n' "$SHAPE_MODELS_DIR"
      ;;
    objects)
      printf 'stage=objects\nSHAPE_MODELS_DIR=%s\nN_OBJECTS=%s\n' "$SHAPE_MODELS_DIR" "$N_OBJECTS"
      ;;
    library)
      printf 'stage=library\nN_BODIES=%s\nLIB_SEED=%s\nLIB_RES=%s\nLIB_DIR=%s\nSHAPE_MODELS=%s\n' \
        "$N_BODIES" "$LIB_SEED" "$LIB_RES" "$LIB_DIR" "$(shape_model_list)"
      ;;
    design)
      printf 'stage=design\nDESIGN_N=%s\nDESIGN_DEVICE=%s\n' \
        "$DESIGN_N" "$DESIGN_DEVICE"
      ;;
    calibrate)
      printf 'stage=calibrate\nDATA_DIR=%s\n' "$DATA_DIR"
      ;;
    fit)
      printf 'stage=fit\nN_BODIES=%s\nLIB_DIR=%s\nLIB_SEED=%s\nLIB_RES=%s\nDESIGN_N=%s\nFIT_STEPS=%s\nFIT_BATCH=%s\nFIT_POINTS=%s\nCODES_FILE=%s\n' \
        "$N_BODIES" "$LIB_DIR" "$LIB_SEED" "$LIB_RES" "$DESIGN_N" "$FIT_STEPS" \
        "$FIT_BATCH" "$FIT_POINTS" "$CODES_FILE"
      ;;
    # Each later stage's signature extends the one before it, so a change anywhere upstream
    # reruns everything downstream.
    corpus)
      stage_signature fit | sed 's/^stage=fit$/stage=corpus/'
      printf 'FLOW_PHASES=%s\nFLOW_OPERATOR_RES=%s\nCONVEX_CKPT=%s\nCORPUS_FILE=%s\n' \
        "$FLOW_PHASES" "$FLOW_OPERATOR_RES" "$CONVEX_CKPT" "$CORPUS_FILE"
      ;;
    prior)
      stage_signature corpus | sed 's/^stage=corpus$/stage=prior/'
      printf 'PRIOR_STEPS=%s\nPRIOR_BATCH=%s\nFLOW_VAL_BODIES=%s\n' \
        "$PRIOR_STEPS" "$PRIOR_BATCH" "$FLOW_VAL_BODIES"
      ;;
    flow)
      stage_signature prior | sed 's/^stage=prior$/stage=flow/'
      printf 'FLOW_STEPS=%s\nFLOW_BATCH=%s\nFLOW_VAL_EVERY=%s\nFLOW_PATIENCE=%s\nFLOW_CKPT_EVERY=%s\nFLOW_CKPT=%s\nFLOW_LOG_EVERY=%s\nFLOW_TRAIN_GEOMS=%s\n' \
        "$FLOW_STEPS" "$FLOW_BATCH" "$FLOW_VAL_EVERY" "$FLOW_PATIENCE" \
        "$FLOW_CKPT_EVERY" "$FLOW_CKPT" "$FLOW_LOG_EVERY" "$FLOW_TRAIN_GEOMS"
      ;;
    flow-rollout)
      stage_signature flow | sed 's/^stage=flow$/stage=flow-rollout/'
      printf 'FLOW_ROLLOUT_STEPS=%s\nFLOW_ROLLOUT_FRAC=%s\n' \
        "$FLOW_ROLLOUT_STEPS" "$FLOW_ROLLOUT_FRAC"
      ;;
    decision)
      stage_signature flow-rollout | sed 's/^stage=flow-rollout$/stage=decision/'
      printf 'RECON_SAMPLES=%s\nRECON_POLISH_STEPS=%s\nRECON_RES=%s\n' \
        "$RECON_SAMPLES" "$RECON_POLISH_STEPS" "$RECON_RES"
      ;;
    convex)
      printf 'stage=convex\nDATA_DIR=%s\nCONVEX_CKPT=%s\n' "$DATA_DIR" "$CONVEX_CKPT"
      ;;
    reconstruct)
      printf 'stage=reconstruct\nDESIGN_N=%s\nFLOW_STEPS=%s\nFLOW_PHASES=%s\nFLOW_BATCH=%s\nFLOW_OPERATOR_RES=%s\nFLOW_TRAIN_GEOMS=%s\nFLOW_ROLLOUT_STEPS=%s\nFLOW_ROLLOUT_FRAC=%s\nCONVEX_CKPT=%s\nRECON_SAMPLES=%s\nRECON_POLISH_STEPS=%s\nRECON_RES=%s\nRECON_SNAP=%s\nMEDOID_VOLUME_ONLY=%s\nMEDOID_SIDE_POINTS=%s\nMEDOID_SIDE_DIRS=%s\nMEDOID_SIDE_RES=%s\nMEDOID_SIDE_MODE=%s\n' \
        "$DESIGN_N" "$FLOW_STEPS" "$FLOW_PHASES" "$FLOW_BATCH" "$FLOW_OPERATOR_RES" \
        "$FLOW_TRAIN_GEOMS" "$FLOW_ROLLOUT_STEPS" "$FLOW_ROLLOUT_FRAC" "$CONVEX_CKPT" \
        "$RECON_SAMPLES" "$RECON_POLISH_STEPS" "$RECON_RES" "$RECON_SNAP" \
        "$MEDOID_VOLUME_ONLY" "$MEDOID_SIDE_POINTS" "$MEDOID_SIDE_DIRS" \
        "$MEDOID_SIDE_RES" "$MEDOID_SIDE_MODE"
      ;;
    score)
      printf 'stage=score\nDATA_DIR=%s\nRECON_DIR=results/lpd\nRECON_SAMPLES=%s\nRECON_RES=%s\nRECON_SNAP=%s\nMEDOID_VOLUME_ONLY=%s\nMEDOID_SIDE_POINTS=%s\nMEDOID_SIDE_DIRS=%s\nMEDOID_SIDE_RES=%s\nMEDOID_SIDE_MODE=%s\n' \
        "$DATA_DIR" "$RECON_SAMPLES" "$RECON_RES" "$RECON_SNAP" \
        "$MEDOID_VOLUME_ONLY" "$MEDOID_SIDE_POINTS" "$MEDOID_SIDE_DIRS" \
        "$MEDOID_SIDE_RES" "$MEDOID_SIDE_MODE"
      ;;
    *)
      printf 'stage=%s\n' "$1"
      ;;
  esac
}

should_run() {
  # A stage runs if it was named by --force-stage, if a stage before it was, or if it has no
  # marker recording exactly this run's settings.
  local stage="$1"
  if [ "$stage" = "$FORCE_STAGE" ]; then STAGES_AFTER_FORCE=1; fi
  if [ "$STAGES_AFTER_FORCE" = "1" ]; then return 0; fi
  local marker="runs/.done/$stage"
  if [ ! -f "$marker" ]; then return 0; fi
  local want have
  want="$(stage_signature "$stage")"
  have="$(cat "$marker")"
  if [ "$want" != "$have" ]; then
    log "=== $stage: config changed since marker was written; rerunning"
    return 0
  fi
  return 1
}

mark_done() { stage_signature "$1" > "runs/.done/$1"; }

run_stage() {
  # run_stage NAME OUTPUT_FILE -- CMD...
  local name="$1" out="$2"; shift 2
  if ! should_run "$name"; then
    log "=== $name: skipped (already done -- rm runs/.done/$name or use --force-stage to redo)"
    return 0
  fi
  log "=== $name: starting"
  # Append rather than truncate: a stage relaunched after a pre-emption continues the earlier
  # attempt, whose log shows what it already did.
  echo "=== $name: starting $(date -u +%FT%TZ) ===" >> "logs/$name.log"
  if "$@" 2>&1 | tee -a "logs/$name.log"; then
    mark_done "$name"
    log "=== $name: done"
  else
    log "=== $name: FAILED -- see logs/$name.log"
    exit 1
  fi
}

valid_design() {
  $PY -c "import sys, numpy as np; n = int(sys.argv[1]); x = np.load(f'hac26/design{n}.npy'); assert x.shape == (n, 3); assert np.isfinite(x).all(); assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-6)" "$1"
}

# ---------------------------------------------------------------- 0. real shape models
# Asteroid models are downloaded here; everyday objects (scripts/fetch_objects.py, which
# needs `pip install thingi10k`) are fetched when FETCH_OBJECTS=1 and otherwise used if
# already under $SHAPE_MODELS_DIR/objects.
if [ "$FETCH_MODELS" = "1" ]; then
  run_stage models "$SHAPE_MODELS_DIR" \
    $PY scripts/fetch_shape_models.py --out "$SHAPE_MODELS_DIR"
else
  log "=== models: skipped (FETCH_MODELS=0)"
fi
if [ "$FETCH_OBJECTS" = "1" ]; then
  run_stage objects "$SHAPE_MODELS_DIR/objects" \
    $PY scripts/fetch_objects.py --out "$SHAPE_MODELS_DIR/objects" --n "$N_OBJECTS"
else
  log "=== objects: skipped (FETCH_OBJECTS=0); using $SHAPE_MODELS_DIR/objects if present"
fi

# ---------------------------------------------------------------- 1. shape library
# The library's signature names the shape-model and object files, so adding one reruns it.
shape_model_list() {
  [ -d "$SHAPE_MODELS_DIR" ] && find "$SHAPE_MODELS_DIR" -maxdepth 2 -type f 2>/dev/null \
    | grep -Ei '\.(obj|wf|stl|ply|tab|txt)$' | sort | tr '\n' ' '
}
LIB_ARGS=(--n "$N_BODIES" --seed "$LIB_SEED" --out "$LIB_DIR" --workers "$LIB_WORKERS"
  --res "$LIB_RES")
if [ -n "$(shape_model_list)" ]; then
  LIB_ARGS+=(--shape-models "$SHAPE_MODELS_DIR")
else
  log "=== library: no shape models under $SHAPE_MODELS_DIR; procedural families only"
fi
run_stage library "$LIB_DIR/manifest.json" \
  $PY scripts/build_shape_library.py "${LIB_ARGS[@]}"

# ---------------------------------------------------------------- 2. spherical design
DESIGN_ARGS=(--n "$DESIGN_N")
if [ -n "$DESIGN_DEVICE" ]; then
  DESIGN_ARGS+=(--device "$DESIGN_DEVICE")
fi
if should_run design; then
  if [ "$FORCE_STAGE" != "design" ] && [ -f "hac26/design${DESIGN_N}.npy" ] \
      && valid_design "$DESIGN_N"; then
    log "=== design: existing hac26/design${DESIGN_N}.npy is valid"
    mark_done design
  else
    run_stage design "hac26/design${DESIGN_N}.npy" \
      $PY scripts/make_design.py "${DESIGN_ARGS[@]}"
  fi
else
  log "=== design: skipped (already done -- rm runs/.done/design or use --force-stage to redo)"
fi

# ---------------------------------------------------------------- the rasteriser
# Everything from here on renders with the exact forward model. HAC26_SOFTWARE_RASTER=1
# selects the slow pure-torch stand-in, which is only for tests on a machine without a GPU.
if [ -z "${HAC26_SOFTWARE_RASTER:-}" ] && ! $PY -c "import nvdiffrast" >/dev/null 2>&1; then
  log "!!! nvdiffrast is not importable. Run scripts/setup_toolchain.sh first (it builds the"
  log "!!! CUDA toolchain nvdiffrast needs), then rerun this script."
  exit 1
fi

# ---------------------------------------------------------------- 3. instrument calibration
valid_instrument() {
  $PY -c "import sys; sys.path.insert(0, '.')
from hac26.forward.mesh.instrument import Instrument
Instrument.load('models/instrument_calibration.pt')" >/dev/null 2>&1
}
if [ "$STAGES_AFTER_FORCE" != "1" ] && [ "$FORCE_STAGE" != "calibrate" ] \
    && [ -f models/instrument_calibration.pt ] && valid_instrument; then
  log "=== calibrate: skipped (models/instrument_calibration.pt already present)"
  mark_done calibrate
elif [ -d "$DATA_DIR" ]; then
  run_stage calibrate models/instrument_calibration.pt \
    $PY scripts/calibrate.py --data-dir "$DATA_DIR"
else
  log "!!! calibrate: $DATA_DIR not present and no models/instrument_calibration.pt."
  log "!!! Training and reconstruction need the calibrated instrument -- stopping here."
  exit 1
fi

# ---------------------------------------------------------------- 4. per-body codes
run_stage fit "$CODES_FILE" \
  $PY scripts/fit_shapes.py \
    --bodies "$N_BODIES" --shapes-dir "$LIB_DIR" --seed "$LIB_SEED" \
    --steps "$FIT_STEPS" --batch "$FIT_BATCH" --workers "$FIT_WORKERS" \
    --points "$FIT_POINTS" \
    --out "$CODES_FILE"

# ---------------------------------------------------------------- 4b. the corpus
# Every body's curves from the exact operator, and the start the convex stage makes from
# them, which is what the flow learns to correct. Resumable body by body under
# $CORPUS_FILE.parts/, so a pre-empted job loses at most one body.
if [ ! -f "$CONVEX_CKPT" ]; then
  log "!!! corpus: $CONVEX_CKPT not present. The flow trains from the convex stage's starts,"
  log "!!! so its checkpoint is needed here (scripts/export_model.py writes it)."
  exit 1
fi
run_stage corpus "$CORPUS_FILE" \
  $PY scripts/build_corpus.py \
    --bodies "$N_BODIES" --phases "$FLOW_PHASES" --operator-res "$FLOW_OPERATOR_RES" \
    --codes-file "$CODES_FILE" --convex "$CONVEX_CKPT" --out "$CORPUS_FILE"

# ---------------------------------------------------------------- 4c. the prior part
run_stage prior runs/prior_flow.pt \
  $PY scripts/train_prior.py \
    --steps "$PRIOR_STEPS" --batch "$PRIOR_BATCH" \
    --val-bodies "$FLOW_VAL_BODIES" --corpus "$CORPUS_FILE" --out runs/prior_flow.pt

# ---------------------------------------------------------------- 5. flow
# One expert first; the second run branches it into the experts (train_lpd.py --experts).
run_stage flow runs/lpd_flow.pt \
  $PY scripts/train_lpd.py \
    --steps "$FLOW_STEPS" --batch "$FLOW_BATCH" --experts 1 \
    --train-geoms "$FLOW_TRAIN_GEOMS" --out runs/lpd_flow.pt \
    --val-bodies "$FLOW_VAL_BODIES" --val-every "$FLOW_VAL_EVERY" \
    --patience "$FLOW_PATIENCE" \
    --ckpt-every "$FLOW_CKPT_EVERY" --ckpt-file "$FLOW_CKPT" \
    --log-every "$FLOW_LOG_EVERY" \
    --corpus "$CORPUS_FILE"

# ---------------------------------------------------------------- 5b. flow, rolled out
# The same run continued from its checkpoint, branched into the default number of experts:
# part of the draws now take their state from the sampler itself, for up to
# FLOW_ROLLOUT_STEPS more steps. This is the main phase; the first run only prepares it.
if [ "$FLOW_ROLLOUT_STEPS" -gt 0 ]; then
  run_stage flow-rollout runs/lpd_flow.pt \
    $PY scripts/train_lpd.py \
      --steps "$FLOW_STEPS" --extra-steps "$FLOW_ROLLOUT_STEPS" \
      --batch "$FLOW_BATCH" \
      --train-geoms "$FLOW_TRAIN_GEOMS" --out runs/lpd_flow.pt \
      --val-bodies "$FLOW_VAL_BODIES" --val-every "$FLOW_VAL_EVERY" \
      --patience "$FLOW_PATIENCE" \
      --ckpt-every "$FLOW_CKPT_EVERY" --ckpt-file "$FLOW_CKPT" \
      --log-every "$FLOW_LOG_EVERY" \
      --rollout-frac "$FLOW_ROLLOUT_FRAC" \
      --corpus "$CORPUS_FILE"
else
  log "=== flow-rollout: skipped (FLOW_ROLLOUT_STEPS=0)"
fi

# ---------------------------------------------------------------- 5c. the decision rule
# Held-out corpus bodies are reconstructed as the challenge models will be, and every rule for
# picking the answer is scored against their truth (see decision_check.py). The summary is in
# logs/decision.log and runs/decision_check.json.
if [ "$FLOW_VAL_BODIES" -gt 0 ]; then
  run_stage decision runs/decision_check.json \
    $PY scripts/decision_check.py \
      --ckpt runs/lpd_flow.pt --corpus "$CORPUS_FILE" --val-bodies "$FLOW_VAL_BODIES" \
      --bodies "$FLOW_VAL_BODIES" --samples "$RECON_SAMPLES" \
      --polish-steps "$RECON_POLISH_STEPS" --res "$RECON_RES" \
      --side-points "$MEDOID_SIDE_POINTS" --out runs/decision_check.json
else
  log "=== decision: skipped (FLOW_VAL_BODIES=0)"
fi

# ---------------------------------------------------------------- 6. the convex starts
# The convex stage's reconstruction of every model, made the way the corpus starts were: the
# same checkpoint and the same decode, so the flow meets at reconstruction what it trained
# on. A model is redone when its STL is missing or older than its curves or the checkpoint,
# so re-downloaded data is picked up.
if [ ! -d "$DATA_DIR" ]; then
  log "!!! convex: $DATA_DIR not present; the measured curves are needed from here on."
  exit 1
fi
if should_run convex; then
  log "=== convex: starting (10 models)"
  mkdir -p results/convex
  ok=1
  for M in 1 2 3 4 5 6 7 8 9 10; do
    P=$(printf "%02d" "$M")
    OUT="results/convex/Asteroid$P.stl"
    if [ -s "$OUT" ] && [ ! "$CONVEX_CKPT" -nt "$OUT" ] && [ "$STAGES_AFTER_FORCE" != "1" ] \
       && [ -z "$(find "$DATA_DIR" -name "Asteroid*${P}_lightcurve_*" -newer "$OUT" 2>/dev/null)" ]; then
      log "  --- model $M: $OUT is current"
      continue
    fi
    log "  --- model $M -> $OUT"
    if ! $PY scripts/reconstruct.py --ckpt "$CONVEX_CKPT" --model "$M" --data-dir "$DATA_DIR" \
        --out "$OUT" 2>&1 | tee -a logs/convex.log; then
      log "  --- model $M FAILED"
      ok=0
    fi
  done
  [ "$ok" = "1" ] && mark_done convex || { log "=== convex: FAILED"; exit 1; }
  log "=== convex: done"
else
  log "=== convex: skipped (already done)"
fi

# ---------------------------------------------------------------- 6b. reconstruct all ten
if should_run reconstruct; then
  log "=== reconstruct: starting (10 models)"
  ok=1
  for M in 1 2 3 4 5 6 7 8 9 10; do
    P=$(printf "%02d" "$M")
    OUT="results/lpd/Asteroid$P.stl"
    # Each model is its own unit of work, so a job that dies on one model costs only that
    # model. The .json is written last, so a model counts as done only when both files exist
    # and are newer than the flow and the convex start they came from.
    if [ -s "$OUT" ] && [ -s "results/lpd/Asteroid$P.json" ] \
       && [ ! runs/lpd_flow.pt -nt "$OUT" ] && [ ! "results/convex/Asteroid$P.stl" -nt "$OUT" ] \
       && [ "$STAGES_AFTER_FORCE" != "1" ]; then
      log "  --- model $M: skipped ($OUT already written -- rm it to redo just this one)"
      continue
    fi
    log "  --- model $M -> $OUT (started $(date -u +%H:%M:%S))"
    RECON_ARGS=(--model "$M" --samples "$RECON_SAMPLES" --res "$RECON_RES"
      --phases "$FLOW_PHASES" --operator-res "$FLOW_OPERATOR_RES"
      --polish-steps "$RECON_POLISH_STEPS"
      --ckpt runs/lpd_flow.pt --data-dir "$DATA_DIR" --out "$OUT"
      --medoid-side-points "$MEDOID_SIDE_POINTS"
      --medoid-side-dirs "$MEDOID_SIDE_DIRS"
      --medoid-side-res "$MEDOID_SIDE_RES"
      --medoid-side-mode "$MEDOID_SIDE_MODE")
    if [ "$RECON_SNAP" = "1" ]; then
      RECON_ARGS+=(--snap)
    fi
    if [ "$MEDOID_VOLUME_ONLY" = "1" ]; then
      RECON_ARGS+=(--medoid-volume-only)
    fi
    if ! $PY scripts/reconstruct_lpd.py "${RECON_ARGS[@]}" \
        2>&1 | tee -a logs/reconstruct.log; then
      log "  --- model $M FAILED"
      ok=0
    fi
  done
  [ "$ok" = "1" ] && mark_done reconstruct || { log "=== reconstruct: FAILED"; exit 1; }
  log "=== reconstruct: done"
else
  log "=== reconstruct: skipped (already done)"
fi

# ---------------------------------------------------------------- 7. score the public models
# The flow's answers and the convex starts they came from, side by side: the flow has to
# beat its start on the non-convex public body without losing on the near-convex ones, since
# a carved-in dent that is not there costs as much as a missed one.
if [ -d "$DATA_DIR" ]; then
  run_stage score "" bash -c "
    echo '--- convex starts' &&
    $PY hac26/scoring/voxel.py --stl results/convex/Asteroid0{1,2,3}.stl --models 1 2 3 &&
    $PY hac26/scoring/side_view.py --models 1 2 3 --recon-dir results/convex &&
    echo '--- flow' &&
    $PY hac26/scoring/voxel.py --stl results/lpd/Asteroid0{1,2,3}.stl --models 1 2 3 &&
    $PY hac26/scoring/side_view.py --models 1 2 3 --recon-dir results/lpd
  "
else
  log "=== score: skipped ($DATA_DIR not present)"
fi

log "=== pipeline complete"
log "    library:  $LIB_DIR ($N_BODIES bodies; see $LIB_DIR/report.md)"
log "    normals:  $DESIGN_N"
log "    corpus:   $CORPUS_FILE"
log "    flow ckpt: runs/lpd_flow.pt"
log "    reconstructions: results/lpd/Asteroid*.stl"

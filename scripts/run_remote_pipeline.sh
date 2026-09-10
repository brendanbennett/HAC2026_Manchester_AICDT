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
# AND the source it ran with, and is skipped on the next invocation only if both still match.
# A pre-empted or disconnected run can therefore be relaunched as is, and editing a shape
# generator, a loss or the forward model reruns the stages downstream of it without anything
# having to be asked for: the alternative is a run that continues from an artefact the current
# code would not have produced. --force-stage remains for redoing a stage nothing has changed.
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
#   5b. flow-experts the same run continued from its best weights, branched into one expert
#                   per interval of t (see train_lpd.py)
#   5c. decision    scripts/decision_check.py         -- reconstructs held-out corpus bodies
#                   and scores every rule for picking the answer against their truth
#   6. convex       scripts/make_submission.py, all ten models -- the submission, and the
#                   starts the flow corrects; needs dataset/raw
#   6b. reconstruct scripts/reconstruct_lpd.py, all ten models -- needs nvdiffrast
#   7. score        scripts/benchmark.py on the public models with the organisers'
#                   measures -- needs dataset/raw
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

FIT_WORKERS=${FIT_WORKERS:-$LIB_WORKERS}
FIT_POINTS=${FIT_POINTS:-60000}
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
FLOW_EXPERT_STEPS=${FLOW_EXPERT_STEPS:-1000}   # cap on the second run's extra steps, branched
                                                # into one expert per interval of t; 0 skips it

RECON_SAMPLES=${RECON_SAMPLES:-8}
RECON_POLISH_STEPS=${RECON_POLISH_STEPS:-30}   # most gradient steps of the polish per draw; 0 skips it
RECON_RES=${RECON_RES:-64}      # extraction resolution of the written body, the one the
                                # shape codes were fitted at (scripts/fit_shapes.py)
RECON_SNAP=${RECON_SNAP:-0}
# Weight on the data part of the velocity when sampling (lpd_flow.LPDFlow.velocity).
# One is the model as trained. The decision stage reconstructs held-out bodies at each
# of RECON_GUIDANCE_SWEEP and names the weight that scores best; set this to it and
# rerun the reconstruct stage, which is cheap beside the training.
RECON_GUIDANCE=${RECON_GUIDANCE:-1.0}
RECON_GUIDANCE_SWEEP=${RECON_GUIDANCE_SWEEP:-1.0 1.5 2.0}
# Noise in the sampler (lpd_flow.churn_step). It corrects a velocity the network gets wrong
# and costs variance where it gets it right, so the level that scores best depends on the
# trained network. The decision stage reconstructs held-out bodies at each of
# RECON_CHURN_SWEEP and names the level that scores best; set this to it and rerun the
# reconstruct stage.
RECON_CHURN=${RECON_CHURN:-0.5}
RECON_CHURN_SWEEP=${RECON_CHURN_SWEEP:-0.0 0.25 0.5}
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

# A stage is skipped when its marker records the signature this run would produce, so a
# signature naming only the settings lets a stage be skipped after the code that produces its
# artefact has been rewritten: the run then continues from an artefact the current code would
# not have made, and nothing says so. The digests below put the source into the signature, so
# editing a shape generator, a loss or the forward model invalidates the stages downstream of
# it and no others.
src_digest() {
  $PY - "$@" <<'PYEOF'
import hashlib, pathlib, sys
h = hashlib.sha256()
for name in sorted(sys.argv[1:]):
    p = pathlib.Path(name)
    h.update(name.encode())
    h.update(p.read_bytes() if p.is_file() else b"<missing>")
print(h.hexdigest()[:16])
PYEOF
}

SRC_LIBRARY=$(src_digest hac26/shape_library.py hac26/shapes.py hac26/library_io.py \
                         hac26/library_metrics.py scripts/build_shape_library.py)
SRC_FORWARD=$(src_digest hac26/conventions.py hac26/geometry.py hac26/noise.py \
                         hac26/forward/mesh/exact.py hac26/forward/mesh/instrument.py \
                         hac26/forward/mesh/radiosity.py hac26/forward/mesh/raster.py \
                         hac26/forward/mesh/sensor.py hac26/forward/shared/coarea.py \
                         hac26/forward/shared/common.py \
                         hac26/forward/shared/software_raster.py scripts/calibrate.py)
SRC_FIT=$(src_digest hac26/field.py scripts/fit_shapes.py)
SRC_CORPUS=$(src_digest scripts/build_corpus.py hac26/solvers/operator.py \
                        hac26/solvers/lpd_convex.py)
SRC_FLOW=$(src_digest hac26/solvers/lpd_flow.py scripts/train_lpd.py scripts/train_prior.py)
SRC_OUTPUT=$(src_digest scripts/reconstruct_lpd.py scripts/decision_check.py \
                        hac26/solvers/output.py hac26/scoring/side_view.py \
                        hac26/scoring/voxel.py hac26/recon.py)
SRC_CONVEX=$(src_digest scripts/make_submission.py scripts/reconstruct.py scripts/eval_exact.py \
                        scripts/check_submission.py hac26/data_io.py hac26/recon.py \
                        hac26/solvers/lpd_convex.py hac26/train.py hac26/solvers/output.py)
SRC_SCORE=$(src_digest scripts/benchmark.py hac26/scoring/official.py)

stage_signature() {
  case "$1" in
    models)
      printf 'stage=models\nSHAPE_MODELS_DIR=%s\n' "$SHAPE_MODELS_DIR"
      ;;
    objects)
      printf 'stage=objects\nSHAPE_MODELS_DIR=%s\nN_OBJECTS=%s\n' "$SHAPE_MODELS_DIR" "$N_OBJECTS"
      ;;
    library)
      printf 'stage=library\nN_BODIES=%s\nLIB_SEED=%s\nLIB_RES=%s\nLIB_DIR=%s\nSHAPE_MODELS=%s\nSRC=%s\n' \
        "$N_BODIES" "$LIB_SEED" "$LIB_RES" "$LIB_DIR" "$(shape_model_list)" "$SRC_LIBRARY"
      ;;
    design)
      printf 'stage=design\nDESIGN_N=%s\nDESIGN_DEVICE=%s\n' \
        "$DESIGN_N" "$DESIGN_DEVICE"
      ;;
    calibrate)
      printf 'stage=calibrate\nDATA_DIR=%s\nSRC=%s\n' "$DATA_DIR" "$SRC_FORWARD"
      ;;
    fit)
      stage_signature library | sed 's/^stage=library$/stage=fit/'
      printf 'DESIGN_N=%s\nFIT_POINTS=%s\nCODES_FILE=%s\nSRC=%s\n' \
        "$DESIGN_N" "$FIT_POINTS" "$CODES_FILE" "$SRC_FIT"
      ;;
    # Each later stage's signature extends the one before it, so a change anywhere upstream
    # reruns everything downstream.
    corpus)
      stage_signature fit | sed 's/^stage=fit$/stage=corpus/'
      printf 'FLOW_PHASES=%s\nFLOW_OPERATOR_RES=%s\nCONVEX_CKPT=%s\nCORPUS_FILE=%s\nSRC=%s\nSRC_FWD=%s\n' \
        "$FLOW_PHASES" "$FLOW_OPERATOR_RES" "$CONVEX_CKPT" "$CORPUS_FILE" "$SRC_CORPUS" "$SRC_FORWARD"
      ;;
    prior)
      stage_signature corpus | sed 's/^stage=corpus$/stage=prior/'
      printf 'PRIOR_STEPS=%s\nPRIOR_BATCH=%s\nFLOW_VAL_BODIES=%s\nSRC=%s\n' \
        "$PRIOR_STEPS" "$PRIOR_BATCH" "$FLOW_VAL_BODIES" "$SRC_FLOW"
      ;;
    flow)
      stage_signature prior | sed 's/^stage=prior$/stage=flow/'
      printf 'FLOW_STEPS=%s\nFLOW_BATCH=%s\nFLOW_VAL_EVERY=%s\nFLOW_PATIENCE=%s\nFLOW_CKPT_EVERY=%s\nFLOW_CKPT=%s\nFLOW_LOG_EVERY=%s\nFLOW_TRAIN_GEOMS=%s\n' \
        "$FLOW_STEPS" "$FLOW_BATCH" "$FLOW_VAL_EVERY" "$FLOW_PATIENCE" \
        "$FLOW_CKPT_EVERY" "$FLOW_CKPT" "$FLOW_LOG_EVERY" "$FLOW_TRAIN_GEOMS"
      ;;
    flow-experts)
      stage_signature flow | sed 's/^stage=flow$/stage=flow-experts/'
      printf 'FLOW_EXPERT_STEPS=%s\n' "$FLOW_EXPERT_STEPS"
      ;;
    decision)
      stage_signature flow-experts | sed 's/^stage=flow-experts$/stage=decision/'
      printf 'RECON_SAMPLES=%s\nRECON_POLISH_STEPS=%s\nRECON_RES=%s\nSWEEP=%s\nSRC=%s\n' \
        "$RECON_SAMPLES" "$RECON_POLISH_STEPS" "$RECON_RES" \
        "$RECON_GUIDANCE_SWEEP $RECON_CHURN_SWEEP" "$SRC_OUTPUT"
      ;;
    convex)
      printf 'stage=convex\nDATA_DIR=%s\nCONVEX_CKPT=%s\nSRC=%s\n' \
        "$DATA_DIR" "$CONVEX_CKPT" "$SRC_CONVEX"
      ;;
    reconstruct)
      printf 'stage=reconstruct\nDESIGN_N=%s\nFLOW_STEPS=%s\nFLOW_PHASES=%s\nFLOW_BATCH=%s\nFLOW_OPERATOR_RES=%s\nFLOW_TRAIN_GEOMS=%s\nFLOW_EXPERT_STEPS=%s\nCONVEX_CKPT=%s\nRECON_SAMPLES=%s\nRECON_POLISH_STEPS=%s\nRECON_RES=%s\nRECON_SNAP=%s\nMEDOID_VOLUME_ONLY=%s\nMEDOID_SIDE_POINTS=%s\nMEDOID_SIDE_DIRS=%s\nMEDOID_SIDE_RES=%s\nMEDOID_SIDE_MODE=%s\n' \
        "$DESIGN_N" "$FLOW_STEPS" "$FLOW_PHASES" "$FLOW_BATCH" "$FLOW_OPERATOR_RES" \
        "$FLOW_TRAIN_GEOMS" "$FLOW_EXPERT_STEPS" "$CONVEX_CKPT" \
        "$RECON_SAMPLES" "$RECON_POLISH_STEPS" "$RECON_RES" "$RECON_SNAP" \
        "$MEDOID_VOLUME_ONLY" "$MEDOID_SIDE_POINTS" "$MEDOID_SIDE_DIRS" \
        "$MEDOID_SIDE_RES" "$MEDOID_SIDE_MODE"
      printf 'RECON_GUIDANCE=%s\nRECON_CHURN=%s\nSRC=%s\nSRC_FLOW=%s\nSRC_FWD=%s\n' \
        "$RECON_GUIDANCE" "$RECON_CHURN" "$SRC_OUTPUT" "$SRC_FLOW" "$SRC_FORWARD"
      ;;
    score)
      stage_signature reconstruct | sed 's/^stage=reconstruct$/stage=score/'
      printf 'SRC_SCORE=%s\n' "$SRC_SCORE"
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
# Asteroid models are downloaded here. Everyday objects (scripts/fetch_objects.py, which needs
# the thingi10k package and a long download) are fetched when FETCH_OBJECTS=1; they are an
# addition to the library, not a requirement, so a failure here is logged and the run goes
# on with whatever is under $SHAPE_MODELS_DIR/objects.
if [ "$FETCH_MODELS" = "1" ]; then
  run_stage models "$SHAPE_MODELS_DIR" \
    $PY scripts/fetch_shape_models.py --out "$SHAPE_MODELS_DIR"
else
  log "=== models: skipped (FETCH_MODELS=0)"
fi
if [ "$FETCH_OBJECTS" != "1" ]; then
  log "=== objects: skipped (FETCH_OBJECTS=0); using $SHAPE_MODELS_DIR/objects if present"
elif should_run objects; then
  log "=== objects: starting"
  echo "=== objects: starting $(date -u +%FT%TZ) ===" >> logs/objects.log
  if ! $PY -c "import thingi10k" >/dev/null 2>&1; then
    # in a subshell: a venv without pip must not stop the run over an optional stage
    ( _ensure_pip && python -m pip install -q thingi10k ) 2>&1 | tee -a logs/objects.log || true
  fi
  if $PY scripts/fetch_objects.py --out "$SHAPE_MODELS_DIR/objects" --n "$N_OBJECTS" \
      2>&1 | tee -a logs/objects.log; then
    mark_done objects
    log "=== objects: done"
  else
    log "=== objects: FAILED -- see logs/objects.log; continuing without new objects"
  fi
else
  log "=== objects: skipped (already done)"
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
    --workers "$FIT_WORKERS" \
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

# ---------------------------------------------------------------- 5b. flow, branched
# The same run continued from its best-scoring weights, branched into the default number of
# experts, for up to FLOW_EXPERT_STEPS more steps on the same straight-line objective.
if [ "$FLOW_EXPERT_STEPS" -gt 0 ]; then
  run_stage flow-experts runs/lpd_flow.pt \
    $PY scripts/train_lpd.py \
      --steps "$FLOW_STEPS" --extra-steps "$FLOW_EXPERT_STEPS" \
      --batch "$FLOW_BATCH" \
      --train-geoms "$FLOW_TRAIN_GEOMS" --out runs/lpd_flow.pt \
      --val-bodies "$FLOW_VAL_BODIES" --val-every "$FLOW_VAL_EVERY" \
      --patience "$FLOW_PATIENCE" \
      --ckpt-every "$FLOW_CKPT_EVERY" --ckpt-file "$FLOW_CKPT" \
      --log-every "$FLOW_LOG_EVERY" \
      --corpus "$CORPUS_FILE"
else
  log "=== flow-experts: skipped (FLOW_EXPERT_STEPS=0)"
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
      --guidance $RECON_GUIDANCE_SWEEP --churn $RECON_CHURN_SWEEP \
      --side-points "$MEDOID_SIDE_POINTS" --out runs/decision_check.json
else
  log "=== decision: skipped (FLOW_VAL_BODIES=0)"
fi

# ---------------------------------------------------------------- 6. the convex answers
# The submission: the convex stage's reconstruction of every model by the recipe fixed in
# scripts/make_submission.py, checked, and scored on the public models with the organisers'
# measures. The flow starts from these bodies.
if [ ! -d "$DATA_DIR" ]; then
  log "!!! convex: $DATA_DIR not present; the measured curves are needed from here on."
  exit 1
fi
run_stage convex results/public_scores.json \
  $PY scripts/make_submission.py --data-dir "$DATA_DIR" --ckpt "$CONVEX_CKPT"

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
    START=results/submission/Asteroid$P.stl
    [ -f "results/public/Asteroid$P.stl" ] && START=results/public/Asteroid$P.stl
    if [ -s "$OUT" ] && [ -s "results/lpd/Asteroid$P.json" ] \
       && [ ! runs/lpd_flow.pt -nt "$OUT" ] && [ ! "$START" -nt "$OUT" ] \
       && [ "$STAGES_AFTER_FORCE" != "1" ]; then
      log "  --- model $M: skipped ($OUT already written -- rm it to redo just this one)"
      continue
    fi
    log "  --- model $M -> $OUT (started $(date -u +%H:%M:%S))"
    RECON_ARGS=(--model "$M" --samples "$RECON_SAMPLES" --res "$RECON_RES"
      --phases "$FLOW_PHASES" --operator-res "$FLOW_OPERATOR_RES"
      --polish-steps "$RECON_POLISH_STEPS" --guidance "$RECON_GUIDANCE" \
      --churn "$RECON_CHURN"
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
# The flow's answers beside the convex answers they started from, under the organisers' two
# measures: the flow has to beat its start on the non-convex public body without losing on
# the near-convex ones, since a carved-in dent that is not there costs as much as a missed one.
if [ -d "$DATA_DIR" ]; then
  run_stage score "" \
    $PY scripts/benchmark.py results/public results/lpd --data-dir "$DATA_DIR"
else
  log "=== score: skipped ($DATA_DIR not present)"
fi

log "=== pipeline complete"
log "    library:  $LIB_DIR ($N_BODIES bodies; see $LIB_DIR/report.md)"
log "    normals:  $DESIGN_N"
log "    corpus:   $CORPUS_FILE"
log "    flow ckpt: runs/lpd_flow.pt"
log "    submission: results/submission/Asteroid*.stl (convex); flow answers: results/lpd/"

#!/usr/bin/env bash
# Build a 5000-body shape library with the new generator, then run the whole HAC26
# pipeline against it end to end, on a fresh (or resumed) remote machine.
#
#   git clone <repo> hac26 && cd hac26
#   scripts/run_remote_pipeline.sh
#
# or from a laptop, to a box that already has the repo:
#
#   rsync -az --exclude runs --exclude dataset/generated . remote:hac26/
#   ssh remote 'cd hac26 && nohup scripts/run_remote_pipeline.sh > pipeline.out 2>&1 &'
#
# A .venv is created (if missing) and activated automatically by scripts/_venv_setup.sh at
# the repo root, which also installs numpy/scipy/torch (from pyproject.toml) plus trimesh
# and scikit-image (used by fit_shapes.py, train_lpd.py, and the shape library, but not
# pinned in pyproject.toml) -- skipped if they're already importable, so a repeat run
# doesn't re-trigger a slow torch download.
#
# Every stage below writes a marker file under runs/.done/ when it finishes and is
# skipped on the next invocation if that marker (and its real output) is already there --
# so a pre-empted or ssh-dropped run is safe to just re-launch, and so is picking one
# stage to redo with --force-stage.
#
#   ./scripts/run_remote_pipeline.sh                      # run everything, skip done stages
#   ./scripts/run_remote_pipeline.sh --force-stage fit     # redo `fit` (and everything after)
#   N_BODIES=2000 ./scripts/run_remote_pipeline.sh         # override any variable below
#
# Stage order and what each needs:
#
#   1. library      scripts/build_shape_library.py  -- CPU only, this is the new generator
#   2. design       scripts/make_design.py           -- GPU if available, falls back to CPU
#   3. calibrate    scripts/calibrate.py              -- needs dataset/raw; SKIPPED if a
#                   calibration already exists at models/instrument_calibration.pt, since
#                   that's the checked-in one and refitting isn't the point of this run
#   4. surrogate    scripts/train_surrogate.py        -- needs dataset/raw AND nvdiffrast
#                   (GPU). This is the one stage that can't run CPU-only; see
#                   scripts/setup_toolchain.sh if nvdiffrast isn't built yet. SKIPPED with
#                   a warning if dataset/raw or nvdiffrast is missing, so the rest of the
#                   pipeline can still run against a surrogate trained earlier.
#   5. fit          scripts/fit_shapes.py --shapes-dir  -- the new library instead of
#                   train_surrogate.shapes()
#   6. flow         scripts/train_lpd.py
#   7. reconstruct  scripts/reconstruct_lpd.py, all 10 models
#   8. score        hac26/scoring/voxel.py + side_view.py on the 3 public models --
#                   needs dataset/raw
#
# All stdout/stderr also goes to logs/<stage>.log with timestamps.
set -uo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------- configuration
N_BODIES=${N_BODIES:-600}
LIB_SEED=${LIB_SEED:-0}
LIB_WORKERS=${LIB_WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}
LIB_RES=${LIB_RES:-64}
LIB_DIR=${LIB_DIR:-dataset/generated/shapes}

DESIGN_N=${DESIGN_N:-4096}
DESIGN_DEVICE=${DESIGN_DEVICE:-}

FIT_STEPS=${FIT_STEPS:-4000}
FIT_BATCH=${FIT_BATCH:-4}
FIT_WORKERS=${FIT_WORKERS:-$LIB_WORKERS}
FIT_POINTS=${FIT_POINTS:-6000}

FLOW_STEPS=${FLOW_STEPS:-1000}   # a cap: the flow stops early once the held-out loss plateaus
FLOW_PHASES=${FLOW_PHASES:-96}
FLOW_BATCH=${FLOW_BATCH:-2}
FLOW_VAL_BODIES=${FLOW_VAL_BODIES:-8}   # held out of training to score early stopping; 0 off
FLOW_VAL_EVERY=${FLOW_VAL_EVERY:-200}
FLOW_PATIENCE=${FLOW_PATIENCE:-5}
FLOW_CKPT_EVERY=${FLOW_CKPT_EVERY:-100}   # steps between resumable checkpoints; 0 disables
FLOW_CKPT=${FLOW_CKPT:-runs/lpd_flow.pt.ckpt}   # in runs/ (EOS), not /tmp: it has to
                                                # outlive the job that wrote it
FLOW_LOG_EVERY=${FLOW_LOG_EVERY:-10}
FLOW_OPERATOR_RES=${FLOW_OPERATOR_RES:-32}
FLOW_TRAIN_GEOMS=${FLOW_TRAIN_GEOMS:-8}

RECON_SAMPLES=${RECON_SAMPLES:-6}
RECON_RES=${RECON_RES:-96}
RECON_SNAP=${RECON_SNAP:-0}
MEDOID_VOLUME_ONLY=${MEDOID_VOLUME_ONLY:-0}
MEDOID_SIDE_POINTS=${MEDOID_SIDE_POINTS:-200000}
MEDOID_SIDE_DIRS=${MEDOID_SIDE_DIRS:-36}
MEDOID_SIDE_RES=${MEDOID_SIDE_RES:-512}
MEDOID_SIDE_MODE=${MEDOID_SIDE_MODE:-side}

# shellcheck disable=SC1091
source scripts/_venv_setup.sh   # creates+activates .venv, installs deps if missing, sets PY

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
    library)
      printf 'stage=library\nN_BODIES=%s\nLIB_SEED=%s\nLIB_RES=%s\nLIB_DIR=%s\n' \
        "$N_BODIES" "$LIB_SEED" "$LIB_RES" "$LIB_DIR"
      ;;
    design)
      printf 'stage=design\nDESIGN_N=%s\nDESIGN_DEVICE=%s\n' \
        "$DESIGN_N" "$DESIGN_DEVICE"
      ;;
    calibrate)
      printf 'stage=calibrate\nDATA_DIR=%s\n' "$DATA_DIR"
      ;;
    surrogate)
      printf 'stage=surrogate\nDATA_DIR=%s\nSURROGATE=default\n' "$DATA_DIR"
      ;;
    fit)
      printf 'stage=fit\nN_BODIES=%s\nLIB_DIR=%s\nLIB_SEED=%s\nLIB_RES=%s\nDESIGN_N=%s\nFIT_STEPS=%s\nFIT_BATCH=%s\nFIT_POINTS=%s\nCODES_FILE=%s\n' \
        "$N_BODIES" "$LIB_DIR" "$LIB_SEED" "$LIB_RES" "$DESIGN_N" "$FIT_STEPS" \
        "$FIT_BATCH" "$FIT_POINTS" "$CODES_FILE"
      ;;
    flow)
      printf 'stage=flow\nN_BODIES=%s\nLIB_DIR=%s\nLIB_SEED=%s\nLIB_RES=%s\nDESIGN_N=%s\nFIT_STEPS=%s\nFIT_BATCH=%s\nFIT_POINTS=%s\nFLOW_STEPS=%s\nFLOW_PHASES=%s\nFLOW_BATCH=%s\nFLOW_VAL_BODIES=%s\nFLOW_VAL_EVERY=%s\nFLOW_PATIENCE=%s\nFLOW_CKPT_EVERY=%s\nFLOW_CKPT=%s\nFLOW_LOG_EVERY=%s\nFLOW_OPERATOR_RES=%s\nFLOW_TRAIN_GEOMS=%s\nCODES_FILE=%s\n' \
        "$N_BODIES" "$LIB_DIR" "$LIB_SEED" "$LIB_RES" "$DESIGN_N" "$FIT_STEPS" \
        "$FIT_BATCH" "$FIT_POINTS" "$FLOW_STEPS" "$FLOW_PHASES" "$FLOW_BATCH" \
        "$FLOW_VAL_BODIES" "$FLOW_VAL_EVERY" "$FLOW_PATIENCE" "$FLOW_CKPT_EVERY" \
        "$FLOW_CKPT" "$FLOW_LOG_EVERY" \
        "$FLOW_OPERATOR_RES" "$FLOW_TRAIN_GEOMS" "$CODES_FILE"
      ;;
    reconstruct)
      printf 'stage=reconstruct\nDESIGN_N=%s\nFLOW_STEPS=%s\nFLOW_PHASES=%s\nFLOW_BATCH=%s\nFLOW_OPERATOR_RES=%s\nFLOW_TRAIN_GEOMS=%s\nRECON_SAMPLES=%s\nRECON_RES=%s\nRECON_SNAP=%s\nMEDOID_VOLUME_ONLY=%s\nMEDOID_SIDE_POINTS=%s\nMEDOID_SIDE_DIRS=%s\nMEDOID_SIDE_RES=%s\nMEDOID_SIDE_MODE=%s\n' \
        "$DESIGN_N" "$FLOW_STEPS" "$FLOW_PHASES" "$FLOW_BATCH" "$FLOW_OPERATOR_RES" \
        "$FLOW_TRAIN_GEOMS" "$RECON_SAMPLES" "$RECON_RES" "$RECON_SNAP" \
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
  # A stage runs if: it was named by --force-stage, a stage before it was, or it has no
  # marker with the exact configuration for this run. --force-stage intentionally reruns the
  # named stage and everything downstream.
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
  # -a, not a truncating tee: a stage that gets re-launched after a pre-emption is a
  # continuation, and the earlier attempt's log is how you tell what it already did.
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

# ---------------------------------------------------------------- 2. shape library
run_stage library "$LIB_DIR/manifest.json" \
  $PY scripts/build_shape_library.py \
    --n "$N_BODIES" --seed "$LIB_SEED" --out "$LIB_DIR" \
    --workers "$LIB_WORKERS" --res "$LIB_RES"

# ---------------------------------------------------------------- 3. spherical design
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

# ---------------------------------------------------------------- 4. instrument calibration
if [ -f models/instrument_calibration.pt ] && [ "$FORCE_STAGE" != "calibrate" ]; then
  log "=== calibrate: skipped (models/instrument_calibration.pt already checked in)"
  mark_done calibrate
elif [ -d "$DATA_DIR" ]; then
  run_stage calibrate calibration.json $PY scripts/calibrate.py --out calibration.json
else
  log "=== calibrate: skipped ($DATA_DIR not present, and no existing calibration)"
fi

# ---------------------------------------------------------------- 5. physical surrogate
if [ -f runs/surrogate.pt ] && [ "$FORCE_STAGE" != "surrogate" ]; then
  log "=== surrogate: skipped (runs/surrogate.pt already exists)"
  mark_done surrogate
elif [ ! -d "$DATA_DIR" ]; then
  log "=== surrogate: SKIPPED -- $DATA_DIR not present. Download the challenge data first"
  log "    (see README's Data section); the rest of the pipeline needs runs/surrogate.pt"
  log "    to exist, from either this stage or a previous run."
elif ! $PY -c "import nvdiffrast" >/dev/null 2>&1; then
  log "=== surrogate: SKIPPED -- nvdiffrast not importable. Run scripts/setup_toolchain.sh"
  log "    first (builds the CUDA toolchain nvdiffrast needs), then rerun with"
  log "    --force-stage surrogate."
else
  run_stage surrogate runs/surrogate.pt $PY scripts/train_surrogate.py
fi

if [ ! -f runs/surrogate.pt ]; then
  log "!!! runs/surrogate.pt is missing and the surrogate stage could not run."
  log "!!! fit_shapes.py does not need it, but train_lpd.py does -- stopping here."
  log "!!! Re-run this script once dataset/raw + nvdiffrast are available, or copy in a"
  log "!!! surrogate.pt trained elsewhere and rerun."
  exit 1
fi

# ---------------------------------------------------------------- 6. per-body codes
CODES_FILE=runs/corpus_codes.npz
run_stage fit "$CODES_FILE" \
  $PY scripts/fit_shapes.py \
    --bodies "$N_BODIES" --shapes-dir "$LIB_DIR" --seed "$LIB_SEED" \
    --steps "$FIT_STEPS" --batch "$FIT_BATCH" --workers "$FIT_WORKERS" \
    --points "$FIT_POINTS" \
    --out "$CODES_FILE"

# ------------------------------------------------- 7a. carry the flow corpus across jobs
# train_lpd.py's stage 1 applies the operator once per body -- a serial loop that is the
# expensive half of the flow stage at N_BODIES=1000 -- and caches the result under /tmp,
# keyed by phases, geometry count, operator resolution, design size and --cache-tag. A batch
# worker gets a fresh /tmp per job, so a pre-empted or retried run would rebuild the whole
# corpus before training a single step. The master copy therefore lives in runs/ (EOS,
# persistent) and is copied -- never moved -- into /tmp here, so the persistent one still
# stands if this job dies mid-run.
N_GEOM=$($PY -c 'from hac26.conventions import cameras; print(len(cameras()))' 2>/dev/null \
         || echo 28)
CACHE_NAME=lpd_corpus_${FLOW_PHASES}_g${N_GEOM}_res${FLOW_OPERATOR_RES}_n${DESIGN_N}_shared.npz
TMP_CACHE=/tmp/$CACHE_NAME
KEEP_CACHE=${FLOW_CACHE_KEEP:-runs/$CACHE_NAME}

save_corpus_cache() {
  # Runs on any exit, including a failed flow stage or a pre-emption signal: the corpus is
  # built before the first training step, so it is worth keeping even when training dies.
  # Written beside the target and renamed, so a job killed mid-copy cannot leave a truncated
  # cache for the next one to load.
  [ -f "$TMP_CACHE" ] || return 0
  if [ ! -f "$KEEP_CACHE" ] || [ "$TMP_CACHE" -nt "$KEEP_CACHE" ]; then
    # A job killed while numpy was writing the cache leaves a truncated .npz. Copying that
    # over a good persistent copy would cost the next job the whole corpus, so check the
    # archive's CRCs first -- an .npz is a zip, and this reads it once, in about a second.
    if ! $PY -c 'import sys, zipfile; sys.exit(zipfile.ZipFile(sys.argv[1]).testzip() is not None)' \
         "$TMP_CACHE" 2>/dev/null; then
      log "WARNING: $TMP_CACHE is incomplete (interrupted write?) -- not saving it"
      return 0
    fi
    if cp -f "$TMP_CACHE" "$KEEP_CACHE.part" && mv -f "$KEEP_CACHE.part" "$KEEP_CACHE"; then
      log "corpus cache saved to $KEEP_CACHE"
    else
      log "WARNING: could not save the corpus cache to $KEEP_CACHE"
      rm -f "$KEEP_CACHE.part"
    fi
  fi
}
trap save_corpus_cache EXIT INT TERM

if [ ! -f "$TMP_CACHE" ] && [ -f "$KEEP_CACHE" ]; then
  # A cache built against different codes is worse than no cache: the key ignores
  # --codes-file and --bodies, so a refitted corpus would be trained against silently stale
  # curves. Mtimes settle it -- the cache has to be newer than the codes it was built from.
  if [ "$CODES_FILE" -nt "$KEEP_CACHE" ] || [ -nt "$KEEP_CACHE" ]; then
    log "=== corpus cache: $KEEP_CACHE predates $CODES_FILE -- ignoring it, stage 1 rebuilds"
  elif cp -p "$KEEP_CACHE" "$TMP_CACHE"; then   # -p: same mtime, so the exit copy is a no-op
    log "=== corpus cache: seeded $TMP_CACHE from $KEEP_CACHE (stage 1 will be skipped)"
  else
    log "WARNING: could not copy $KEEP_CACHE to $TMP_CACHE -- stage 1 rebuilds the corpus"
  fi
fi

# ---------------------------------------------------------------- 7. flow
run_stage flow runs/lpd_flow.pt \
  $PY scripts/train_lpd.py \
    --bodies "$N_BODIES" --steps "$FLOW_STEPS" --phases "$FLOW_PHASES" \
    --batch "$FLOW_BATCH" --operator-res "$FLOW_OPERATOR_RES" \
    --train-geoms "$FLOW_TRAIN_GEOMS" --out runs/lpd_flow.pt \
    --val-bodies "$FLOW_VAL_BODIES" --val-every "$FLOW_VAL_EVERY" \
    --patience "$FLOW_PATIENCE" \
    --ckpt-every "$FLOW_CKPT_EVERY" --ckpt-file "$FLOW_CKPT" \
    --log-every "$FLOW_LOG_EVERY" \
    --codes-file "$CODES_FILE"

# ---------------------------------------------------------------- 8. reconstruct all 10
if should_run reconstruct; then
  log "=== reconstruct: starting (10 models)"
  ok=1
  for M in 1 2 3 4 5 6 7 8 9 10; do
    P=$(printf "%02d" "$M")
    OUT="results/lpd/Asteroid$P.stl"
    # Each model is its own unit of work: a job that dies on model 7 should cost model 7,
    # not the six that already finished. The .json is written last, so a model counts as
    # done only when both files are there.
    if [ -s "$OUT" ] && [ -s "results/lpd/Asteroid$P.json" ] \
       && [ "$STAGES_AFTER_FORCE" != "1" ]; then
      log "  --- model $M: skipped ($OUT already written -- rm it to redo just this one)"
      continue
    fi
    log "  --- model $M -> $OUT (started $(date -u +%H:%M:%S))"
    RECON_ARGS=(--model "$M" --samples "$RECON_SAMPLES" --res "$RECON_RES"
      --ckpt runs/lpd_flow.pt --out "$OUT"
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

# ---------------------------------------------------------------- 9. score the public 3
if [ -d "$DATA_DIR" ]; then
  run_stage score "" bash -c "
    $PY hac26/scoring/voxel.py --stl results/lpd/Asteroid0{1,2,3}.stl --models 1 2 3 &&
    $PY hac26/scoring/side_view.py --models 1 2 3 --recon-dir results/lpd
  "
else
  log "=== score: skipped ($DATA_DIR not present)"
fi

log "=== pipeline complete"
log "    library:  $LIB_DIR ($N_BODIES bodies; see $LIB_DIR/report.md)"
log "    normals:  $DESIGN_N"
log "    flow ckpt: runs/lpd_flow.pt"
log "    reconstructions: results/lpd/Asteroid*.stl"

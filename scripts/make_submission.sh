#!/usr/bin/env bash
# Rebuild the ten submitted reconstructions from the released data and the checkpoints in
# models/. This is the entry point for anyone reproducing the submission, and it does no
# training: the pipeline that produced the checkpoints is scripts/run_remote_pipeline.sh and
# takes days on a GPU, whereas this takes a few minutes per model.
#
#   ./scripts/make_submission.sh                   # the submitted reconstructions
#   MODELS="1 2 3" ./scripts/make_submission.sh    # a subset
#   METHOD=flow ./scripts/make_submission.sh       # the flow stage instead; see below
#
# Two methods live in this repository and the submitted one is the convex stage:
#
#   convex   scripts/reconstruct.py with models/lpd_convex.pt. A learned primal-dual solver
#            for the convex hull's support function. This is what is submitted.
#
#   flow     scripts/reconstruct_lpd.py with models/lpd_flow.pt, which carves the convex
#            stage's answer using a conditional flow trained on a synthetic corpus. It is
#            better than the convex stage on held-out corpus bodies and worse on the three
#            public asteroids, so it is not what is submitted; see docs/results.md.
#
# What it needs:
#
#   dataset/raw                 the organisers' released data, as `make data` fetches it
#   models/lpd_convex.pt        the convex stage, committed
#   models/lpd_flow.pt          only for METHOD=flow
#   a CUDA GPU and nvdiffrast   `make toolchain` builds it; `make check` reports on it
#
# A model already written is left alone, so an interrupted run resumes by being run again;
# pass FORCE=1 to redo everything.
set -uo pipefail
cd "$(dirname "$0")/.."

METHOD=${METHOD:-convex}
DATA_DIR=${DATA_DIR:-dataset/raw}
CONVEX_CKPT=${CONVEX_CKPT:-models/lpd_convex.pt}
FLOW_CKPT=${FLOW_CKPT:-models/lpd_flow.pt}
CONVEX_DIR=${CONVEX_DIR:-results/convex}
FLOW_DIR=${FLOW_DIR:-results/lpd}
MODELS=${MODELS:-1 2 3 4 5 6 7 8 9 10}
FORCE=${FORCE:-0}

# --------------------------------------------------- the settings the checkpoints were made with
# Not free parameters. GUIDANCE in particular was chosen by the decision check
# (scripts/decision_check.py) against held-out bodies; changing it changes the answers.
GUIDANCE=${GUIDANCE:-2.0}            # weight on the data part of the velocity when sampling
SAMPLES=${SAMPLES:-8}                # draws per model, reduced to one answer by the medoid rule
POLISH_STEPS=${POLISH_STEPS:-30}
RES=${RES:-96}                       # extraction resolution of the written mesh
PHASES=${PHASES:-96}
OPERATOR_RES=${OPERATOR_RES:-32}
MEDOID_SIDE_POINTS=${MEDOID_SIDE_POINTS:-200000}
MEDOID_SIDE_DIRS=${MEDOID_SIDE_DIRS:-36}
MEDOID_SIDE_RES=${MEDOID_SIDE_RES:-512}
MEDOID_SIDE_MODE=${MEDOID_SIDE_MODE:-side}

case "$METHOD" in
  convex) SUBMIT_DIR="$CONVEX_DIR" ;;
  flow)   SUBMIT_DIR="$FLOW_DIR" ;;
  *) echo "METHOD=$METHOD: expected 'convex' or 'flow'" >&2; exit 1 ;;
esac

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --------------------------------------------------------------------------- the venv
# _venv_setup.sh creates and activates .venv, installs the dependencies and sets PY. It is
# the same one every other entry point uses, so a checkout that can run the pipeline can run
# this without further setup.
# shellcheck disable=SC1091
source scripts/_venv_setup.sh

# --------------------------------------------------------------------------- prerequisites
# Checked up front and all at once: each of these otherwise costs a run that fails on the
# first model, and on a queued GPU that is an hour to find out.
fail=0
[ -s "$CONVEX_CKPT" ] || { echo "ERROR: $CONVEX_CKPT is missing or empty" >&2; fail=1; }
if [ "$METHOD" = "flow" ] && [ ! -s "$FLOW_CKPT" ]; then
  echo "ERROR: $FLOW_CKPT is missing or empty (needed for METHOD=flow)" >&2
  fail=1
fi
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
  echo "ERROR: $DATA_DIR is missing or empty -- run \`make data\` first" >&2
  fail=1
fi
if ! $PY -c "import torch, nvdiffrast.torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: need a CUDA GPU and a working nvdiffrast -- run \`make toolchain\` and" >&2
  echo "       \`make check\`, which say which of the two is missing" >&2
  fail=1
fi
[ "$fail" = "0" ] || exit 1

mkdir -p "$CONVEX_DIR" "$SUBMIT_DIR" logs
log "method $METHOD -> $SUBMIT_DIR, data $DATA_DIR"
log "models: $MODELS"

# --------------------------------------------------------------------------- the models
# One model at a time, each its own unit of work, so a run that dies on one costs only that
# one. The flow needs the convex answer first: its support function is the starting h the
# flow corrects, so METHOD=flow builds both.
ok=1
for M in $MODELS; do
  P=$(printf "%02d" "$M")
  CONVEX_STL="$CONVEX_DIR/Asteroid$P.stl"
  OUT_STL="$SUBMIT_DIR/Asteroid$P.stl"

  if [ "$FORCE" != "1" ] && [ -s "$OUT_STL" ] \
     && { [ "$METHOD" = "convex" ] || [ -s "$FLOW_DIR/Asteroid$P.json" ]; }; then
    log "model $M: $OUT_STL is already written (FORCE=1 to redo)"
    continue
  fi

  if [ "$FORCE" = "1" ] || [ ! -s "$CONVEX_STL" ]; then
    log "model $M: convex -> $CONVEX_STL"
    if ! $PY scripts/reconstruct.py --ckpt "$CONVEX_CKPT" --model "$M" \
        --data-dir "$DATA_DIR" --out "$CONVEX_STL" 2>&1 | tee -a logs/submission.log; then
      log "model $M: convex FAILED"
      ok=0
      continue
    fi
  fi

  if [ "$METHOD" = "flow" ]; then
    log "model $M: flow -> $OUT_STL"
    if ! $PY scripts/reconstruct_lpd.py \
        --model "$M" --ckpt "$FLOW_CKPT" --data-dir "$DATA_DIR" --out "$OUT_STL" \
        --support-from "$CONVEX_STL" \
        --samples "$SAMPLES" --guidance "$GUIDANCE" --polish-steps "$POLISH_STEPS" \
        --res "$RES" --phases "$PHASES" --operator-res "$OPERATOR_RES" \
        --medoid-side-points "$MEDOID_SIDE_POINTS" --medoid-side-dirs "$MEDOID_SIDE_DIRS" \
        --medoid-side-res "$MEDOID_SIDE_RES" --medoid-side-mode "$MEDOID_SIDE_MODE" \
        2>&1 | tee -a logs/submission.log; then
      log "model $M: flow FAILED"
      ok=0
    fi
  fi
done

# --------------------------------------------------------------------------- the check
# The pose the challenge asks for -- rotation axis on z, the model touching z = -1 and z = 1,
# inside the published bounding cylinder -- is what submission.py checks, on the files as
# written. A reconstruction that is right and posed wrong scores zero, so this runs every
# time and its exit status is the script's.
log "checking the written meshes against what the submission requires"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" $PY hac26/submission.py --dir "$SUBMIT_DIR"
check=$?

if [ "$ok" != "1" ]; then
  log "FAILED: at least one model did not reconstruct; see logs/submission.log"
  exit 1
fi
[ "$check" = "0" ] || { log "FAILED: the pose check rejected at least one mesh"; exit 1; }
log "done: $SUBMIT_DIR/Asteroid<NN>.stl for $MODELS"

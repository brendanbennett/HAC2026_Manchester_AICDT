#!/bin/bash --login
# Genetic-algorithm hyperparameter tuning with the EXACT forward model (real shadows +
# radiosity, hac26.forward.mesh.exact.ExactForward via hac26.genetic_utils.ExactForwardModel)
# on one CSF3 GPU, sized to fit a single <=10 hour job. Modelled on submit_csf3.sh (module
# loads, scratch symlinks, the Makefile's venv/toolchain/check), but a separate script: that
# one runs the full LPD pipeline; this runs scripts/calibrate.py, then
# scripts/profile_exact_forward.py, then scripts/tune_genetic_hyperparams.py --forward-model
# exact -- all CPU-cheap or GPU-bound, none of it the multi-stage training pipeline.
#
#   ./submit_csf3_ga_tuning.sh                       # gpuL (L40S 48GB), the default below
#   CSF_PARTITION=gpuA ./submit_csf3_ga_tuning.sh    # A100 80GB
#   sbatch submit_csf3_ga_tuning.sh                  # gpuL only; ignores CSF_PARTITION
#
# Free-at-point-of-use access (ri.itservices.manchester.ac.uk/csf3/batch-slurm/gpu-jobs-slurm/,
# read 2026-09-11) covers gpuA and gpuL with no approval needed, up to 4 such GPUs in use at
# once, combined, across both. gpuH/gpuH_short (H200, see submit_csf3.sh) need a specific
# allocation account free-tier access does not include -- this script deliberately only offers
# gpuA/gpuL.
#
# =============================================================================
# What this job does NOT do: run the tuned hyperparameters on the ten challenge models. That
# is scripts/run_final_ga_all_models.py --forward-model exact, a separate, much shorter job
# (ten GA runs at ONE fixed hyperparameter setting, not thousands of Optuna trials) --
# deliberately left for after this job's results/genetic/hyperparameter_tuning/best_params.json
# exists, rather than guessing its wall-clock budget now.
#
# Time budget inside the 10h walltime, tracked live (see "budget" below): environment setup
# (make venv/toolchain/check) and calibration are one-off, largely fixed costs; profiling is a
# few minutes; everything left over goes to tuning via --timeout, not a guessed --n-trials --
# exactly scripts/run_tuning_laptop.sh's own reasoning, ported from 8 CPU workers to 1 GPU.
#
# Memory: hac26/solvers/genetic.py evaluates its population with a plain Python loop
# ([self.fitness_fn(params) for params in population]) and keeps only the current
# population and a per-generation scalar/param-vector history (kilobytes over even hundreds
# of generations) -- no unbounded accumulation there. ExactForwardModel.curves() converts its
# result to .cpu().numpy() before returning, so no GPU tensor is held past one candidate's
# evaluation; torch's caching allocator will show rising "reserved" memory up to a working-set
# plateau, which is normal, not a leak. The real cost is compute time per candidate, not
# memory -- see the profiling step.
# =============================================================================
#SBATCH --job-name=hac26-ga-exact-tuning
#SBATCH --partition=gpuL
#SBATCH -G 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4          # the GA has no internal parallelism (see "Memory" above),
                                    # so more cores here would idle, not speed anything up --
                                    # 4 is enough for mesh decimation and the geodesic
                                    # control-point setup (scipy, single-threaded-ish already)
#SBATCH -t 10:00:00
#SBATCH --output=logs/csf3_ga_tuning_%j.out
#SBATCH --error=logs/csf3_ga_tuning_%j.err

# ---------------------------------------------------------------- submission
if [ -z "${SLURM_JOB_ID:-}" ]; then
  set -euo pipefail
  PARTITION=${CSF_PARTITION:-gpuL}
  case "$PARTITION" in
    gpuA|gpuL) ARGS=(-c 4 -t "${CSF_TIME:-10:00:00}") ;;
    *) echo "CSF_PARTITION=$PARTITION: this script only offers gpuA or gpuL (free-tier" >&2
       echo "  access; gpuH/gpuH_short need an allocation account -- see submit_csf3.sh)" >&2
       exit 1 ;;
  esac
  cd "$(dirname "$0")"
  mkdir -p logs
  exec sbatch -p "$PARTITION" "${ARGS[@]}" "$@" "$(basename "$0")"
fi

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
JOB_START=$(date +%s)
TOTAL_BUDGET_S=$((10 * 3600))
SAFETY_MARGIN_S=$((15 * 60))   # left unspent so results still get written if a step overruns

budget_remaining() {
  echo $(( TOTAL_BUDGET_S - ( $(date +%s) - JOB_START ) - SAFETY_MARGIN_S ))
}

if [ -n "${CSF_PARTITION:-}" ] && [ "$CSF_PARTITION" != "${SLURM_JOB_PARTITION:-}" ]; then
  echo "ERROR: CSF_PARTITION=$CSF_PARTITION, but this job is on ${SLURM_JOB_PARTITION:-?}." >&2
  echo "       Submit with ./submit_csf3_ga_tuning.sh rather than sbatch to use it." >&2
  exit 1
fi

# ---------------------------------------------------------------- modules (see submit_csf3.sh)
module purge
module load tools/env/proxy2
module load cuda/12.6.2
module load python/3.13.1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi || { echo "ERROR: no GPU visible; this job needs one." >&2; exit 1; }

# ---------------------------------------------------------------- scratch (see submit_csf3.sh)
# Only dataset/raw and runs/ -- the two things that grow large or download fresh -- go to
# scratch. dataset/generated/shapes (the shape-library test bodies tuning scores candidates
# against) and models/instrument_calibration.pt are small and expected to already be in this
# checkout (rsynced from the machine that built them), so this job does not rebuild the
# library or the "real"/objects shape families from scratch.
SCRATCH=${SCRATCH_DIR:-$HOME/scratch/hac26}
mkdir -p "$SCRATCH/cache"
export XDG_CACHE_HOME="$SCRATCH/cache"
export UV_CACHE_DIR="$SCRATCH/cache/uv"
mkdir -p dataset

# If this checkout already has real content at dataset/raw or runs/ (e.g. rsynced from a
# machine that already fetched and validated it), move it into scratch and symlink to it,
# rather than deleting it and letting `make data`/a fresh run re-fetch or rebuild for
# nothing -- and potentially diverge from what was already validated, since the organisers'
# hosted files can drift after a checked-in dataset/MANIFEST.sha256 was cut.
_scratch_link() {
  local path="$1" target="$2"
  if [ -L "$path" ]; then return; fi
  if [ -d "$path" ] && [ -n "$(ls -A "$path" 2>/dev/null)" ]; then
    echo "$path already has content -- moving it into scratch instead of re-fetching/rebuilding"
    rm -rf "$target"; mv "$path" "$target"
  else
    rm -rf "$path" "$target"; mkdir -p "$target"
  fi
  ln -s "$target" "$path"
}
_scratch_link dataset/raw "$SCRATCH/raw"
_scratch_link runs "$SCRATCH/runs"
echo "scratch: $SCRATCH"; df -h "$SCRATCH" | tail -1

# ---------------------------------------------------------------- environment
set -e
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.9}   # A100, L40S (see submit_csf3.sh)
make venv CUDA=12 2>&1 | tee logs/csf3_ga_venv.log
make toolchain      2>&1 | tee logs/csf3_ga_toolchain.log
make check          2>&1 | tee logs/csf3_ga_check.log
if ! .venv/bin/python -c "import nvdiffrast.torch" >/dev/null 2>&1; then
  echo "ERROR: nvdiffrast did not build/import; see logs/csf3_ga_toolchain.log. Aborting" >&2
  echo "       rather than silently falling back to the ~100x slower software rasteriser" >&2
  echo "       for a job whose whole point is real per-candidate cost." >&2
  exit 1
fi
set +e

# ---------------------------------------------------------------- data
# The organisers' dataset. Unlike submit_csf3.sh's tolerant use of this (its later stages
# don't all need dataset/raw), calibrate.py right below does -- fail clearly here rather than
# have it fail confusingly on missing files three steps later.
make data 2>&1 | tee logs/csf3_ga_data.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "ERROR: make data failed; see logs/csf3_ga_data.log. calibrate.py needs dataset/raw," >&2
  echo "       so there is nothing useful this job can still do. Stopping." >&2
  exit 1
fi

# ---------------------------------------------------------------- calibration
# models/instrument_calibration.pt is known stale against the current Instrument class (a
# pedestal -> raw_pedestal reparametrisation with no valid inverse map -- Instrument.load
# raises rather than silently using wrong values). Regenerate it unconditionally; a fresh fit
# on 3 bodies with early stopping is not worth trying to detect and skip.
echo "=== [$(( $(date +%s) - JOB_START ))s elapsed] calibrating ==="
.venv/bin/python scripts/calibrate.py --data-dir dataset/raw 2>&1 | tee logs/csf3_ga_calibrate.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "ERROR: calibrate.py failed; see logs/csf3_ga_calibrate.log. Tuning against the" >&2
  echo "       still-broken checkpoint would defeat the point of this job. Stopping." >&2
  exit 1
fi

# ---------------------------------------------------------------- profiling
# Real numbers before committing the rest of the budget to a tuning plan -- the whole reason
# this step exists rather than guessing --exact-max-population/--exact-max-generations.
# hac26/forward/mesh/exact.py's own mesh_constants docstring warns the CPU-bound visibility
# ray test, not the GPU render, dominates "on a carved mesh" (i.e. once the GA has actually
# deformed something concave) -- profiling only the smooth LPD start mesh could understate
# cost on the shapes tuning will spend most of its time evaluating. Both are profiled.
# --radiosity-faces/--production-m match what the tuning invocation below actually uses
# (600, 25) so this estimate is for the real job, not a different hypothetical one.
echo "=== [$(( $(date +%s) - JOB_START ))s elapsed] profiling ==="
.venv/bin/python scripts/profile_exact_forward.py --device cuda --n-geoms 28 --m 50 \
  --radiosity-faces "${EXACT_RADIOSITY_FACES:-600}" --production-m "${TUNING_M:-25}" \
  2>&1 | tee logs/csf3_ga_profile_smooth.log

.venv/bin/python - <<'PY' 2>&1 | tee logs/csf3_ga_profile_concave_setup.log
# A crudely concave test mesh: push a ring of an icosphere's vertices inward. Not a real GA
# candidate, just something for _prepare()'s visibility ray test to work harder against than
# the LPD's already-fairly-smooth output.
import numpy as np, trimesh
from hac26.shapes import icosphere
v, f = icosphere(subdiv=3)
d = np.linalg.norm(v[:, :2], axis=1)
ring = (d > 0.3) & (d < 0.6)
v = v.copy(); v[ring] *= 0.5
trimesh.Trimesh(vertices=v, faces=f, process=False).export("/tmp/concave_probe.stl")
print(f"wrote /tmp/concave_probe.stl: {len(v)} verts, {len(f)} faces, "
     f"{ring.sum()} vertices pulled inward")
PY
.venv/bin/python scripts/profile_exact_forward.py --device cuda --mesh /tmp/concave_probe.stl \
  --n-geoms 28 --m 50 --radiosity-faces "${EXACT_RADIOSITY_FACES:-600}" \
  --production-m "${TUNING_M:-25}" 2>&1 | tee logs/csf3_ga_profile_concave.log

# ---------------------------------------------------------------- tuning
REMAINING=$(budget_remaining)
if [ "$REMAINING" -lt 600 ]; then
  echo "ERROR: only ${REMAINING}s left for tuning after setup/calibration/profiling -- see" >&2
  echo "       logs/csf3_ga_*.log for what took long; nothing to tune with, stopping." >&2
  exit 1
fi
echo "=== [$(( $(date +%s) - JOB_START ))s elapsed] tuning: ${REMAINING}s budget " \
    "(timeout governs trial count, not a guess -- see trials.csv afterward) ==="

# Sizing, revised from a real measurement (scripts/profile_exact_forward.py on an actual
# CSF3 GPU): ~92.6s/candidate at production scale (28 geoms x 50 phases, radiosity_faces=200).
# Decomposing fixed (_prepare, CPU-bound, ~0.2s) from variable (render) cost showed:
#   - radiosity_faces 200 -> 600 costs almost nothing (_prepare is <1% of total call time at
#     production frame counts) -- raised to 600 (RenderConfig's own default, full fidelity)
#     rather than trading fidelity against budget, since that trade barely exists here.
#   - population_size/n_generations/n_shapes/m all multiply LINEARLY into the real cost, and
#     population=33/generations=134 (the convex optimum) would cost ~115h for ONE trial --
#     not a tuning-budget problem, a does-this-fit-at-all problem. These are cut hard, and
#     m is halved for tuning only (the final run, run_final_ga_all_models.py, stays at m=50).
# --warm-start-from: rather than search this cut-down space blind, seed it from the 59-trial
# convex-mode search already on disk -- narrows mutation_scale/mutation_decay (pure search
# mechanics, expected to transfer) tightly around the convex optimum, enqueues one trial
# there directly, and leaves deform_width/max_amp/n_cpts at full range (concavity-sculpting
# capacity a convex-blind search should not be trusted to have found the right region for).
# At these settings (n_shapes=1, population/generations fixed near their floor by the tight
# budget) expect on the order of a HANDFUL of completed trials in the time available, not the
# convex search's 59 -- see scripts/tune_genetic_hyperparams.py's own --warm-start-from
# docstring and this job's own trials.csv afterward for what actually ran.
OUT_DIR=results/genetic/hyperparameter_tuning_exact
mkdir -p "$OUT_DIR"
.venv/bin/python scripts/tune_genetic_hyperparams.py \
  --forward-model exact \
  --exact-device cuda \
  --exact-radiosity-faces "${EXACT_RADIOSITY_FACES:-600}" \
  --m "${TUNING_M:-25}" \
  --n-shapes "${N_SHAPES:-1}" \
  --exact-max-population "${EXACT_MAX_POPULATION:-12}" \
  --exact-max-generations "${EXACT_MAX_GENERATIONS:-15}" \
  --warm-start-from "${WARM_START:-results/genetic/hyperparameter_tuning/best_params.json}" \
  --storage "sqlite:///$SCRATCH/runs/genetic_tuning_exact.db" \
  --output-dir "$OUT_DIR" \
  --timeout "$REMAINING" \
  --seed 42 \
  2>&1 | tee logs/csf3_ga_tuning.log

echo "=== done at $(( $(date +%s) - JOB_START ))s; results in $OUT_DIR ==="
echo "Next step (a separate, much shorter job, NOT run here): "
echo "  scripts/run_final_ga_all_models.py --forward-model exact --exact-device cuda \\"
echo "    --best-params $OUT_DIR/best_params.json"

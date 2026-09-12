#!/bin/bash --login
# Run scripts/run_final_ga_all_models.py --forward-model exact on all ten challenge models,
# at the hyperparameters submit_csf3_ga_tuning.sh found. A separate, much shorter job from
# tuning: ten GA runs at ONE fixed hyperparameter setting, not an open-ended Optuna search,
# so --timeout does not apply here -- this either finishes or it doesn't, and how long it
# takes is exactly what submit_csf3_ga_tuning.sh's profiling numbers were for.
#
# Unlike tuning, this DOES save output: an STL + results.json (checkpoints, Dice for models
# 1-3) per model under results/genetic/final_run_exact/ -- see scripts/run_final_ga_all_models.py's
# own docstring. --workers defaults to 1 for --forward-model exact there already (one shared
# GPU, no benefit from concurrent CUDA contexts -- see the script itself), so this job asks
# for only one GPU.
#
#   ./submit_csf3_ga_final.sh
#   CSF_PARTITION=gpuA ./submit_csf3_ga_final.sh
#   ./submit_csf3_ga_final.sh --models 1 2 3          # just the public models, e.g. to check
#                                                       # Dice before spending time on 4-10
#
# Then pull the results back to your laptop (run FROM the laptop, not here):
#   rsync -avz <user>@csf3.itservices.manchester.ac.uk:~/hac26/results/genetic/final_run_exact/ \
#       /Users/user/Desktop/hac-2026/results/genetic/final_run_exact/
#SBATCH --job-name=hac26-ga-exact-final
#SBATCH --partition=gpuL
#SBATCH -G 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 10:00:00
#SBATCH --output=logs/csf3_ga_final_%j.out
#SBATCH --error=logs/csf3_ga_final_%j.err

if [ -z "${SLURM_JOB_ID:-}" ]; then
  set -euo pipefail
  PARTITION=${CSF_PARTITION:-gpuL}
  case "$PARTITION" in
    gpuA|gpuL) ;;
    *) echo "CSF_PARTITION=$PARTITION: this script only offers gpuA or gpuL" >&2; exit 1 ;;
  esac
  cd "$(dirname "$0")"
  mkdir -p logs
  exec sbatch -p "$PARTITION" -c 4 -t "${CSF_TIME:-10:00:00}" "$(basename "$0")" "$@"
fi

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

module purge
module load tools/env/proxy2
module load cuda/12.6.2
module load python/3.13.1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi || { echo "ERROR: no GPU visible." >&2; exit 1; }

SCRATCH=${SCRATCH_DIR:-$HOME/scratch/hac26}
mkdir -p "$SCRATCH/cache"
export XDG_CACHE_HOME="$SCRATCH/cache"
export UV_CACHE_DIR="$SCRATCH/cache/uv"
mkdir -p dataset

# See submit_csf3_ga_tuning.sh's own comment: preserve already-fetched content by moving it
# into scratch, rather than deleting it and forcing a wasteful/possibly-diverging re-fetch.
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

# Assumes submit_csf3_ga_tuning.sh already ran on this checkout: reuses its venv/toolchain
# (make venv/toolchain are no-ops once built -- see the Makefile) and its calibration.
set -e
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.9}
make venv CUDA=12 2>&1 | tee logs/csf3_final_venv.log
make toolchain      2>&1 | tee logs/csf3_final_toolchain.log
if ! .venv/bin/python -c "import nvdiffrast.torch" >/dev/null 2>&1; then
  echo "ERROR: nvdiffrast did not build/import." >&2
  exit 1
fi
if [ ! -f models/instrument_calibration.pt ] || \
   ! .venv/bin/python -c "
from hac26.forward.mesh.instrument import Instrument
Instrument.load('models/instrument_calibration.pt', device='cpu')" >/dev/null 2>&1; then
  echo "ERROR: models/instrument_calibration.pt missing or stale -- run" >&2
  echo "       submit_csf3_ga_tuning.sh first (it regenerates this), or run" >&2
  echo "       scripts/calibrate.py --data-dir dataset/raw directly." >&2
  exit 1
fi
set +e

BEST_PARAMS=${BEST_PARAMS:-results/genetic/hyperparameter_tuning_exact/best_params.json}
if [ ! -f "$BEST_PARAMS" ]; then
  echo "ERROR: $BEST_PARAMS not found -- run submit_csf3_ga_tuning.sh first, or set" >&2
  echo "       BEST_PARAMS=path/to/best_params.json." >&2
  exit 1
fi

.venv/bin/python scripts/run_final_ga_all_models.py \
  --best-params "$BEST_PARAMS" \
  --forward-model exact \
  --exact-device cuda \
  --output-dir results/genetic/final_run_exact \
  "$@" \
  2>&1 | tee logs/csf3_ga_final_run.log

echo "=== done; results in results/genetic/final_run_exact/ (this machine) ==="
echo "Pull them to your laptop with (run FROM the laptop):"
echo "  rsync -avz \$USER@csf3.itservices.manchester.ac.uk:~/hac26/results/genetic/final_run_exact/ \\"
echo "      /Users/user/Desktop/hac-2026/results/genetic/final_run_exact/"

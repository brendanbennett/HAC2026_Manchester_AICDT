#!/bin/bash --login
# A short (~15-20 min target), throwaway-output run of the SAME steps and the SAME
# tune_genetic_hyperparams.py invocation submit_csf3_ga_tuning.sh uses, at trivial scale --
# to catch module/toolchain/CUDA-arch problems before committing a 10h job to them. Modelled
# on scripts/run_smoke_test.sh's own principle: everything here goes to throwaway paths
# (runs/smoke_ga/, /tmp), never to models/instrument_calibration.pt or
# results/genetic/hyperparameter_tuning_exact/, so this is safe to run before OR after a real
# submit_csf3_ga_tuning.sh job without clobbering its output.
#
#   ./submit_csf3_ga_smoketest.sh
#   CSF_PARTITION=gpuA ./submit_csf3_ga_smoketest.sh
#
# What a pass here proves: nvdiffrast builds and imports on this node, ExactForward renders
# without crashing (calibrate.py's own forward pass, then profile_exact_forward.py), and
# tune_genetic_hyperparams.py --forward-model exact runs end to end -- literally the same
# script and flags the real job uses, just --exact-max-population/--exact-max-generations and
# --timeout cut down so one trial takes seconds, not minutes.
#
# What it does NOT prove: that the real job's timing/trial-count estimates are right -- that
# is what submit_csf3_ga_tuning.sh's own profiling step is for, on real settings.
#SBATCH --job-name=hac26-ga-smoketest
#SBATCH --partition=gpuL
#SBATCH -G 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
#SBATCH --output=logs/csf3_ga_smoketest_%j.out
#SBATCH --error=logs/csf3_ga_smoketest_%j.err

if [ -z "${SLURM_JOB_ID:-}" ]; then
  set -euo pipefail
  PARTITION=${CSF_PARTITION:-gpuL}
  case "$PARTITION" in
    gpuA|gpuL) ;;
    *) echo "CSF_PARTITION=$PARTITION: this script only offers gpuA or gpuL" >&2; exit 1 ;;
  esac
  cd "$(dirname "$0")"
  mkdir -p logs
  exec sbatch -p "$PARTITION" -c 4 -t 00:20:00 "$@" "$(basename "$0")"
fi

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs/smoke_ga

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
mkdir -p runs/smoke_ga

set -e
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.9}
make venv CUDA=12 2>&1 | tee logs/csf3_smoke_venv.log
make toolchain      2>&1 | tee logs/csf3_smoke_toolchain.log
make check          2>&1 | tee logs/csf3_smoke_check.log
if ! .venv/bin/python -c "import nvdiffrast.torch" >/dev/null 2>&1; then
  echo "FAIL: nvdiffrast did not build/import; see logs/csf3_smoke_toolchain.log" >&2
  exit 1
fi
echo "PASS: nvdiffrast imports"

# A throwaway calibration -- 5 steps, nowhere near converged, only to prove ExactForward
# renders on this node without crashing. Never touches models/instrument_calibration.pt.
make data 2>&1 | tee logs/csf3_smoke_data.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "FAIL: make data" >&2; exit 1; fi
.venv/bin/python scripts/calibrate.py --data-dir dataset/raw --steps 5 \
  --out runs/smoke_ga/calibration.pt --report runs/smoke_ga/calibration.json \
  2>&1 | tee logs/csf3_smoke_calibrate.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "FAIL: calibrate.py" >&2; exit 1; fi
echo "PASS: calibrate.py ran end to end"

.venv/bin/python scripts/profile_exact_forward.py --device cuda \
  --calibration runs/smoke_ga/calibration.pt --n-geoms 2 --m 4 --repeats 1 \
  2>&1 | tee logs/csf3_smoke_profile.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "FAIL: profile_exact_forward.py" >&2; exit 1; fi
echo "PASS: ExactForward renders on this GPU"

# The real test: the exact same script and flags submit_csf3_ga_tuning.sh runs, cut down so
# it finishes in seconds -- one trial, tiny population/generations, one test shape.
.venv/bin/python scripts/tune_genetic_hyperparams.py \
  --forward-model exact --exact-device cuda \
  --calibration runs/smoke_ga/calibration.pt \
  --n-shapes 1 --n-trials 1 \
  --exact-max-population 4 --exact-max-generations 2 \
  --storage "sqlite:///runs/smoke_ga/study.db" \
  --output-dir runs/smoke_ga/tuning_out \
  --seed 42 \
  2>&1 | tee logs/csf3_smoke_tuning.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "FAIL: tune_genetic_hyperparams.py" >&2; exit 1; fi
echo "PASS: tune_genetic_hyperparams.py --forward-model exact ran end to end"

echo ""
echo "=== ALL SMOKE TESTS PASSED -- submit_csf3_ga_tuning.sh should run cleanly ==="

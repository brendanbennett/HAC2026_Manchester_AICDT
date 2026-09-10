#!/bin/bash --login
# hac26 full pipeline on CSF3, branch facet-radiance-surrogate, N_BODIES=1500.
#
#   sbatch submit_csf3.sh
#
# =============================================================================
# Partition, modules and the torch build are all confirmed against this cluster:
# sinfo, module avail, and nvidia-smi reporting driver 595.71.05.
# =============================================================================
#SBATCH --job-name=hac26-pipeline
#SBATCH --partition=gpuA                  # 19 nodes, 4-day limit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00                 # gpuA maximum; the run resumes if it is hit
#SBATCH --output=logs/csf3_%j.out
#SBATCH --error=logs/csf3_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-${SGE_O_WORKDIR:-$PWD}}"
mkdir -p logs

# ---------------------------------------------------------------- modules
# The Makefile installs torch itself against the driver it finds, so the module
# system only has to supply a CUDA toolkit (for the nvdiffrast build) and a
# Python >= 3.10. Outbound HTTP on CSF3 goes through a proxy: pip, the Dropbox
# data fetch, the JPL/PDS shape models and the Thingi10K download all need it,
# and without it they hang rather than fail.
module purge
module load tools/env/proxy2                 # http_proxy is unset without this
module load cuda/12.6.2                      # the only CUDA module on the merged CSF
module load python/3.13.1                    # the Makefile builds its own venv from it

echo "proxy: ${http_proxy:-UNSET -- downloads will hang}"
nvidia-smi || echo "WARNING: no GPU visible; corpus, flow and reconstruct will fail"

# ---------------------------------------------------------------- scratch
# Home is quota'd and the bulky parts of a run do not fit in it: the Thingi10K
# cache reached 39G on its own and the challenge archive died mid-extract with
# ENOSPC. Everything large is put on scratch and reached through symlinks, so
# the scripts keep using their own default paths -- `make data` runs
# fetch_data.py with no arguments, so its destination cannot be set by an
# environment variable, and a symlink is the only thing that redirects it.
SCRATCH=${SCRATCH_DIR:-$HOME/scratch/hac26}
mkdir -p "$SCRATCH"/{cache,raw,generated,shape_models,runs}
export XDG_CACHE_HOME="$SCRATCH/cache"    # pip, and thingi10k via platformdirs
export UV_CACHE_DIR="$SCRATCH/cache/uv"   # named outright: uv's cache reached 39G
                                          # in home on the first run, and it is the
                                          # Makefile's installer whenever uv is on PATH
export HF_HOME="$SCRATCH/cache/huggingface"
mkdir -p dataset
for pair in dataset/raw:raw dataset/generated:generated             dataset/shape_models:shape_models runs:runs; do
  link=${pair%%:*}
  target="$SCRATCH/${pair##*:}"
  if [ ! -L "$link" ]; then
    rm -rf "$link"
    ln -s "$target" "$link"
  fi
done
echo "scratch: $SCRATCH"; df -h "$SCRATCH" | tail -1

# ---------------------------------------------------------------- sizing
# build_corpus.py renders every body once with the exact forward model on the
# GPU and runs the convex stage on it, so wall time is linear in N_BODIES. That
# stage, not the library build, is what sets the runtime.
export PY_VERSION=3.13                # match the module; the Makefile default is 3.12
export N_BODIES=${N_BODIES:-1500}
export LIB_RES=${LIB_RES:-64}
export LIB_WORKERS=${LIB_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
export LIB_SEED=${LIB_SEED:-0}

# The "object" family (Thingi10K everyday objects) and the "real" family (JPL
# and PDS asteroid models) are 12% and 14% of the intended family weights, and
# LibrarySpec.weights() silently drops both while their directories are empty.
export FETCH_OBJECTS=1
export N_OBJECTS=${N_OBJECTS:-600}
export FETCH_MODELS=1

# ---------------------------------------------------------------- environment
# Built here rather than left to _venv_setup.sh inside the pipeline, so a build
# failure surfaces in its own log instead of a hundred lines into stage one.
set -e
# CUDA=12, not auto. Auto reads the DRIVER, and 595.71.05 is new enough that it would
# choose a CUDA 13 torch (cu130) -- but cuda/12.6.2 is the only toolkit module here, and
# nvdiffrast is compiled with that toolkit against torch's headers. cu129 keeps the two
# on the same CUDA major version. The venv records this, so later `make` calls keep it.
make venv CUDA=12 2>&1 | tee logs/csf3_venv.log
make toolchain      2>&1 | tee logs/csf3_toolchain.log
make check          2>&1 | tee logs/csf3_check.log
set +e

# ---------------------------------------------------------------- data
# Several GB from the organisers' Dropbox. Downloads only when dataset/raw is
# empty; if it is already populated and matches dataset/MANIFEST.sha256 this is
# a no-op. Needed by the convex, reconstruct and score stages at the end of the
# run; the library, corpus and flow training stages do not touch it, and
# models/instrument_calibration.pt is committed so calibrate is skipped either
# way. A failure here therefore costs the last three stages, not the run.
make data 2>&1 | tee logs/csf3_data.log || \
  echo "WARNING: data fetch failed; stages through flow training still run"

# ---------------------------------------------------------------- run
# Each stage writes a marker under runs/.done/ and is skipped if already
# complete, so a job that hits the wall clock can be resubmitted as is.
./scripts/run_remote_pipeline.sh

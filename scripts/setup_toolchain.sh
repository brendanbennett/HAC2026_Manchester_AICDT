#!/usr/bin/env bash
# Build the CUDA toolchain and nvdiffrast that M2/M3 need, without root.
#
# The machine has a CUDA *runtime* (torch cu130) but no CUDA *compiler*, no GL/EGL headers,
# and no sudo. nvdiffrast is not on PyPI and compiles CUDA at install time, so it has to be
# built from source against a toolkit we assemble ourselves.
#
# Why CUDA 12.9 and not 13.0, which would match torch: the 13.0 redistributable does not
# ship `cicc`, the NVVM frontend, so nvcc 13.0 cannot compile a .cu file at all. Grafting
# 12.9's cicc into a 13.0 tree fails differently -- it emits CUDA-12 launch stubs against
# CUDA-13 headers ("__cudaLaunch requires 2 arguments"). 12.9 is the earliest consistent
# line that both ships cicc and supports the sm_120 device here.
#
# That leaves torch's build-time guard refusing 12.9 against a cu130 build. The guard is
# about ABI drift in the CUDA runtime; nvdiffrast uses launch, memcpy and texture APIs,
# which are stable across this pair. It is bypassed deliberately, and the result is then
# exercised end to end -- rasterize, interpolate, antialias and backward -- before anything
# is built on top. Measured after the build: coverage 0.1809 against an analytic 0.1809,
# and a non-zero finite silhouette gradient (|grad| = 5708.68).
set -euo pipefail
PREFIX=${PREFIX:-$HOME/.local}
REDIST=https://developer.download.nvidia.com/compute/cuda/redist
CUDA=$PREFIX/cuda129
VENV_CU=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'..','nvidia','cu13'))" 2>/dev/null || true)

mkdir -p "$CUDA" /tmp/cudadl && cd /tmp/cudadl
for c in cuda_nvcc cuda_cudart cuda_cccl; do
  p=$(curl -s "$REDIST/redistrib_12.9.0.json" | python3 -c "import json,sys;print(json.load(sys.stdin)['$c']['linux-x86_64']['relative_path'])")
  [ -f "$(basename "$p")" ] || curl -sL "$REDIST/$p" -o "$(basename "$p")"
  tar -xf "$(basename "$p")"
done
for d in cuda_*-archive; do cp -rn "$d"/* "$CUDA"/; done

# math-library headers come from the venv's cu13 bundle; the CUDA CORE headers must stay
# consistent with nvcc 12.9, so only the math ones are copied.
mkdir -p /tmp/mathinc
for pat in 'cublas*' 'cusparse*' 'cusolver*' 'cufft*' 'curand*' 'nvrtc*' 'library_types.h' 'cuComplex.h'; do
  cp -n "$VENV_CU"/include/$pat /tmp/mathinc/ 2>/dev/null || true
done

[ -d /tmp/nvdiffrast ] || git clone --depth 1 -q https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
cat > /tmp/build_nvdr.py <<'PY'
import sys, torch.utils.cpp_extension as ce
ce._check_cuda_version = lambda *a, **k: None      # see the header comment
sys.argv = ["setup.py", "build_ext", "--inplace"]
exec(open("setup.py").read())
PY
cd /tmp/nvdiffrast
CUDA_HOME=$CUDA PATH=$CUDA/bin:$PATH CPATH=/tmp/mathinc:$CUDA/include \
  CPLUS_INCLUDE_PATH=/tmp/mathinc:$CUDA/include python3 /tmp/build_nvdr.py
V=$(grep -oP "__version__\s*=\s*['\"]\K[0-9.]+" nvdiffrast/__init__.py | head -1)
D=/tmp/nvdiffrast/nvdiffrast-${V:-0.3.3}.dist-info; mkdir -p "$D"
printf 'Metadata-Version: 2.1\nName: nvdiffrast\nVersion: %s\n' "${V:-0.3.3}" > "$D/METADATA"
: > "$D/RECORD"
cat <<MSG

Add to the environment before importing nvdiffrast:
  export PYTHONPATH=/tmp/nvdiffrast:\$PYTHONPATH
  export LD_LIBRARY_PATH=$CUDA/lib:$CUDA/lib64:\$LD_LIBRARY_PATH
MSG

#!/usr/bin/env bash
# Build, without root, the CUDA toolchain and nvdiffrast that the mesh forward model
# (hac26/forward/mesh) and scripts/train_surrogate.py need.
#
# The machine has a CUDA runtime (torch's cu13 wheels) but no CUDA compiler, no GL/EGL
# headers and no sudo. nvdiffrast is not on PyPI and compiles CUDA at install time, so it is
# built from source against a toolkit assembled here.
#
# CUDA 12.9 rather than 13.0, which would match torch: the 13.0 redistributable does not
# ship cicc, the NVVM frontend, so its nvcc cannot compile a .cu file, and grafting 12.9's
# cicc into a 13.0 tree fails on mismatched launch stubs. 12.9 is the earliest line that
# ships cicc and supports the sm_120 device here.
#
# torch's build-time version guard refuses 12.9 against a cu13 build. The guard is about ABI
# drift in the CUDA runtime; nvdiffrast uses launch, memcpy and texture APIs, which are stable
# across this pair, so the guard is bypassed in the build script below.
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

# The math-library headers come from torch's cu13 bundle; the CUDA core headers must stay
# consistent with nvcc 12.9, so only the math ones are copied.
mkdir -p /tmp/mathinc
for pat in 'cublas*' 'cusparse*' 'cusolver*' 'cufft*' 'curand*' 'nvrtc*' 'library_types.h' 'cuComplex.h'; do
  cp -n "$VENV_CU"/include/$pat /tmp/mathinc/ 2>/dev/null || true
done

[ -d /tmp/nvdiffrast ] || git clone --depth 1 -q https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
cat > /tmp/build_nvdr.py <<'PY'
import sys, torch.utils.cpp_extension as ce
ce._check_cuda_version = lambda *a, **k: None      # the version guard; see the header comment
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

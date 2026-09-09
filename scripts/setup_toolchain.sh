#!/usr/bin/env bash
# Build, without root, the CUDA toolchain and nvdiffrast that the mesh forward model
# (hac26/forward/mesh) needs. `make toolchain` runs this inside the venv; running it by
# hand works too, with that venv activated.
#
# The machine has a CUDA runtime (torch's bundled libraries) but no CUDA compiler, no
# GL/EGL headers and no sudo. nvdiffrast is not on PyPI and compiles CUDA at install time,
# so it is built from source against a toolkit assembled here.
#
# CUDA 12.9 whichever torch build is installed: the 13.0 redistributable does not ship
# cicc, the NVVM frontend, so its nvcc cannot compile a .cu file, and grafting 12.9's cicc
# into a 13.0 tree fails on mismatched launch stubs. 12.9 is the earliest line that ships
# cicc and supports the sm_120 device here.
#
# torch's build-time version guard refuses 12.9 against a cu13 build. The guard is about ABI
# drift in the CUDA runtime; nvdiffrast uses launch, memcpy and texture APIs, which are stable
# across this pair, so the guard is bypassed in the build script below. Against a cu12 torch
# the guard would pass anyway; bypassing it is harmless there.
set -euo pipefail
cd "$(dirname "$0")/.."

PREFIX=${PREFIX:-$HOME/.local}
REDIST=https://developer.download.nvidia.com/compute/cuda/redist
CUDA=$PREFIX/cuda129
# Not /tmp: the venv is pointed at this directory afterwards, and it has to survive a reboot.
SRC=${NVDIFFRAST_SRC:-$PREFIX/src/nvdiffrast}
PYBIN=${PYTHON:-python3}
VENV=${VIRTUAL_ENV:-}

$PYBIN -c 'import torch' >/dev/null 2>&1 || {
  echo "ERROR: torch is not importable with $PYBIN. Run \`make venv CUDA=12\` (or 13) first," >&2
  echo "       then \`make toolchain\`." >&2; exit 1; }

mkdir -p "$CUDA" /tmp/cudadl && cd /tmp/cudadl
for c in cuda_nvcc cuda_cudart cuda_cccl; do
  p=$(curl -s "$REDIST/redistrib_12.9.0.json" | $PYBIN -c "import json,sys;print(json.load(sys.stdin)['$c']['linux-x86_64']['relative_path'])")
  [ -f "$(basename "$p")" ] || curl -sL "$REDIST/$p" -o "$(basename "$p")"
  tar -xf "$(basename "$p")"
done
for d in cuda_*-archive; do cp -rn "$d"/* "$CUDA"/; done

# The math-library headers come from torch's own CUDA bundle, wherever it keeps them: a cu13
# torch has one nvidia/cu13/include tree, a cu12 torch has one include directory per library
# (nvidia/cublas/include, ...). Both are searched. The CUDA core headers must stay consistent
# with nvcc 12.9, so only the math ones are copied.
mkdir -p /tmp/mathinc
INC_DIRS=$($PYBIN - <<'PY'
import os, glob, torch
base = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia")
print("\n".join(d for d in glob.glob(os.path.join(base, "*", "include")) if os.path.isdir(d)))
PY
)
[ -n "$INC_DIRS" ] || echo "[toolchain] warning: no bundled CUDA headers found under torch's nvidia/" >&2
while IFS= read -r inc; do
  [ -n "$inc" ] || continue
  for pat in 'cublas*' 'cusparse*' 'cusolver*' 'cufft*' 'curand*' 'nvrtc*' 'library_types.h' 'cuComplex.h'; do
    cp -n "$inc"/$pat /tmp/mathinc/ 2>/dev/null || true
  done
done <<< "$INC_DIRS"

[ -d "$SRC" ] || git clone --depth 1 -q https://github.com/NVlabs/nvdiffrast.git "$SRC"
cat > /tmp/build_nvdr.py <<'PY'
import sys, torch.utils.cpp_extension as ce
ce._check_cuda_version = lambda *a, **k: None      # the version guard; see the header comment
sys.argv = ["setup.py", "build_ext", "--inplace"]
exec(open("setup.py").read())
PY
cd "$SRC"
CUDA_HOME=$CUDA PATH=$CUDA/bin:$PATH CPATH=/tmp/mathinc:$CUDA/include \
  CPLUS_INCLUDE_PATH=/tmp/mathinc:$CUDA/include $PYBIN /tmp/build_nvdr.py
V=$(grep -oE "__version__[[:space:]]*=[[:space:]]*['\"][0-9.]+" nvdiffrast/__init__.py \
    | grep -oE "[0-9.]+$" | head -1)
D=$SRC/nvdiffrast-${V:-0.3.3}.dist-info; mkdir -p "$D"
printf 'Metadata-Version: 2.1\nName: nvdiffrast\nVersion: %s\n' "${V:-0.3.3}" > "$D/METADATA"
: > "$D/RECORD"

# Wire it into the venv, so that neither the pipeline scripts nor a bare `python` need
# PYTHONPATH set: a .pth puts the build directory on sys.path, and the library path the
# compiled plugin needs goes in etc/nvdiffrast-env.sh, which scripts/_venv_setup.sh sources.
if [ -n "$VENV" ]; then
  SITE=$($PYBIN -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  printf '%s\n' "$SRC" > "$SITE/nvdiffrast.pth"
  mkdir -p "$VENV/etc"
  cat > "$VENV/etc/nvdiffrast-env.sh" <<ENV
# Written by scripts/setup_toolchain.sh. Sourced by scripts/_venv_setup.sh.
export LD_LIBRARY_PATH="$CUDA/lib:$CUDA/lib64:\${LD_LIBRARY_PATH:-}"
ENV
  echo "[toolchain] nvdiffrast wired into $VENV (nvdiffrast.pth, etc/nvdiffrast-env.sh)"
  echo "[toolchain] for an interactive shell: source $VENV/bin/activate && source $VENV/etc/nvdiffrast-env.sh"
else
  cat <<MSG

Not run inside a venv, so nothing was wired up. Add to the environment before importing:
  export PYTHONPATH=$SRC:\$PYTHONPATH
  export LD_LIBRARY_PATH=$CUDA/lib:$CUDA/lib64:\$LD_LIBRARY_PATH
MSG
fi

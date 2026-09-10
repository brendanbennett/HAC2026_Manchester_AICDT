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
#
# What is *not* free across that pair is the math-library headers: see the comment on
# $MATHINC below. They have to come from the 12.x line nvcc belongs to, whatever torch is.
set -euo pipefail
cd "$(dirname "$0")/.."

PREFIX=${PREFIX:-$HOME/.local}
REDIST=https://developer.download.nvidia.com/compute/cuda/redist
CUDA_VER=12.9.0                 # see the header comment; do not move this to 13.x
CUDA_MAJOR=${CUDA_VER%%.*}
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
  p=$(curl -s "$REDIST/redistrib_$CUDA_VER.json" | $PYBIN -c "import json,sys;print(json.load(sys.stdin)['$c']['linux-x86_64']['relative_path'])")
  [ -f "$(basename "$p")" ] || curl -sL "$REDIST/$p" -o "$(basename "$p")"
  tar -xf "$(basename "$p")"
done
for d in cuda_*-archive; do cp -rn "$d"/* "$CUDA"/; done

# The math-library headers come from torch's own CUDA bundle, wherever it keeps them: a cu13
# torch has one nvidia/cu13/include tree, a cu12 torch has one include directory per library
# (nvidia/cublas/include, ...). The CUDA core headers stay nvcc 12.9's, and the math headers
# have to belong to that same line: CUDA 13's cublas_api.h and cusolverDn.h declare functions
# taking cudaEmulation* types that the 12.9 runtime headers never define, so a 13.x tree does
# not compile against this toolkit however torch itself was built -- ATen/cuda/CUDAContext.h
# pulls cublas_v2.h into every translation unit. The trees are therefore filtered by CUDA
# major and only the matching line is copied. What the extension links against is untouched by
# this: nvdiffrast calls no cublas, cusolver or cusparse entry point, it only has to compile
# past their declarations.
#
# The directory is rebuilt on every run. It used to be filled with cp -n and never cleared, so
# the first torch a machine ever had decided its contents for good.
MATHINC=/tmp/mathinc
rm -rf "$MATHINC"; mkdir -p "$MATHINC"
INC_DIRS=$($PYBIN - "$CUDA_MAJOR" <<'PY'
import glob, os, re, sys, torch

want = int(sys.argv[1])
base = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia")

def cuda_major(inc):
    """Which CUDA line an include tree belongs to, or None when it does not say."""
    m = re.fullmatch(r"cu(\d+)", os.path.basename(os.path.dirname(inc)))
    if m:
        return int(m.group(1))                     # the cu13 layout names its line
    for header, macro in (("cublas_api.h", "CUBLAS_VER_MAJOR"),
                          ("cusparse.h", "CUSPARSE_VER_MAJOR")):
        try:
            text = open(os.path.join(inc, header)).read()
        except OSError:
            continue
        m = re.search(r"#define\s+%s\s+(\d+)" % macro, text)
        if m:
            return int(m.group(1))                 # the per-library layout does not
    return None    # curand, nvrtc and friends say nothing, and are not what breaks

print("\n".join(d for d in sorted(glob.glob(os.path.join(base, "*", "include")))
                if os.path.isdir(d) and cuda_major(d) in (None, want)))
PY
)
USED=
while IFS= read -r inc; do
  [ -n "$inc" ] || continue
  before=$(find "$MATHINC" -type f | wc -l)
  for pat in 'cublas*' 'cusparse*' 'cusolver*' 'cufft*' 'curand*' 'nvrtc*' 'library_types.h' 'cuComplex.h'; do
    cp -n "$inc"/$pat "$MATHINC"/ 2>/dev/null || true
  done
  [ "$(find "$MATHINC" -type f | wc -l)" = "$before" ] || USED="$USED$inc"$'\n'
done <<< "$INC_DIRS"

# A missing tree is silent above, so what was actually collected is checked here.
if ! grep -qsE "^#define CUBLAS_VER_MAJOR $CUDA_MAJOR[[:space:]]*$" "$MATHINC/cublas_api.h"; then
  echo "ERROR: no CUDA $CUDA_MAJOR math-library headers to build against." >&2
  echo "       torch here is $($PYBIN -c 'import torch; print(torch.__version__)'), and a cu13" >&2
  echo "       torch bundles CUDA 13 headers only, which nvcc $CUDA_VER cannot parse (see the" >&2
  echo "       comment above this check). They cannot be fetched on their own either: the" >&2
  echo "       redistributable that carries them is ~2 GB. Install a CUDA 12 torch instead:" >&2
  echo "         make venv CUDA=12 && make toolchain" >&2
  echo "       That venv keeps its cu129 torch afterwards; no other target swaps it back." >&2
  exit 1
fi
echo "[toolchain] CUDA $CUDA_MAJOR math headers from:" >&2
printf '%s' "$USED" | sed 's/^/  /' >&2

[ -d "$SRC" ] || git clone --depth 1 -q https://github.com/NVlabs/nvdiffrast.git "$SRC"

# setup.py rebuilds no object whose file is newer than its source, so a tree left from a build
# against different headers would end up half relinked. When the header set changes -- a torch
# swapped for another CUDA line, or a build that failed the way this check exists for -- the
# objects go.
STAMP=$SRC/.hac26-mathinc
if [ "$(cat "$STAMP" 2>/dev/null)" != "$INC_DIRS" ]; then
  rm -rf "$SRC/build"
  find "$SRC" -name '_nvdiffrast_c*.so' -delete
  printf '%s\n' "$INC_DIRS" > "$STAMP"
fi
cat > /tmp/build_nvdr.py <<'PY'
import sys, torch.utils.cpp_extension as ce
ce._check_cuda_version = lambda *a, **k: None      # the version guard; see the header comment
sys.argv = ["setup.py", "build_ext", "--inplace"]
exec(open("setup.py").read())
PY
cd "$SRC"
CUDA_HOME=$CUDA PATH=$CUDA/bin:$PATH CPATH=$MATHINC:$CUDA/include \
  CPLUS_INCLUDE_PATH=$MATHINC:$CUDA/include $PYBIN /tmp/build_nvdr.py
# The .dist-info below is not bookkeeping: since 0.4.0 nvdiffrast/__init__.py reads its own
# version through importlib.metadata, so `import nvdiffrast` raises PackageNotFoundError
# without it. That release also moved the number into pyproject.toml; older checkouts keep a
# literal in __init__.py, so both are read, and neither may fail the script under `set -e`
# (grep exits 1 on no match, and pipefail passes that on).
V=$(grep -m1 -oE '^version[[:space:]]*=[[:space:]]*"[0-9][0-9.]*"' pyproject.toml 2>/dev/null \
    | grep -oE '[0-9][0-9.]*' || true)
[ -n "$V" ] || V=$(grep -oE "__version__[[:space:]]*=[[:space:]]*['\"][0-9.]+" \
                     nvdiffrast/__init__.py 2>/dev/null | grep -oE "[0-9.]+$" | head -1 || true)
rm -rf "$SRC"/nvdiffrast-*.dist-info      # a stale one is a second distribution to metadata
D=$SRC/nvdiffrast-${V:-0.4.0}.dist-info; mkdir -p "$D"
printf 'Metadata-Version: 2.1\nName: nvdiffrast\nVersion: %s\n' "${V:-0.4.0}" > "$D/METADATA"
: > "$D/RECORD"
echo "[toolchain] built nvdiffrast ${V:-0.4.0} in $SRC"

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

# Sourced by run_smoke_test.sh and run_remote_pipeline.sh; not meant to be run directly.
# Creates a venv if there is none, activates it, and installs the dependencies if they are
# not already importable. Callers cd to the repo root before sourcing this.
#
# Location: REPO_ROOT/.venv is tried first. If creating it fails, as it can under WSL when
# the repo lives on a Windows-mounted drive (/mnt/c/...), whose filesystem may refuse to
# create symlinks or set the executable bit, the venv is created at ~/.venvs/<reponame> on
# WSL's own filesystem instead. That choice is recorded in .venv-external-path, a one-line
# file at the repo root, so later runs go straight there; delete the file to try the local
# .venv again.
#
# VENV_DIR, if set, overrides all of the above and is tried with no fallback.
#
# Safe to source repeatedly: activation and the dependency check are cheap once the venv
# exists and the packages are installed.

REPO_ROOT="$(pwd)"
VENV_MARKER="$REPO_ROOT/.venv-external-path"

if [ -n "${VENV_DIR:-}" ]; then
  _venv_target="$VENV_DIR"
elif [ -f "$VENV_MARKER" ]; then
  _venv_target="$(cat "$VENV_MARKER")"
else
  _venv_target="$REPO_ROOT/.venv"
fi

# Creates or repairs a venv at $1. Returns non-zero, with $1 removed, if it does not end up
# with a working bin/activate.
_venv_create() {
  local dir="$1"
  if [ -d "$dir" ] && [ ! -f "$dir/bin/activate" ]; then
    rm -rf "$dir"
  fi
  if [ -f "$dir/bin/activate" ]; then
    return 0
  fi
  local boot_py=""
  if command -v python3 >/dev/null 2>&1; then boot_py=python3
  elif command -v python >/dev/null 2>&1; then boot_py=python
  else
    echo "ERROR: neither python3 nor python found on PATH to create a venv." >&2
    return 1
  fi
  echo "[venv] creating $dir with $boot_py" >&2
  mkdir -p "$(dirname "$dir")"
  # --copies: real files instead of symlinks, which a Windows-mounted WSL path may refuse. It
  # does not help if the mount also refuses chmod on copied files; the fallback handles that.
  "$boot_py" -m venv --copies "$dir" 2>&1 | sed 's/^/[venv]   /' >&2
  if [ -f "$dir/bin/activate" ]; then
    return 0
  fi
  rm -rf "$dir"      # do not leave a half-created directory behind
  return 1
}

# The venv package is version-specific on Debian and Ubuntu (python3.12-venv, ...), so the
# hint names the one for the interpreter found.
_venv_pkg_hint() {
  local v
  v=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' \
      2>/dev/null) || v=""
  if [ -n "$v" ]; then
    echo "  sudo apt update && sudo apt install python${v}-venv"
  else
    echo "  sudo apt update && sudo apt install python3-venv"
  fi
}

if [ -z "${VIRTUAL_ENV:-}" ] || [ "$VIRTUAL_ENV" != "$_venv_target" ]; then
  if ! _venv_create "$_venv_target"; then
    if [ -n "${VENV_DIR:-}" ] || [ -f "$VENV_MARKER" ]; then
      # an explicit override or a recorded fallback failed; nothing left to try
      echo "ERROR: could not create a working venv at $_venv_target." >&2
      echo "If python3-venv isn't installed:" >&2
      _venv_pkg_hint >&2
      exit 1
    fi
    _fallback="$HOME/.venvs/$(basename "$REPO_ROOT")"
    echo "[venv] could not create a working venv at $_venv_target." >&2
    echo "[venv] This is the usual WSL failure when the repo lives on a Windows-mounted" >&2
    echo "[venv] drive (/mnt/c/...): that filesystem can refuse to create symlinks or set" >&2
    echo "[venv] the executable bit at all, which nothing in this script can work around" >&2
    echo "[venv] in place. Falling back to a venv on WSL's own filesystem instead:" >&2
    echo "[venv]   $_fallback" >&2
    if _venv_create "$_fallback"; then
      _venv_target="$_fallback"
      echo "$_fallback" > "$VENV_MARKER"
      echo "[venv] using $_fallback from now on (recorded in .venv-external-path;" >&2
      echo "[venv] delete that file to make this try $REPO_ROOT/.venv again)" >&2
    else
      echo "ERROR: could not create a working venv at $_venv_target OR at $_fallback." >&2
      echo "If python3-venv isn't installed:" >&2
      _venv_pkg_hint >&2
      exit 1
    fi
  fi
  # shellcheck disable=SC1091
  source "$_venv_target/bin/activate"
fi
echo "[venv] active: $(command -v python) ($(python --version 2>&1))"
PY=python

_ensure_pip() {
  if python -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  echo "[venv] pip is missing; bootstrapping it with ensurepip"
  if python -m ensurepip --upgrade >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: this Python venv has no pip and ensurepip could not install it." >&2
  echo "Try recreating the venv, or install pip for this Python distribution." >&2
  exit 1
}

# Dependency check. The packages are installed by name rather than with `pip install -e .`:
# an editable install writes hac26.egg-info/ inside the repo, which fails on a Windows-mounted
# WSL path, and it is not needed, since every script in scripts/ puts the repo root on
# sys.path itself.
#
# Versions match pyproject.toml. rtree backs trimesh's nearest.signed_distance (used by
# fit_shapes.py), fast_simplification backs simplify_quadric_decimation (the radiosity
# patches and the released meshes in hac26/forward/mesh/exact.py) and embreex backs the ray
# tests of the form factors; trimesh imports all three lazily, so their absence only shows
# when that code runs. truststore is needed by scripts/fetch_shape_models.py (stage 0) and
# spiceypy by scripts/fetch_dsk_shapes.py, both to reach hosts whose certificate chains the
# stdlib ssl module's static CA bundle does not cover. cloudpickle, matplotlib and psutil are
# the genetic-algorithm branch's own (hac26/genetic_utils.py, scripts/reconstruct_genetic.py).
# Importability is checked instead of running pip every time, which keeps a repeat run fast.
NEED_INSTALL=0
python -c "import numpy, scipy, torch, trimesh, skimage, rtree, fast_simplification, embreex, \
  truststore, spiceypy, cloudpickle, matplotlib, psutil" >/dev/null 2>&1 || NEED_INSTALL=1
if [ "$NEED_INSTALL" = "1" ] || [ "${FORCE_DEPS:-0}" = "1" ]; then
  echo "[venv] installing dependencies (this can take a while, especially torch)"
  _ensure_pip
  python -m pip install --upgrade pip -q
  python -m pip install "numpy>=1.24" "scipy>=1.10" "torch>=2.1" \
    trimesh scikit-image rtree fast_simplification embreex truststore spiceypy \
    "cloudpickle>=3.1.2" "matplotlib>=3.10.9" "psutil>=7.2.2" -q
  python -c "import numpy, scipy, torch, trimesh, skimage, rtree, fast_simplification, embreex, \
    truststore, spiceypy, cloudpickle, matplotlib, psutil" || {
    echo "ERROR: dependency install ran but imports still fail; see the pip output above." >&2
    exit 1
  }
  echo "[venv] dependencies installed"
else
  echo "[venv] dependencies already satisfied (numpy, scipy, torch, trimesh, skimage," \
       "rtree, fast_simplification, embreex, truststore, spiceypy, cloudpickle, matplotlib," \
       "psutil)"
fi

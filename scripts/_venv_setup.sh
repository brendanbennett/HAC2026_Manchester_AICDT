# Sourced by run_smoke_test.sh and run_remote_pipeline.sh -- not meant to be run directly.
# Creates a venv if one doesn't exist, activates it, and installs dependencies if they
# aren't already importable. Callers `cd` to the repo root before sourcing this.
#
# Location logic: try REPO_ROOT/.venv first (the normal case, and the only thing that
# happens on a native Linux filesystem). If that fails in the specific way WSL fails when a
# repo lives on a Windows-mounted drive (/mnt/c/...) -- DrvFs can refuse to create symlinks
# OR set the executable bit on copied files, so `python -m venv`, even with --copies, can
# come back with "Operation not permitted" -- fall back automatically to a venv at
# ~/.venvs/<reponame>, which sits on WSL's own native filesystem and therefore always
# supports normal file permissions. That choice is remembered in .venv-external-path (a
# one-line file at the repo root) so later runs go straight there instead of re-attempting
# and re-failing the local .venv every time. Delete that file to make it try locally again
# (e.g. after moving the repo itself onto a native filesystem).
#
# VENV_DIR overrides all of the above if set, and is tried with no fallback.
#
# Safe to source repeatedly: activation and the dependency check are both cheap once a venv
# exists and packages are installed, so this doesn't slow down a second run.

REPO_ROOT="$(pwd)"
VENV_MARKER="$REPO_ROOT/.venv-external-path"

if [ -n "${VENV_DIR:-}" ]; then
  _venv_target="$VENV_DIR"
elif [ -f "$VENV_MARKER" ]; then
  _venv_target="$(cat "$VENV_MARKER")"
else
  _venv_target="$REPO_ROOT/.venv"
fi

# Creates (or repairs) a venv at $1. Returns non-zero, with $1 removed, if it doesn't end
# up with a working bin/activate -- the one thing every failure mode has in common.
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
  # --copies: real files instead of symlinks, since symlinks are the first thing a
  # Windows-mounted WSL path refuses. Doesn't help if the mount also refuses chmod on
  # copied files (that's the fallback's job), but it's strictly better than the default
  # everywhere else, so it stays on unconditionally.
  "$boot_py" -m venv --copies "$dir" 2>&1 | sed 's/^/[venv]   /' >&2
  if [ -f "$dir/bin/activate" ]; then
    return 0
  fi
  rm -rf "$dir"      # don't leave a half-created directory behind after a failed attempt
  return 1
}

# The venv package is version-specific on Debian/Ubuntu (python3.12-venv, python3.14-venv,
# ...), and the generic `python3-venv` name doesn't always resolve -- Python's own
# ensurepip error already reports the exact one needed, this just surfaces it up front
# instead of making the person read it out of nested pip/venv output.
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
      # an explicit override or an already-recorded fallback failed too -- nothing left to
      # try automatically
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

# Dependency check. Deliberately NOT `pip install -e ".[torch]"`: an editable install writes
# hac26.egg-info/ INSIDE the repo to build its metadata, which fails with the exact same
# "Operation not permitted" as venv creation if the repo itself is on a Windows-mounted WSL
# path -- moving the venv doesn't help here, because this failure is about the SOURCE tree,
# not the venv. It also turns out to be unnecessary: every script in this repo
# (build_shape_library.py, fit_shapes.py, train_lpd.py, ...) already does its own
# `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` to find the repo root at
# runtime, so hac26 never needs to be an installed package at all -- only its dependencies
# do. Installing named packages (not `.` or `-e .`) never touches the current directory:
# pip builds them in its own temp/cache dirs regardless of where REPO_ROOT lives.
#
# Versions match pyproject.toml's runtime dependencies and the "torch" extra. rtree backs
# trimesh's spatial index (nearest.signed_distance, used to build SDF training samples) and
# fast_simplification backs simplify_quadric_decimation (used by hac26/calibrate.py's mesh
# decimation, called from train_lpd.py's curve rendering) -- trimesh imports both lazily and
# neither falls back gracefully, so their absence only surfaces once that specific code path
# runs, not at trimesh's own import time. Checking importability rather than unconditionally
# re-running pip is what makes a repeat run fast (torch is a large download).
NEED_INSTALL=0
python -c "import numpy, scipy, torch, trimesh, skimage, rtree, fast_simplification" \
  >/dev/null 2>&1 || NEED_INSTALL=1
if [ "$NEED_INSTALL" = "1" ] || [ "${FORCE_DEPS:-0}" = "1" ]; then
  echo "[venv] installing dependencies (this can take a while, especially torch)"
  _ensure_pip
  python -m pip install --upgrade pip -q
  python -m pip install "numpy>=1.24" "scipy>=1.10" "torch>=2.1" \
    trimesh scikit-image rtree fast_simplification -q
  python -c "import numpy, scipy, torch, trimesh, skimage, rtree, fast_simplification" || {
    echo "ERROR: dependency install ran but imports still fail; see the pip output above." >&2
    exit 1
  }
  echo "[venv] dependencies installed"
else
  echo "[venv] dependencies already satisfied (numpy, scipy, torch, trimesh, skimage," \
       "rtree, fast_simplification)"
fi

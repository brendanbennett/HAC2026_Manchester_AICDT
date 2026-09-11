"""Import nvdiffrast with its CUDA runtime preloaded.

The extension is built against CUDA 12.9 (scripts/setup_toolchain.sh says why) while torch
ships its own CUDA 13 runtime, so `_nvdiffrast_c.so` needs a libcudart.so.12 that is not on
the default loader path. Preloading it with RTLD_GLOBAL before the import avoids setting
LD_LIBRARY_PATH in every shell. Setting HAC26_SKIP_CUDART_PRELOAD skips the preload.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

_CANDIDATES = [Path.home() / ".local/cuda129/lib", Path.home() / ".local/cuda129/lib64"]


def _preload() -> None:
    if os.environ.get("HAC26_SKIP_CUDART_PRELOAD"):
        return
    for d in _CANDIDATES:
        so = d / "libcudart.so.12"
        if so.exists():
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            return


def load():
    """Return nvdiffrast.torch, or raise ImportError with the reason."""
    _preload()
    try:
        import nvdiffrast.torch as dr
    except ImportError as e:
        # The extension links libc10/libtorch, so a torch that is not the one it was built
        # against fails here on a mangled C++ symbol that names neither torch nor the fix.
        # `make venv` rebuilds it when it swaps torch itself; this catches the cases it
        # cannot see, such as a torch installed by hand.
        if "undefined symbol" in str(e):
            import torch

            raise ImportError(
                f"nvdiffrast's compiled extension does not match torch {torch.__version__} "
                f"in this environment -- it was built against a different one. Rebuild it "
                f"with `make toolchain`. ({e})"
            ) from e
        raise
    return dr

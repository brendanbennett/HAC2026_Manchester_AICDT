"""Import nvdiffrast with its CUDA runtime preloaded.

The extension is compiled against CUDA 12.9 (see tools/setup_toolchain.sh for why that is
the only consistent choice on this machine) while torch ships cu13, so `_nvdiffrast_c.so`
needs libcudart.so.12 which is not on the default loader path. Preloading it RTLD_GLOBAL
before the import satisfies that without requiring LD_LIBRARY_PATH to be set in every shell
that runs a test.
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
    import nvdiffrast.torch as dr
    return dr

"""Disk format for a saved shape library: one .npz per body, plus a JSON manifest.

Kept separate from `shape_library.py`'s generation code because it has nothing to do with
CSG or extraction -- it's read by `scripts/fit_shapes.py` too, which should not have to
import the generator to load a library someone already built.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .shape_library import Body

__all__ = ["save_body", "load_body", "write_manifest", "read_manifest", "load_library_dir"]


def save_body(path: str, body: Body) -> None:
    np.savez_compressed(
        path,
        verts=body.verts.astype(np.float32),
        faces=body.faces.astype(np.int32),
        recipe=json.dumps(body.recipe),
        info=json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in body.info.items()}),
    )


def load_body(path: str) -> Body:
    z = np.load(path, allow_pickle=False)
    return Body(z["verts"].astype(np.float64), z["faces"].astype(np.int64),
               json.loads(str(z["recipe"])), json.loads(str(z["info"])))


def write_manifest(directory: str, entries: list[dict]) -> None:
    """`entries`: list of {"file": ..., "index": ..., "base": ..., "convexity": ..., ...}."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(Path(directory) / "manifest.json", "w") as fh:
        json.dump({"n": len(entries), "entries": entries}, fh, indent=1)


def read_manifest(directory: str) -> dict:
    p = Path(directory) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no manifest.json in {directory}; run scripts/build_shape_library.py first")
    with open(p) as fh:
        return json.load(fh)


def load_library_dir(directory: str, n: int | None = None, seed: int | None = None) -> list:
    """Bodies from a directory `build_shape_library.py` wrote, as (verts, faces) pairs --
    the exact tuple shape `train_surrogate.shapes()` returns, so it's a drop-in source.

    `seed` shuffles the manifest order before truncating to `n`, so different callers (or
    different `--bodies` counts against the same directory) don't all get the library's
    first N regardless of how many they asked for.
    """
    man = read_manifest(directory)
    entries = list(man["entries"])
    if seed is not None:
        np.random.default_rng(seed).shuffle(entries)
    if n is not None:
        entries = entries[:n]
    out = []
    for e in entries:
        b = load_body(str(Path(directory) / e["file"]))
        out.append((b.verts, b.faces))
    return out

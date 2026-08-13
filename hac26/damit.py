"""Load DAMIT (Database of Asteroid Models from Inversion Techniques) shape models.

DAMIT models are triangulated shapes derived from lightcurve inversion (mostly convex-
inversion outputs, so their convex hull ~= the model). We parse the standard `shape.txt`
files and expose them as a pool of (verts, faces) meshes for use as a training prior in
place of the synthetic bodies in shapes.sample_training_shape.

shape.txt format (DAMIT docs):
    line 1:            N_vertices  N_facets
    next N_vertices:   x y z           (floats)
    next N_facets:     v1 v2 v3        (1-based vertex indices; optional leading "3")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_shape_txt(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a DAMIT shape.txt into (verts (V,3) float, faces (F,3) int, 0-based)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    nv, nf = (int(x) for x in lines[0].split()[:2])
    verts = np.array([[float(t) for t in lines[1 + i].split()[:3]] for i in range(nv)],
                     dtype=np.float64)
    faces = np.empty((nf, 3), dtype=np.int64)
    for j in range(nf):
        toks = lines[1 + nv + j].split()
        # last 3 tokens are the vertex indices (tolerates an optional leading count "3")
        faces[j] = [int(t) for t in toks[-3:]]
    faces -= 1  # DAMIT indices are 1-based
    return verts, faces


def load_damit_pool(directory: str, max_models: int | None = None,
                    min_verts: int = 20) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load all shape.txt meshes under `directory`. Centers each at its centroid.

    Returns a list of (verts, faces). Skips files that fail to parse or are degenerate.
    `max_models` caps the pool (deterministic: sorted by path) to bound memory.
    """
    paths = sorted(Path(directory).rglob("shape.txt"))
    if max_models is not None and len(paths) > max_models:
        # evenly-spaced subsample so the pool spans all asteroids, not just the
        # alphabetically-first ones (deterministic; no RNG so workers agree).
        idx = np.linspace(0, len(paths) - 1, max_models).round().astype(int)
        paths = [paths[i] for i in np.unique(idx)]
    pool: list[tuple[np.ndarray, np.ndarray]] = []
    for p in paths:
        try:
            v, f = parse_shape_txt(p.read_text())
        except Exception:
            continue
        if v.shape[0] < min_verts or f.shape[0] < min_verts:
            continue
        if not np.isfinite(v).all():
            continue
        v = v - v.mean(axis=0)  # center; scale/pose handled downstream (rescale_touch_z)
        pool.append((v, f))
        if max_models is not None and len(pool) >= max_models:
            break
    return pool

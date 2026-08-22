#!/usr/bin/env python3
"""Deliberately out-of-family meshes for a representation stress test.

Nothing here is built from `hac26.shape_library`'s primitives or modifiers. Each shape is
chosen to violate a different assumption the library's nine base kinds share -- they are all
blobs of roughly one scale, star-shaped or nearly so, with concavities that are dents in a
surface rather than structure.

    jack        six slender arms on the axes. The concavity between arms is the whole body,
                not a dent in it; nothing in the library has a limb thinner than its core.
    cup         a thick-walled open bowl. One deep cavity behind a narrow mouth -- the
                library's `basin` is wide and shallow by construction.
    trefoil     a tube swept along a trefoil knot. Genus 1 like `arch`, but the material is
                a thin curved tube everywhere; there is no bulk at all.
    steps       a rectilinear staircase block. Sharp orthogonal steps, no curvature, and a
                silhouette that is a different polygon from every direction.

Each is returned as a closed triangle mesh from marching cubes on its own analytic field, at
a resolution well above the 96 the library extracts at, so what `body_from_mesh` ingests is
limited by ITS grid rather than by this one.
"""
from __future__ import annotations

import numpy as np
from skimage import measure

__all__ = ["oof_meshes", "jack", "cup", "trefoil", "steps"]


def _grid(res: int, extent: float):
    a = np.linspace(-extent, extent, res)
    X, Y, Z = np.meshgrid(a, a, a, indexing="ij")
    return np.stack([X, Y, Z], -1), a


def _mesh(f, res: int = 96, extent: float = 1.5):
    """Marching cubes on f < 0, returned in world coordinates."""
    P, a = _grid(res, extent)
    V = f(P.reshape(-1, 3)).reshape(res, res, res)
    if not (V.min() < 0.0 < V.max()):
        raise ValueError("field has no zero crossing on the grid")
    verts, faces, _, _ = measure.marching_cubes(V, level=0.0)
    step = a[1] - a[0]
    verts = verts * step - extent
    return np.ascontiguousarray(verts, dtype=float), np.ascontiguousarray(faces, dtype=np.int64)


def _capsule(p, a, b, r):
    """Distance to a segment [a, b] minus r."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=-1) - r


def jack(arm_r: float = 0.17, arm_l: float = 1.15, core_r: float = 0.30):
    """Six slender arms along +-x, +-y, +-z, on a small central ball."""
    def f(p):
        d = np.linalg.norm(p, axis=-1) - core_r
        for ax in range(3):
            for s in (+1.0, -1.0):
                e = np.zeros(3); e[ax] = s * arm_l
                d = np.minimum(d, _capsule(p, np.zeros(3), e, arm_r))
        return d
    return _mesh(f)


def cup(outer: float = 0.95, wall: float = 0.20, mouth_z: float = 0.30):
    """Thick-walled bowl: a sphere with a smaller sphere removed, opened by a flat cut."""
    r_in = outer - wall

    def f(p):
        r = np.linalg.norm(p, axis=-1)
        rxy = np.linalg.norm(p[..., :2], axis=-1)
        # cavity: the inner ball, opened upward by a bore of the same radius
        bore = np.maximum(rxy - r_in, mouth_z - p[..., 2])
        cavity = np.minimum(r - r_in, bore)
        return np.maximum(r - outer, -cavity)
    return _mesh(f)


def trefoil(tube_r: float = 0.17, n_seg: int = 220, scale: float = 0.42):
    """A tube of constant radius swept along a trefoil knot."""
    t = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    C = scale * np.stack([np.sin(t) + 2 * np.sin(2 * t),
                          np.cos(t) - 2 * np.cos(2 * t),
                          -np.sin(3 * t)], axis=1)

    def f(p):
        out = np.full(len(p), np.inf)
        for i in range(0, len(C), 8):                    # chunk the segments, keep memory flat
            seg = C[i:i + 9]
            for j in range(len(seg) - 1):
                out = np.minimum(out, _capsule(p, seg[j], seg[j + 1], tube_r))
        out = np.minimum(out, _capsule(p, C[-1], C[0], tube_r))
        return out
    return _mesh(f)


def steps(n: int = 4, half: float = 0.95):
    """A staircase: n rectilinear treads of decreasing width, sharp orthogonal edges."""
    def box(p, lo, hi):
        q = np.maximum(lo - p, p - hi)
        return np.max(q, axis=-1)

    def f(p):
        d = np.full(len(p), np.inf)
        for k in range(n):
            zlo = -half + 2 * half * k / n
            zhi = -half + 2 * half * (k + 1) / n
            w = half * (1.0 - 0.78 * k / n)
            lo = np.array([-half, -w, zlo])
            hi = np.array([half - 1.55 * half * k / n, w, zhi])
            d = np.minimum(d, box(p, lo, hi))
        return d
    return _mesh(f)


def oof_meshes() -> dict:
    return {"jack": jack(), "cup": cup(), "trefoil": trefoil(), "steps": steps()}


if __name__ == "__main__":
    for k, (v, fc) in oof_meshes().items():
        print(f"{k:9s} verts {len(v):6d}  faces {len(fc):6d}  "
              f"extent {np.abs(v).max():.3f}")

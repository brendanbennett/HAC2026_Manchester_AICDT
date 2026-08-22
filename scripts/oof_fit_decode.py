#!/usr/bin/env python3
"""Stage 2: fit the field to each ingested body and decode it back, in numpy.

WHY NOT `scripts/fit_shapes.py` DIRECTLY. That path needs torch (for `ImplicitBody` and for
FlexiCubes inside `extract_mesh`), which is not installable in this sandbox. Everything it
does that matters here is reproduced exactly:

    core(y) = max_j (n_j . y - h_j)      same committed hac26/design4096.npy normals
    h       = max_v <n_j, v>             frozen at the hull support, as BatchedFit pins it
    Delta   = sum_k g_k exp(-||(y-p_k)/sigma||^2/2)   same 12^3 lattice, same sigma
    samples = the same 6000 uniform + 3000 surface-jittered points over the same extent

TWO DELIBERATE DEVIATIONS, both of which make this a CEILING rather than a replica:

1. g is solved in closed form, not by Adam. The residual is linear in g -- Delta is a fixed
   basis, not a network -- so `min_g ||core + Phi g - S||^2` is a normal-equation solve. The
   3000 Adam steps the real fit runs from g = 0 cannot beat it. Any shortfall here is
   therefore the REPRESENTATION's ceiling, not the optimiser's; the real pipeline can only do
   worse. A small ridge stands in for the implicit regularisation of a finite-step fit.

2. Decoding uses marching cubes rather than FlexiCubes. HANDOFF section 7 measured extraction
   resolution 24 -> 64 as worth 0.007 in convexity, so the extractor is not where the
   interesting variance lives, but the two are not identical and the numbers below should not
   be quoted against FlexiCubes numbers to three decimals.

Scoring is voxel Dice on a common grid, which is the identity the challenge measure reduces
to (see hac26/recon.py), plus the g = 0 core-only floor for the same body -- without that
floor a Dice figure has no scale, since the hull alone already scores about 0.93.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.shape_library import _parity_occupancy, convexity_ratio  # noqa: E402

LATTICE_SHAPE = (12, 12, 12)
LATTICE_EXTENT = 1.1
LATTICE_ALPHA = 0.9
SAMPLE_EXTENT = LATTICE_EXTENT + 3.0 * LATTICE_ALPHA * (2.0 * LATTICE_EXTENT / 12)


def lattice():
    axes, spacing = [], []
    for n_ax in LATTICE_SHAPE:
        step = 2.0 * LATTICE_EXTENT / n_ax
        axes.append(-LATTICE_EXTENT + (np.arange(n_ax) + 0.5) * step)
        spacing.append(step)
    p = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inv2 = 1.0 / (LATTICE_ALPHA * np.asarray(spacing)) ** 2
    return p.astype(np.float64), inv2.astype(np.float64)


def basis(P, p, inv2, chunk=2000):
    """Phi[i, k] = exp(-||(y_i - p_k)/sigma||^2 / 2), the expanded-square form."""
    pb = (p ** 2 * inv2).sum(1)
    out = np.empty((len(P), len(p)))
    for i in range(0, len(P), chunk):
        Q = P[i:i + chunk]
        d2 = (Q ** 2 * inv2).sum(-1, keepdims=True) + pb - 2.0 * ((Q * inv2) @ p.T)
        out[i:i + chunk] = np.exp(-0.5 * np.maximum(d2, 0.0))
    return out


def core_values(P, n, h, chunk=2000):
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        out[i:i + chunk] = (P[i:i + chunk] @ n.T - h).max(-1)
    return out


def sample_sdf(verts, faces, n_pts, seed):
    """Identical to fit_shapes.sample_arrays."""
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    rng = np.random.default_rng(seed)
    ext = max(float(np.abs(verts).max()) * 1.3, SAMPLE_EXTENT)
    pts = rng.uniform(-ext, ext, (n_pts, 3))
    surf, _ = trimesh.sample.sample_surface(m, n_pts // 2)
    pts = np.vstack([pts, surf + rng.normal(0, 0.03, surf.shape)])
    # Chunked. trimesh's signed_distance allocates against (points x faces): measured 1.5 GB
    # for 2000 points against a 58k-face body, so the 9000 points fit_shapes asks for would
    # need about 7 GB on that same body. fit_shapes calls it unchunked, and with --workers 16
    # that is a real exposure on the remote build, not an artefact of this sandbox.
    sd = np.concatenate([-m.nearest.signed_distance(pts[i:i + 500])
                         for i in range(0, len(pts), 500)])
    return pts, sd


def occupancy(verts, faces, res, extent, chunk=400):
    """`shape_library._parity_occupancy`, same arithmetic, smaller triangle chunks.

    The library version fixes the chunk at 3000, which allocates five (res^2, 3000) float64
    arrays -- about 1.1 GB at res=96 -- regardless of how big the mesh is. That is fine on a
    workstation and fatal on a 4 GB box for a 58k-face body.
    """
    verts = np.asarray(verts, float); faces = np.asarray(faces, np.int64)
    a = np.linspace(-extent, extent, res)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    X, Y = np.meshgrid(a, a, indexing="ij")
    px, py = X.ravel(), Y.ravel()
    count = np.zeros((len(px), res), dtype=np.int32)
    for t in range(0, len(faces), chunk):
        A, B, C = v0[t:t + chunk], v1[t:t + chunk], v2[t:t + chunk]
        d = ((B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0])
             + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1]))
        ok = np.abs(d) > 1e-14
        if not ok.any():
            continue
        A, B, C, d = A[ok], B[ok], C[ok], d[ok]
        l1 = ((B[None, :, 1] - C[None, :, 1]) * (px[:, None] - C[None, :, 0])
              + (C[None, :, 0] - B[None, :, 0]) * (py[:, None] - C[None, :, 1])) / d
        l2 = ((C[None, :, 1] - A[None, :, 1]) * (px[:, None] - C[None, :, 0])
              + (A[None, :, 0] - C[None, :, 0]) * (py[:, None] - C[None, :, 1])) / d
        l3 = 1.0 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not inside.any():
            continue
        zh = l1 * A[None, :, 2] + l2 * B[None, :, 2] + l3 * C[None, :, 2]
        col, tri = np.nonzero(inside)
        idx = np.clip(np.searchsorted(a, zh[col, tri]), 0, res - 1)
        np.add.at(count, (col, idx), 1)
    occ = (np.cumsum(count[:, ::-1], axis=1)[:, ::-1] % 2 == 1)
    return occ.reshape(res, res, res)


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    return float(2.0 * inter / max(a.sum() + b.sum(), 1))


def decode(n, h, p, inv2, g, res, extent):
    a = np.linspace(-extent, extent, res)
    G = np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)
    vals = np.empty(len(G))
    step = 40000
    for i in range(0, len(G), step):
        Q = G[i:i + step]
        v = core_values(Q, n, h)
        if g is not None:
            v = v + basis(Q, p, inv2) @ g
        vals[i:i + step] = v
    V = vals.reshape(res, res, res)
    if not (V.min() < 0.0 < V.max()):
        return None, None
    verts, faces, _, _ = measure.marching_cubes(V, level=0.0)
    verts = verts * (a[1] - a[0]) - extent
    return np.ascontiguousarray(verts, float), np.ascontiguousarray(faces, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True)
    ap.add_argument("--bodies-dir", default="runs/audit/bodies")
    ap.add_argument("--out", default="runs/audit/fits")
    ap.add_argument("--points", type=int, default=6000)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--decode-res", type=int, default=64)
    ap.add_argument("--score-res", type=int, default=96)
    ap.add_argument("--extent", type=float, default=1.6)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{a.only}.json"
    if dst.exists():
        print(f"[skip] {dst} exists"); return

    z = np.load(Path(a.bodies_dir) / f"{a.only}.npz")
    verts, faces = z["verts"].astype(float), z["faces"].astype(np.int64)
    meta = json.loads(str(z["meta"]))

    n = np.load("hac26/design4096.npy").astype(np.float64)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    p, inv2 = lattice()

    t0 = time.time()
    P, S = sample_sdf(verts, faces, a.points, seed=0)
    h = np.maximum((verts @ n.T).max(axis=0), 1e-3)
    C = core_values(P, n, h)
    Phi = basis(P, p, inv2)
    r = S - C
    A = Phi.T @ Phi + a.ridge * len(P) * np.eye(Phi.shape[1])
    g = np.linalg.solve(A, Phi.T @ r)

    res_core = float(np.mean(r ** 2))
    res_fit = float(np.mean((Phi @ g - r) ** 2))

    truth_occ = occupancy(verts, faces, a.score_res, a.extent)
    rec = {"name": a.only, **meta, "points": int(a.points), "ridge": a.ridge,
           "sdf_mse_core_only": res_core, "sdf_mse_fitted": res_fit,
           "g_absmax": float(np.abs(g).max()), "g_rms": float(np.sqrt((g ** 2).mean())),
           "truth_convexity": float(convexity_ratio(verts, faces))}

    for tag, gg in (("core_only", None), ("fitted", g)):
        dv, df = decode(n, h, p, inv2, gg, a.decode_res, a.extent)
        if dv is None:
            rec[f"dice_{tag}"] = None
            continue
        occ = occupancy(dv, df, a.score_res, a.extent)
        rec[f"dice_{tag}"] = dice(truth_occ, occ)
        rec[f"convexity_{tag}"] = float(convexity_ratio(dv, df))
        rec[f"volume_ratio_{tag}"] = float(occ.sum() / max(truth_occ.sum(), 1))
        if tag == "fitted":
            np.savez_compressed(out / f"{a.only}_decoded.npz", verts=dv, faces=df, g=g)

    rec["seconds"] = round(time.time() - t0, 1)
    dst.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

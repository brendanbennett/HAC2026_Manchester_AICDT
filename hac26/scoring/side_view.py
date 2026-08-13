#!/usr/bin/env python3
"""The challenge's second scoring measure: the distance between projection boundary curves.

The score is the sum over the secret bodies of two measures, so this is half of it.

Side views are taken as horizontal directions, elevation 0. Distance between two closed
boundary curves is reported as the symmetric mean nearest-neighbour distance and the
Hausdorff distance, in model units; the body spans z in [-1, 1], so 0.01 is 0.5% of its
height.

This measure sees non-convexity directly: the projection of a convex hull is the convex hull
of the projection, so a neck or waist appears as a concave stretch of the outline that no
convex reconstruction can produce.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from hac26.shapes import hull_mesh, rescale_touch_z  # noqa: E402
from hac26.stl_io import load_stl  # noqa: E402


def surface_points(verts, faces, n=1_000_000, seed=0):
    import trimesh
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(pts)


def silhouette_contours(pts, e1, e2, ext, res):
    """Binary silhouette from projected surface samples, then its boundary curves."""
    u = pts @ e1
    v = pts @ e2
    ix = np.clip(((u / ext) * 0.5 + 0.5) * (res - 1), 0, res - 1).astype(int)
    iy = np.clip(((v / ext) * 0.5 + 0.5) * (res - 1), 0, res - 1).astype(int)
    img = np.zeros((res, res), dtype=bool)
    img[iy, ix] = True
    img = ndimage.binary_closing(img, iterations=2)
    img = ndimage.binary_fill_holes(img)
    from skimage import measure
    cons = measure.find_contours(img.astype(float), 0.5)
    if not cons:
        return None
    pix = 2.0 * ext / (res - 1)
    pooled = np.concatenate(cons, axis=0)
    return (pooled - (res - 1) / 2.0) * pix          # back to model units


def boundary_distance(ca, cb):
    """Symmetric mean nearest-neighbour distance and Hausdorff, model units."""
    ta, tb = cKDTree(ca), cKDTree(cb)
    dab = tb.query(ca)[0]
    dba = ta.query(cb)[0]
    assd = 0.5 * (dab.mean() + dba.mean())
    haus = max(dab.max(), dba.max())
    return assd, haus


GOLDEN = 0.5 * (5 ** 0.5 - 1.0)          # 0.6180339887..., the golden ratio conjugate


def projection_directions(n: int = 36, mode: str = "side"):
    """The viewing directions whose outlines are compared. Returns (n, 3) unit vectors.

    NOT UNIFORM. Azimuths at 2 pi k / n resonate with any body whose symmetry order shares a
    factor with n: at n = 36 a 4-fold body -- a cube, which challenge model 2 is -- yields
    only 9 distinct outlines, each sampled four times. The worst direction is then invisible
    and the spread is understated, both in the direction that flatters the reconstruction.

    Azimuths advance by the golden angle instead, theta_k = 2 pi frac(k * GOLDEN). An
    irrational rotation number cannot resonate with any integer symmetry, and the sequence is
    low-discrepancy: its star discrepancy falls as log(n)/n against 1/sqrt(n) for random
    directions, so a given number of views estimates the mean outline distance more tightly.

    mode "side" keeps elevation 0, which is what a side view is and what the rules say. mode
    "sphere" spreads directions over the whole sphere by the same golden-angle spiral, for
    checking that a result is not an artefact of the equatorial band.
    """
    k = np.arange(n)
    if mode == "sphere":
        z = 1.0 - 2.0 * (k + 0.5) / n
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        th = 2.0 * np.pi * ((k * GOLDEN) % 1.0)
        return np.stack([r * np.cos(th), r * np.sin(th), z], 1)
    th = 2.0 * np.pi * ((k * GOLDEN) % 1.0)
    return np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)


def side_view_measure(pts_a, pts_b, n_dirs=36, res=512, mode="side"):
    """Aggregate boundary distance over well-spread viewing directions."""
    ext = 1.05 * max(np.abs(pts_a).max(), np.abs(pts_b).max())
    assds, hauss = [], []
    up = np.array([0.0, 0.0, 1.0])
    for v in projection_directions(n_dirs, mode):
        e1 = np.cross(up, v)
        if np.linalg.norm(e1) < 1e-8:
            e1 = np.array([1.0, 0.0, 0.0])
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(v, e1)
        e2 = e2 / np.linalg.norm(e2)
        ca = silhouette_contours(pts_a, e1, e2, ext, res)
        cb = silhouette_contours(pts_b, e1, e2, ext, res)
        if ca is None or cb is None:
            continue
        a, h = boundary_distance(ca, cb)
        assds.append(a)
        hauss.append(h)
    return {"assd_mean": float(np.mean(assds)), "assd_worst": float(np.max(assds)),
            "hausdorff_mean": float(np.mean(hauss)),
            "hausdorff_worst": float(np.max(hauss)), "n_dirs": len(assds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--recon-dir", default="results/lpd")
    ap.add_argument("--data-dir", default="../data/raw")
    ap.add_argument("--n-dirs", type=int, default=36)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default="projection_scores.json")
    args = ap.parse_args()

    out = {}
    for M in args.models:
        tf = glob.glob(f"{args.data_dir}/AsteroidModel0{M}_shape_public/asteroid{M}.stl")
        if not tf:
            print(f"model {M}: no truth STL", flush=True)
            continue
        tv, tfc = load_stl(tf[0])
        tv = rescale_touch_z(tv)
        tp = surface_points(tv, tfc)

        rows = {}
        # self-distance: the resolution floor of this implementation, for calibration
        rows["truth_vs_truth"] = side_view_measure(
            tp, surface_points(tv, tfc, seed=1), args.n_dirs, args.res)
        # the convex ceiling under THIS measure: what hulling the truth costs
        hv, hf = hull_mesh(tv)
        rows["hull_vs_truth"] = side_view_measure(
            surface_points(hv, hf), tp, args.n_dirs, args.res)
        # the reconstruction
        rf = Path(args.recon_dir) / f"Asteroid{M:02d}.stl"
        if rf.exists():
            rv, rfc = load_stl(str(rf))
            rows["shipped_vs_truth"] = side_view_measure(
                surface_points(rv, rfc), tp, args.n_dirs, args.res)

        out[M] = rows
        print(f"\nmodel {M}", flush=True)
        for k, r in rows.items():
            print(f"  {k:<18} ASSD {r['assd_mean']:.4f} (worst dir {r['assd_worst']:.4f})"
                  f"   Hausdorff {r['hausdorff_mean']:.4f}", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

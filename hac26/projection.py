#!/usr/bin/env python3
"""The challenge's SECOND scoring measure: side-view boundary-curve distance.

From the challenge page (fips.fi, section "Rules"):

    "Side-view measure: Look at 2D projections along unspecified directions.
     We calculate the distance between two boundary curves."

and the total is "the highest sum of all scores over the 7 secret asteroid models
(minimum score 0, maximum score 14)" -- seven models times TWO measures. So this measure
is HALF the score, and this project has never computed it: every selection and every
report so far used the voxel measure only.

The official Matlab/Python code is "released later"; until then this implements the
plain reading. "Side-view" is taken as horizontal directions (elevation 0), since that
is what a side view is; a ring of azimuths stands in for "unspecified directions".
Distance between two closed boundary curves is reported as the symmetric mean
nearest-neighbour distance (ASSD) and the Hausdorff distance, in model units
(the body spans z in [-1, 1], so 0.01 = 0.5% of the body height).

Why this matters for the non-convex direction specifically: projections of conv(K) are
the convex hulls of projections of K, so a neck or waist -- the signature of a contact
binary like Mithra -- shows up DIRECTLY as a concave stretch of the outline that a convex
reconstruction cannot produce. The hull row below measures exactly what that costs.

    python eval_projection.py --models 1 2 3 --recon-dir ../data/eval_final2
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


def side_view_measure(pts_a, pts_b, n_dirs=36, res=512):
    """Aggregate boundary distance over a ring of horizontal viewing directions."""
    ext = 1.05 * max(np.abs(pts_a).max(), np.abs(pts_b).max())
    assds, hauss = [], []
    for k in range(n_dirs):
        th = 2 * np.pi * k / n_dirs
        v = np.array([np.cos(th), np.sin(th), 0.0])
        e1 = np.array([-np.sin(th), np.cos(th), 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
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
    ap.add_argument("--recon-dir", default="../data/eval_final2")
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

#!/usr/bin/env python3
"""The challenge's second scoring measure: the distance between the boundary curves of two
bodies' side-view projections.

Side views are taken along horizontal directions, at elevation 0. The distance between two
closed boundary curves is reported as the symmetric mean nearest-neighbour distance (ASSD)
and the Hausdorff distance, both in model units.

This measure sees non-convexity directly: the projection of a convex hull is the convex hull
of the projection, so a neck or waist shows as a concave stretch of the outline that no convex
reconstruction can produce.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from hac26.data_io import public_stl  # noqa: E402
from hac26.shapes import hull_mesh, rescale_touch_z  # noqa: E402
from hac26.stl_io import load_stl  # noqa: E402


def surface_points(verts, faces, n=1_000_000, seed=0):
    """Uniform random points on the mesh surface, shape (n, 3)."""
    import trimesh
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(pts)


def silhouette_contours(pts, e1, e2, ext, res):
    """Project the points onto the (e1, e2) image plane, fill the silhouette, and return its
    boundary curves in model units (None if the silhouette is empty)."""
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


GOLDEN = 0.5 * (5 ** 0.5 - 1.0)          # the golden ratio conjugate


def projection_directions(n: int = 36, mode: str = "side"):
    """Unit viewing directions whose outlines are compared, shape (n, 3).

    Azimuths advance by the golden angle, theta_k = 2 pi frac(k * GOLDEN), rather than by
    2 pi / n. Equally spaced azimuths resonate with a body whose symmetry order shares a factor
    with n, so several views repeat the same outline and the worst direction can be missed. An
    irrational step cannot resonate with any symmetry, and the sequence is low-discrepancy, so
    a given number of views estimates the mean outline distance more tightly.

    mode "side" keeps elevation 0, which is what a side view is. mode "sphere" spreads the
    directions over the whole sphere by the same spiral, for checking that a result is not an
    artefact of the equatorial band.
    """
    k = np.arange(n)
    if mode == "sphere":
        z = 1.0 - 2.0 * (k + 0.5) / n
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        th = 2.0 * np.pi * ((k * GOLDEN) % 1.0)
        return np.stack([r * np.cos(th), r * np.sin(th), z], 1)
    th = 2.0 * np.pi * ((k * GOLDEN) % 1.0)
    return np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)


def view_frames(n_dirs=36, mode="side"):
    """The (e1, e2) image axes for each viewing direction, in order."""
    up = np.array([0.0, 0.0, 1.0])
    out = []
    for v in projection_directions(n_dirs, mode):
        e1 = np.cross(up, v)
        if np.linalg.norm(e1) < 1e-8:
            e1 = np.array([1.0, 0.0, 0.0])
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(v, e1)
        e2 = e2 / np.linalg.norm(e2)
        out.append((e1, e2))
    return out


def outline_set(pts, ext, n_dirs=36, res=512, mode="side"):
    """One cloud's boundary curves, one per viewing direction (None where the view is empty).

    Split out of side_view_measure so a caller comparing several clouds pairwise projects each
    of them once. `ext` must be shared across the clouds being compared: the contours are in
    model units, and a per-pair ext would put each pair on a different pixel pitch.
    """
    return [silhouette_contours(pts, e1, e2, ext, res) for e1, e2 in view_frames(n_dirs, mode)]


def measure_outlines(oa, ob):
    """Mean and worst ASSD and Hausdorff distance between two outline sets from `outline_set`,
    over the directions where both outlines exist."""
    assds, hauss = [], []
    for ca, cb in zip(oa, ob):
        if ca is None or cb is None:
            continue
        a, h = boundary_distance(ca, cb)
        assds.append(a)
        hauss.append(h)
    if not assds:
        return {"assd_mean": float("inf"), "assd_worst": float("inf"),
                "hausdorff_mean": float("inf"), "hausdorff_worst": float("inf"), "n_dirs": 0}
    return {"assd_mean": float(np.mean(assds)), "assd_worst": float(np.max(assds)),
            "hausdorff_mean": float(np.mean(hauss)),
            "hausdorff_worst": float(np.max(hauss)), "n_dirs": len(assds)}


def side_view_measure(pts_a, pts_b, n_dirs=36, res=512, mode="side"):
    """Aggregate boundary distance between two point clouds over the viewing directions."""
    ext = 1.05 * max(np.abs(pts_a).max(), np.abs(pts_b).max())
    return measure_outlines(outline_set(pts_a, ext, n_dirs, res, mode),
                            outline_set(pts_b, ext, n_dirs, res, mode))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, nargs="+", default=[1, 2, 3],
                    help="public models to score")
    ap.add_argument("--recon-dir", default="results/lpd",
                    help="directory holding the reconstructed Asteroid<NN>.stl files")
    ap.add_argument("--data-dir", default="dataset/raw", help="challenge data directory")
    ap.add_argument("--n-dirs", type=int, default=36, help="number of viewing directions")
    ap.add_argument("--res", type=int, default=512, help="silhouette image size in pixels")
    ap.add_argument("--out", default="projection_scores.json", help="output JSON file")
    args = ap.parse_args()

    out = {}
    for M in args.models:
        try:
            tf = public_stl(args.data_dir, M)
        except ValueError:
            tf = None
        if tf is None or not Path(tf).exists():
            print(f"model {M}: no truth STL", flush=True)
            continue
        tv, tfc = load_stl(tf)
        tv = rescale_touch_z(tv, tfc)
        tp = surface_points(tv, tfc)

        rows = {}
        # the truth against a second sampling of itself: the floor set by sampling and pixels
        rows["truth_vs_truth"] = side_view_measure(
            tp, surface_points(tv, tfc, seed=1), args.n_dirs, args.res)
        # the convex hull of the truth: what a convex reconstruction costs under this measure
        hv, hf = hull_mesh(tv)
        rows["hull_vs_truth"] = side_view_measure(
            surface_points(hv, hf), tp, args.n_dirs, args.res)
        # the reconstruction
        rf = Path(args.recon_dir) / f"Asteroid{M:02d}.stl"
        if rf.exists():
            rv, rfc = load_stl(str(rf))
            rv = rescale_touch_z(rv, rfc)
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

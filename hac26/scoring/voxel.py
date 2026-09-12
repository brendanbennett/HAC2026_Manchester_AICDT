#!/usr/bin/env python3
"""Voxel overlap of a reconstruction against the public ground truth.

The challenge defines its voxel measure as

    1 - ( #(A \\ B) + #(B \\ A) ) / ( #(A) + #(B) )

which is the Dice coefficient: #(A\\B) + #(B\\A) = #A + #B - 2#(A and B), so the expression
reduces to 2#(A and B) / (#A + #B).

Both meshes are posed with rescale_touch_z before voxelising, since the challenge fixes z to
[-1, 1] and comparing before that pose compares two different frames.

occupancy/prepare_truth/score_mesh/score_meshes/score_stls are the genetic-algorithm branch's
own repeated-scoring path (hac26.genetic_utils, scripts/reconstruct_genetic.py import
score_mesh and prepare_truth directly); score/main are main's CLI entry point, used by
scripts/run_pipeline.sh and scripts/run_remote_pipeline.sh's scoring stage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.data_io import public_stl               # noqa: E402
from hac26.recon import dice, mesh_occupancy       # noqa: E402
from hac26.shapes import rescale_touch_z           # noqa: E402

import trimesh

TRUTH = {1: "AsteroidModel01_shape_public/asteroid1.stl",
         2: "AsteroidModel02_shape_public/asteroid2.stl",
         3: "AsteroidModel03_shape_public/asteroid3.stl"}


def occupancy(v, f, n, extent):
    # hac26.recon.mesh_to_sdf was renamed to mesh_occupancy, which already returns the
    # boolean occupancy grid directly (same (v, f, n, extent) signature), so this wrapper is
    # now a thin pass-through kept for the genetic-algorithm functions below.
    return mesh_occupancy(np.ascontiguousarray(v), np.ascontiguousarray(f), n, extent)


def prepare_truth(
    truth_vertices,
    truth_faces,
    n=128,
    simplify_faces=None,
):
    """Prepare a truth mesh for repeated Dice evaluation."""

    truth_mesh = trimesh.Trimesh(
        vertices=np.asarray(truth_vertices),
        faces=np.asarray(truth_faces),
        process=False,
    )

    print(
        f"Truth mesh: "
        f"{len(truth_mesh.vertices):,} vertices, "
        f"{len(truth_mesh.faces):,} faces"
    )

    if simplify_faces is not None:
        truth_mesh = truth_mesh.simplify_quadric_decimation(
            face_count=simplify_faces
        )

        print(
            f"Simplified truth mesh: "
            f"{len(truth_mesh.vertices):,} vertices, "
            f"{len(truth_mesh.faces):,} faces"
        )

    # Already posed in the challenge frame (a released public STL, or this project's own
    # reconstruction) -- centre_xy=False leaves it there. Re-centering on its own solid
    # centroid (the old default) shifts it off the rotation axis, which is exactly the bug
    # main's score() was fixed for; prepare_truth/score_mesh need the identical fix, since a
    # truth mesh and a candidate built by two different pipelines drift apart under it in
    # a way two library-synthetic bodies from the same pipeline mostly do not.
    truth_vertices = rescale_touch_z(
        np.asarray(truth_mesh.vertices),
        centre_xy=False,
    )

    truth_faces = np.asarray(truth_mesh.faces)

    # Per-axis max understates the bounding radius whenever the extreme vertex isn't
    # axis-aligned (e.g. near 45 degrees in xy): a vector norm is the correct bound.
    extent = float(np.linalg.norm(truth_vertices, axis=1).max()) * 1.05

    truth_occupancy = occupancy(
        truth_vertices,
        truth_faces,
        n,
        extent,
    )

    return {
        "occupancy": truth_occupancy,
        "extent": extent,
        "n": n,
    }


def score_mesh(
    recon_vertices,
    recon_faces,
    truth,
):
    """Calculate voxel Dice against a prepared truth.

    Parameters
    ----------
    recon_vertices, recon_faces
        Vertices and triangular faces of the reconstruction.

    truth
        Prepared truth returned by ``prepare_truth``.

    Returns
    -------
    float
        Voxel Dice score.
    """

    # Same reasoning as prepare_truth above: leave the candidate on the rotation axis rather
    # than re-centering it on its own solid centroid.
    recon_vertices = rescale_touch_z(
        np.asarray(recon_vertices),
        centre_xy=False,
    )
    recon_faces = np.asarray(recon_faces)

    recon_occupancy = occupancy(
        recon_vertices,
        recon_faces,
        truth["n"],
        truth["extent"],
    )

    return float(
        dice(
            recon_occupancy,
            truth["occupancy"],
        )
    )


def score_meshes(
    recon_vertices,
    recon_faces,
    truth_vertices,
    truth_faces,
    n=128,
):
    """Calculate voxel Dice between two meshes.

    This is the backwards-compatible interface. For repeated scoring
    against the same truth, use ``prepare_truth`` and ``score_mesh``.
    """

    truth = prepare_truth(
        truth_vertices,
        truth_faces,
        n=n,
    )

    return score_mesh(
        recon_vertices,
        recon_faces,
        truth,
    )


def score_stls(
    recon_stl,
    truth_stl,
    n=128,
):
    """Calculate voxel Dice between two STL files."""

    import trimesh

    recon = trimesh.load(
        recon_stl,
        process=False,
    )

    truth = trimesh.load(
        truth_stl,
        process=False,
    )

    return score_meshes(
        recon.vertices,
        recon.faces,
        truth.vertices,
        truth.faces,
        n=n,
    )


def score_model(
    stl: str,
    model: int,
    data_dir: str = "dataset/raw",
    n: int = 128,
    ) -> float:
    """The genetic-algorithm branch's own convenience wrapper around score_stls, kept for
    backward compatibility; nothing outside this file calls it (score() below, main's own
    entry point, is what scripts/run_pipeline.sh and run_remote_pipeline.sh actually use)."""

    truth_stl = Path(data_dir) / TRUTH[model]

    return score_stls(
        stl,
        truth_stl,
        n=n,
    )


def load_one_solid(path: str):
    """An STL as a single mesh, warning if it is not closed.

    mesh_occupancy decides inside by a parity scan up each column, which is only valid for a
    closed surface: a hole inverts every cell in the column through it, so an open mesh
    scores wrongly rather than failing. The truth STLs are the organisers' files and are
    taken as they come, so this warns and continues rather than refusing to score.
    """
    import trimesh
    m = trimesh.load(path, process=False)
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"{path}: expected a single solid, got {type(m).__name__} -- a "
                         f"multi-solid STL has no one body to score")
    if not m.is_watertight:
        print(f"  WARNING: {path} is not closed; the parity scan that decides inside is "
              f"only valid for a closed surface, so this score is unreliable", flush=True)
    return m


def score(stl: str, model: int, data_dir: str = "dataset/raw", n: int = 128) -> float:
    """Dice between the reconstruction and the public truth, both posed, on one n^3 grid."""
    r = load_one_solid(stl)
    t = load_one_solid(public_stl(data_dir, model))
    # Both meshes are already in the challenge frame -- the truth as released, the
    # reconstruction as this package builds it -- so the pose only rescales z and leaves the
    # rotation axis where it is. Centring either on its own centroid would slide them apart.
    rv = rescale_touch_z(np.asarray(r.vertices), np.asarray(r.faces), centre_xy=False)
    tv = rescale_touch_z(np.asarray(t.vertices), np.asarray(t.faces), centre_xy=False)
    # Per-axis max understates the bounding radius whenever the extreme vertex isn't
    # axis-aligned (e.g. near 45 degrees in xy): a vector norm is the correct bound.
    e = max(float(np.linalg.norm(rv, axis=1).max()),
            float(np.linalg.norm(tv, axis=1).max())) * 1.05
    occ_r = mesh_occupancy(rv, np.asarray(r.faces), n, e)
    occ_t = mesh_occupancy(tv, np.asarray(t.faces), n, e)
    if not occ_r.any() or not occ_t.any():
        # dice() reads two empty grids as two identical bodies and returns 1.0, so a pair
        # that voxelised to nothing would be reported as a perfect reconstruction
        raise ValueError(f"nothing to compare for model {model}: the reconstruction "
                         f"voxelised to {int(occ_r.sum())} cells and the truth to "
                         f"{int(occ_t.sum())} on a {n}^3 grid of half-width {e:.4g}")
    return float(dice(occ_r, occ_t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", nargs="+", required=True,
                    help="one STL per public model, in the order given by --models")
    ap.add_argument("--models", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    out = {}
    for stl, m in zip(a.stl, a.models):
        out[m] = score(stl, m, a.data_dir, a.n)
        print(f"  model {m}: voxel measure {out[m]:.4f}   {stl}", flush=True)
    print(f"{a.label} summed voxel measure over {len(out)} models: {sum(out.values()):.4f}")
    print(json.dumps({"label": a.label, "dice": out, "sum": sum(out.values())}))


if __name__ == "__main__":
    main()

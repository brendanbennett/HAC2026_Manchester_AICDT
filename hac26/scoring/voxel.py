#!/usr/bin/env python3
"""Voxel overlap of a reconstruction against the public ground truth.

The challenge defines its voxel measure as

    1 - ( #(A \\ B) + #(B \\ A) ) / ( #(A) + #(B) )

which is the Dice coefficient: #(A\\B) + #(B\\A) = #A + #B - 2#(A and B), so the expression
reduces to 2#(A and B) / (#A + #B).

Both meshes are posed with rescale_touch_z before voxelising, since the challenge fixes z to
[-1, 1] and comparing before that pose compares two different frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.recon import dice, mesh_to_sdf          # noqa: E402
from hac26.shapes import rescale_touch_z           # noqa: E402

TRUTH = {1: "AsteroidModel01_shape_public/asteroid1.stl",
         2: "AsteroidModel02_shape_public/asteroid2.stl",
         3: "AsteroidModel03_shape_public/asteroid3.stl"}


def occupancy(v, f, n, extent):
    return mesh_to_sdf(np.ascontiguousarray(v), np.ascontiguousarray(f), n, extent) < 0


def prepare_truth(
    truth_vertices,
    truth_faces,
    n=128,
):
    """Prepare a truth mesh for repeated Dice evaluation.

    The returned object contains the voxelised truth and the voxel grid
    extent to use for all subsequent reconstructions.
    """

    truth_vertices = rescale_touch_z(
        np.asarray(truth_vertices)
    )
    truth_faces = np.asarray(truth_faces)

    extent = (
        float(np.abs(truth_vertices).max())
        * 1.05
    )

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

    recon_vertices = rescale_touch_z(
        np.asarray(recon_vertices)
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


def score(
    stl: str,
    model: int,
    data_dir: str = "dataset/raw",
    n: int = 128,
    ) -> float:

    truth_stl = Path(data_dir) / TRUTH[model]

    return score_stls(
        stl,
        truth_stl,
        n=n,
    )

# def score(stl: str, model: int, data_dir: str = "dataset/raw", n: int = 128) -> float:
#     import trimesh
#     r = trimesh.load(stl, process=False)
#     t = trimesh.load(Path(data_dir) / TRUTH[model], process=False)
#     rv, tv = rescale_touch_z(np.asarray(r.vertices)), rescale_touch_z(np.asarray(t.vertices))
#     e = max(float(np.abs(rv).max()), float(np.abs(tv).max())) * 1.05
#     return float(dice(occupancy(rv, np.asarray(r.faces), n, e),
#                       occupancy(tv, np.asarray(t.faces), n, e)))


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

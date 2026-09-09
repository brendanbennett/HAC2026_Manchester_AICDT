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

from hac26.data_io import public_stl               # noqa: E402
from hac26.recon import dice, mesh_occupancy       # noqa: E402
from hac26.shapes import rescale_touch_z           # noqa: E402


def score(stl: str, model: int, data_dir: str = "dataset/raw", n: int = 128) -> float:
    """Dice between the reconstruction and the public truth, both posed, on one n^3 grid."""
    import trimesh
    r = trimesh.load(stl, process=False)
    t = trimesh.load(public_stl(data_dir, model), process=False)
    # Both meshes are already in the challenge frame -- the truth as released, the
    # reconstruction as this package builds it -- so the pose only rescales z and leaves the
    # rotation axis where it is. Centring either on its own centroid would slide them apart.
    rv = rescale_touch_z(np.asarray(r.vertices), np.asarray(r.faces), centre_xy=False)
    tv = rescale_touch_z(np.asarray(t.vertices), np.asarray(t.faces), centre_xy=False)
    e = max(float(np.abs(rv).max()), float(np.abs(tv).max())) * 1.05
    return float(dice(mesh_occupancy(rv, np.asarray(r.faces), n, e),
                      mesh_occupancy(tv, np.asarray(t.faces), n, e)))


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

#!/usr/bin/env python3
"""Voxel Dice of a reconstructed STL against the public ground truth.

ONE scorer for every method in the repository. The two shipping bugs this project has had
were both the evaluated configuration and the written configuration quietly diverging, so
the convex baseline and the LPD flow are scored by the same code path on the same grid.

Both meshes are posed with rescale_touch_z first -- the challenge fixes z to [-1, 1], and
comparing before that pose is comparing two different frames.
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


def score(stl: str, model: int, data_dir: str = "data/raw", n: int = 128) -> float:
    import trimesh
    r = trimesh.load(stl, process=False)
    t = trimesh.load(Path(data_dir) / TRUTH[model], process=False)
    rv, tv = rescale_touch_z(np.asarray(r.vertices)), rescale_touch_z(np.asarray(t.vertices))
    e = max(float(np.abs(rv).max()), float(np.abs(tv).max())) * 1.05
    return float(dice(occupancy(rv, np.asarray(r.faces), n, e),
                      occupancy(tv, np.asarray(t.faces), n, e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", nargs="+", required=True,
                    help="one STL per public model, in the order given by --models")
    ap.add_argument("--models", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    out = {}
    for stl, m in zip(a.stl, a.models):
        out[m] = score(stl, m, a.data_dir, a.n)
        print(f"  model {m}: Dice {out[m]:.4f}   {stl}", flush=True)
    print(f"{a.label} summed Dice over {len(out)} public models: {sum(out.values()):.4f}")
    print(json.dumps({"label": a.label, "dice": out, "sum": sum(out.values())}))


if __name__ == "__main__":
    main()

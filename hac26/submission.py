#!/usr/bin/env python3
"""Check the ten written STLs against what the submission requires.

Per model: the file exists, loads and has a usable number of vertices and faces; the
challenge pose holds (z in [-1, 1] to a small tolerance, xy-centroid near the axis); and
the largest axis distance does not exceed the published bounding-cylinder radius
(hac26.conventions.CYLINDER_R) by more than a small tolerance. The radius check is
one-sided because the radius is a bound, not a target. Exits non-zero on any problem.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from hac26.conventions import CYLINDER_R
from hac26.shapes import hull_mesh
from hac26.stl_io import load_stl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="directory holding Asteroid01.stl to Asteroid10.stl")
    args = ap.parse_args()

    ok = True
    print(f"{'model':<7}{'verts':>8}{'faces':>8}{'zmin':>8}{'zmax':>8}"
          f"{'r_max':>8}{'R':>8}{'r/R':>7}  status")
    for M in range(1, 11):
        p = Path(args.dir) / f"Asteroid{M:02d}.stl"
        if not p.exists():
            print(f"{M:<7}  MISSING {p}")
            ok = False
            continue
        v, f = load_stl(str(p))
        hull_mesh(v)                 # raises if the vertices do not span a solid
        zmin, zmax = float(v[:, 2].min()), float(v[:, 2].max())
        r = float(np.sqrt((v[:, :2] ** 2).sum(1)).max())
        R = CYLINDER_R[M]
        cx, cy = float(v[:, 0].mean()), float(v[:, 1].mean())
        msgs = []
        if abs(zmin + 1) > 1e-3 or abs(zmax - 1) > 1e-3:
            msgs.append("POSE(z)")
        if max(abs(cx), abs(cy)) > 0.05 * max(1.0, r):
            msgs.append("POSE(xy)")
        if r > R * 1.02:
            msgs.append("OUTSIDE-CYLINDER")
        if len(v) < 8 or len(f) < 4:
            msgs.append("DEGENERATE")
        status = "ok" if not msgs else " ".join(msgs)
        if msgs:
            ok = False
        print(f"{M:<7}{len(v):>8}{len(f):>8}{zmin:>8.3f}{zmax:>8.3f}"
              f"{r:>8.3f}{R:>8.3f}{r/R:>7.3f}  {status}")
    print("\nALL CHECKS PASSED" if ok else "\nPROBLEMS FOUND -- see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Re-pose an existing reconstruction onto the published bounding cylinder.

WHY
---
`hac26.recon.fit_to_cylinder` exists and its own docstring records that the published R is
TIGHT in the challenge pose -- measured r/R = 0.99 / 1.03 / 1.01 on models 1/2/3 -- so R
pins the aspect ratio, which is the one degree of freedom a scale-free EGI inversion cannot
recover. But `reconstruct_lpd.py` only reads `R = CYLINDER_R[a.model]` to feed the solver;
it never applies `fit_to_cylinder` before `export_stl`. The results in `results/lpd/` are
therefore whatever width the solver happened to land on.

Measured on those STLs by `eval_gate.py`:

    model 1   R 1.003  against published 1.12   ->  xy is 0.90x too narrow
    model 2   R 0.840  against published 1.42   ->  xy is 0.59x too narrow
    model 3   R 0.892  against published 0.88   ->  correct

For a body scaled by s in xy and otherwise contained in the truth, Dice = 2s^2/(1+s^2).
That predicts 0.52 for model 2 and 0.89 for model 1, against the 0.5216 and 0.8786 actually
measured -- i.e. essentially ALL of model 2's loss, and most of model 1's, is aspect ratio,
not shape. Nothing about the corpus or the flow can fix a width error; one multiplication
can.

This script applies the pose the challenge asks for -- z rescaled to touch +-1, xy-centroid
on the axis, xy scaled so the minimum enclosing radius equals R -- and rewrites the STL.

    python scripts/fix_pose.py --stl results/lpd/Asteroid0{1,2,3}.stl \\
                               --models 1 2 3 --out-dir results/lpd_fitted

Then re-run eval_gate.py against `results/lpd_fitted/` and compare. If model 2 does not
move from 0.52 to roughly 0.85-0.89, the diagnosis above is wrong and I want to know.

NOTE ON THE SCALE FACTOR: R is scaled to 0.99 R by default, not exactly R. The published
value is an upper bound the body very nearly touches (measured min-enclosing radius over
published R = 0.99 / 0.99 / 0.98 on the three public models), so landing marginally inside
it is both closer to the truth and safe against a strict organiser-side cylinder check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.shapes import rescale_touch_z          # noqa: E402

CYLINDER_R = {1: 1.12, 2: 1.42, 3: 0.88, 4: 1.475, 5: 1.22,
              6: 0.925, 7: 1.205, 8: 1.24, 9: 0.67, 10: 3.95}


def min_enclosing_radius(p2: np.ndarray) -> tuple:
    """Radius and centre of the minimum enclosing circle of the xy projection.

    Not the radius about the vertex mean: on a body with uneven tessellation the mean sits
    off axis, and the radius then reads several percent high -- enough to make a compliant
    body look non-compliant and to mis-scale it if used for fitting.
    """
    from scipy.optimize import minimize
    from scipy.spatial import ConvexHull
    p = p2
    if len(p) > 3:
        try:
            p = p[ConvexHull(p).vertices]
        except Exception:
            pass
    r = minimize(lambda c: np.sqrt(((p - c) ** 2).sum(1)).max(), p.mean(0),
                 method="Nelder-Mead",
                 options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 6000})
    return float(r.fun), np.asarray(r.x, float)


def repose(v: np.ndarray, f: np.ndarray, R: float, fill: float = 0.99):
    v = rescale_touch_z(np.asarray(v, float))          # z -> [-1, 1]
    r0, c = min_enclosing_radius(v[:, :2])
    v[:, 0] -= c[0]                                     # put the axis through the circle
    v[:, 1] -= c[1]
    s = (fill * R) / max(r0, 1e-12)
    v[:, :2] *= s
    return v, f, r0, s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stl", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fill", type=float, default=0.99,
                    help="fraction of R to fill (default 0.99)")
    a = ap.parse_args()
    if len(a.stl) != len(a.models):
        raise SystemExit("--stl and --models must have the same length")

    import trimesh
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stl, m in zip(a.stl, a.models):
        R = CYLINDER_R[m]
        mesh = trimesh.load(stl, process=False)
        v, f, r0, s = repose(np.asarray(mesh.vertices, float),
                             np.asarray(mesh.faces, np.int64), R, a.fill)
        dst = out / Path(stl).name
        trimesh.Trimesh(v, f, process=False).export(str(dst), file_type="stl")
        pred = 2 * s * s / (1 + s * s) if s < 1 else None
        note = "" if pred is None else \
            f"   (a pure-width error of this size costs Dice {pred:.3f})"
        print(f"model {m}: R {r0:.3f} -> {a.fill * R:.3f}  (xy x {s:.3f}){note}")
        print(f"          wrote {dst}")


if __name__ == "__main__":
    sys.exit(main())

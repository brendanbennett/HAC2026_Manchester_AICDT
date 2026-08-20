#!/usr/bin/env python3
"""Acceptance gate for a corpus or checkpoint change, measured on the three public models.

WHY A GATE AND NOT A SCORE
--------------------------
The corpus rewrite in `hac26/shapes_nonconvex.py` is a bet: it trades a corpus that was
almost entirely convex (five of the seven old archetypes measured D_rms <= 0.021) for one
stratified over hull-deficit depth. The obvious failure mode is symmetric to the one it
fixes -- a flow that has learned to produce concavity will HALLUCINATE it on models 1 and
2, and the voxel measure is a symmetric difference, so an invented concavity is scored
exactly as harshly as a missed one. Model 3 improving is therefore not sufficient
evidence; models 1 and 2 holding is the other half of the test.

So this script does not report a number, it returns a verdict, and exits non-zero when the
verdict is FAIL. Run it once on the current checkpoints to record a baseline, then again
after retraining:

    # 1. baseline, BEFORE changing anything
    python scripts/eval_gate.py --stl results/lpd/Asteroid01.stl \\
                                      results/lpd/Asteroid02.stl \\
                                      results/lpd/Asteroid03.stl \\
           --label baseline --out runs/gate_baseline.json

    # 2. retrain with the new corpus, re-run reconstruct_lpd.py, then
    python scripts/eval_gate.py --stl runs/new/Asteroid01.stl \\
                                      runs/new/Asteroid02.stl \\
                                      runs/new/Asteroid03.stl \\
           --label new_corpus --out runs/gate_new.json \\
           --baseline runs/gate_baseline.json

WHAT IS MEASURED
----------------
1. The challenge voxel measure, via `hac26.scoring.voxel` (Dice; higher is better).
2. The projection boundary distance, via `hac26.scoring.side_view` (ASSD in model units,
   the body spans z in [-1, 1] so 0.01 is 0.5% of its height; LOWER is better).
3. A composite, purely for convenience. The organisers say the side-view measure is
   normalised to [0, 1] but have not released the code, so the mapping used here
   (1 - assd/ASSD_REF, clipped) is OURS, not theirs. The gate never keys on the composite
   -- it keys on the two measured quantities separately -- and the composite is printed
   with that caveat so nobody quotes it as a challenge score.
4. Pose compliance: the body must touch z = +-1 and fit the published bounding cylinder.
   A reconstruction that fails this is not scored, it is rejected, because the organisers
   would reject it too.

The script needs numpy/scipy/trimesh/scikit-image and the ground-truth STLs. It does NOT
need torch, and it does not import any training code, so it can be run on the machine that
holds the dataset without a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.recon import dice, mesh_to_sdf                      # noqa: E402
from hac26.scoring.side_view import side_view_measure, surface_points  # noqa: E402
from hac26.shapes import rescale_touch_z                       # noqa: E402

CYLINDER_R = {1: 1.12, 2: 1.42, 3: 0.88, 4: 1.475, 5: 1.22,
              6: 0.925, 7: 1.205, 8: 1.24, 9: 0.67, 10: 3.95}

TRUTH_REL = {1: "AsteroidModel01_shape_public/asteroid1.stl",
             2: "AsteroidModel02_shape_public/asteroid2.stl",
             3: "AsteroidModel03_shape_public/asteroid3.stl"}

# Scale for the convenience composite only. 0.20 model units = 10% of the body height;
# an ASSD of that size means the outline is wrong by a tenth of the object.
ASSD_REF = 0.20


# ----------------------------------------------------------------------------------
def _load(path):
    import trimesh
    m = trimesh.load(str(path), process=False)
    if hasattr(m, "geometry"):                      # a Scene: concatenate its parts
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def occupancy(v, f, n, extent, verbose=False):
    """Voxel occupancy by z-column ray stabbing: n^2 rays instead of n^3 point queries.

    `hac26.recon.mesh_to_sdf` tests every one of the n^3 grid points with `mesh.contains`
    and then runs two Euclidean distance transforms. At n = 128 against the 800k-face
    public STL that is 2.1M point-in-mesh queries plus two float arrays of 2.1M elements,
    which is what kills a laptop -- and the distance transforms are pure waste here, since
    Dice needs occupancy, not distance.

    Stabbing one ray up each z-column and filling between consecutive entry/exit hits gives
    16384 queries instead of 2.1M. It is exact for a closed surface: verified against
    mesh_to_sdf on model 3 at n = 128, identical voxel count and Dice 1.00000, 2.3s -> 0.03s.
    Columns with an odd number of hits (a ray exactly grazing an edge) are skipped; if more
    than 2% of columns are degenerate the mesh is probably not closed and we fall back.
    """
    import trimesh
    mesh = trimesh.Trimesh(v, f, process=False)
    ax = (np.arange(n) + 0.5) / n * 2.0 * extent - extent
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    org = np.stack([X.ravel(), Y.ravel(), np.full(X.size, -2.0 * extent)], 1)
    dirs = np.zeros_like(org)
    dirs[:, 2] = 1.0
    loc, ray_idx, _ = mesh.ray.intersects_location(org, dirs, multiple_hits=True)
    occ = np.zeros((n, n, n), dtype=bool)
    if not len(loc):
        return occ
    z = loc[:, 2]
    order = np.lexsort((z, ray_idx))
    ray_idx, z = ray_idx[order], z[order]
    bad = 0
    for seg in np.split(np.arange(len(z)), np.flatnonzero(np.diff(ray_idx)) + 1):
        zz = z[seg]
        if len(zz) % 2:
            bad += 1
            continue
        i, j = divmod(int(ray_idx[seg[0]]), n)
        for a, b in zip(zz[0::2], zz[1::2]):
            k0 = int(np.ceil((a + extent) / (2 * extent) * n - 0.5))
            k1 = int(np.floor((b + extent) / (2 * extent) * n - 0.5))
            if k1 >= k0:
                occ[i, j, max(k0, 0):k1 + 1] = True
    if bad > 0.02 * n * n:
        if verbose:
            print(f"      {bad} degenerate columns: mesh not closed, using the slow "
                  f"point-in-mesh path", flush=True)
        return mesh_to_sdf(np.ascontiguousarray(v), np.ascontiguousarray(f), n, extent) < 0
    return occ


def voxel_measure(rv, rf, tv, tf, n=128, verbose=False):
    a, b = rescale_touch_z(rv), rescale_touch_z(tv)
    e = max(float(np.abs(a).max()), float(np.abs(b).max())) * 1.05
    return float(dice(occupancy(a, rf, n, e, verbose), occupancy(b, tf, n, e, verbose)))


def _center_xy(v, f):
    """Put the xy-centroid on the rotation axis, using the VOLUME centroid.

    `rescale_touch_z` centres on the vertex mean, which is not the geometric centre when
    the tessellation is uneven -- on the cube it sits 2% off axis and the cylinder check
    then reads R = 1.52 against the published 1.42 and cries foul on a body that is
    actually compliant. The challenge wording is "centroid on the axis", so use the volume
    centroid where the mesh is closed enough to have one.
    """
    import trimesh
    m = trimesh.Trimesh(v, f, process=False)
    try:
        c = np.asarray(m.centroid, float) if m.is_volume else v.mean(0)
    except Exception:
        c = v.mean(0)
    out = v.copy()
    out[:, 0] -= c[0]
    out[:, 1] -= c[1]
    return out


def pose_check(rv, model):
    """The submission rules: rotation axis = z, touching z = +-1, inside the cylinder.

    Two radii are reported, because they fail for different reasons and only one of them
    is a real shape error. `max_axis_dist` is measured about the origin, which is the rule
    as written; `min_enclosing_R` is the radius the body would need if it were re-centred.
    A body whose min-enclosing radius fits but whose about-origin radius does not is simply
    sitting off the axis -- a translation fixes it, and that is a warning, not a rejection.
    A body whose min-enclosing radius exceeds R is genuinely too wide and IS a rejection.
    Distinguishing them matters: mean-centring alone leaves the cube 8% off, which reads as
    a violation on a body that is compliant.
    """
    from scipy.optimize import minimize
    from scipy.spatial import ConvexHull
    z0, z1 = float(rv[:, 2].min()), float(rv[:, 2].max())
    rmax = float(np.sqrt((rv[:, :2] ** 2).sum(1)).max())
    p = rv[:, :2]
    if len(p) > 3:
        try:
            p = p[ConvexHull(p).vertices]
        except Exception:
            pass
    r_min = float(minimize(lambda c: np.sqrt(((p - c) ** 2).sum(1)).max(), p.mean(0),
                           method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000}).fun)
    R = CYLINDER_R.get(model)
    return {"z_min": z0, "z_max": z1,
            "touches_planes": bool(abs(z0 + 1) < 1e-3 and abs(z1 - 1) < 1e-3),
            "max_axis_dist": rmax, "min_enclosing_R": r_min, "cylinder_R": R,
            "fits_cylinder": bool(R is None or r_min <= R * 1.02),
            "on_axis": bool(R is None or rmax <= R * 1.02)}


def evaluate(stl, model, truth_path, n_vox=128, n_dirs=36, n_pts=200_000, res=512, seed=0,
             verbose=True):
    import time
    t0 = time.time()
    rv, rf = _load(stl)
    tv, tf = _load(truth_path)
    if verbose:
        print(f"      loaded ({len(rf)} / {len(tf)} faces, {time.time()-t0:.0f}s), "
              f"voxelising at {n_vox}^3...", flush=True)
    vox = voxel_measure(rv, rf, tv, tf, n_vox, verbose)
    if verbose:
        print(f"      voxel done ({time.time()-t0:.0f}s), projecting {n_dirs} views...",
              flush=True)
    pa = surface_points(rescale_touch_z(rv), rf, n_pts, seed=seed)
    pb = surface_points(rescale_touch_z(tv), tf, n_pts, seed=seed)
    sv = side_view_measure(pa, pb, n_dirs=n_dirs, res=res)
    side_score = float(np.clip(1.0 - sv["assd_mean"] / ASSD_REF, 0.0, 1.0))
    return {"model": model, "stl": str(stl), "voxel": vox,
            "assd_mean": sv["assd_mean"], "assd_worst": sv["assd_worst"],
            "hausdorff_mean": sv["hausdorff_mean"],
            "side_score_UNOFFICIAL": side_score,
            "composite_UNOFFICIAL": 0.5 * (vox + side_score),
            "pose": pose_check(_center_xy(rescale_touch_z(rv), rf), model)}


# ----------------------------------------------------------------------------------
def verdict(new: dict, base: dict, min_gain: float, tol_vox: float, tol_assd: float,
            target: int, protect: list) -> tuple:
    """Gate: `target` must improve, `protect` must not regress. Returns (ok, lines)."""
    lines, ok = [], True

    if target in new and target in base:
        dv = new[target]["voxel"] - base[target]["voxel"]
        da = new[target]["assd_mean"] - base[target]["assd_mean"]
        good_v = dv >= min_gain
        good_a = da <= 0.0
        ok &= bool(good_v and good_a)
        lines.append(f"  TARGET  model {target}: voxel {base[target]['voxel']:.4f} -> "
                     f"{new[target]['voxel']:.4f}  ({dv:+.4f}, need >= {min_gain:+.4f})"
                     f"  {'OK' if good_v else 'FAIL'}")
        lines.append(f"                   assd  {base[target]['assd_mean']:.4f} -> "
                     f"{new[target]['assd_mean']:.4f}  ({da:+.4f}, need <= 0)"
                     f"  {'OK' if good_a else 'FAIL'}")
    else:
        ok = False
        lines.append(f"  TARGET  model {target}: MISSING from one of the runs")

    for m in protect:
        if m not in new or m not in base:
            continue
        dv = new[m]["voxel"] - base[m]["voxel"]
        da = new[m]["assd_mean"] - base[m]["assd_mean"]
        good_v = dv >= -tol_vox
        good_a = da <= tol_assd
        ok &= bool(good_v and good_a)
        lines.append(f"  PROTECT model {m}: voxel {base[m]['voxel']:.4f} -> "
                     f"{new[m]['voxel']:.4f}  ({dv:+.4f}, tol {-tol_vox:+.4f})"
                     f"  {'OK' if good_v else 'FAIL'}")
        lines.append(f"                   assd  {base[m]['assd_mean']:.4f} -> "
                     f"{new[m]['assd_mean']:.4f}  ({da:+.4f}, tol {tol_assd:+.4f})"
                     f"  {'OK' if good_a else 'FAIL'}")
    return ok, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stl", nargs="+", required=True,
                    help="one reconstruction per model, in --models order")
    ap.add_argument("--models", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--data-dir", default="dataset/raw",
                    help="root holding AsteroidModel0N_shape_public/")
    ap.add_argument("--truth", nargs="+", default=None,
                    help="explicit ground-truth STLs, overrides --data-dir")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--baseline", default=None,
                    help="a previous --out JSON; enables the PASS/FAIL gate")
    ap.add_argument("--target", type=int, default=3, help="model that must improve")
    ap.add_argument("--protect", nargs="+", type=int, default=[1, 2],
                    help="models that must not regress")
    ap.add_argument("--min-gain", type=float, default=0.010,
                    help="required voxel gain on the target model")
    ap.add_argument("--tol-voxel", type=float, default=0.005,
                    help="allowed voxel loss on a protected model")
    ap.add_argument("--tol-assd", type=float, default=0.002,
                    help="allowed outline-distance increase on a protected model")
    ap.add_argument("--n-vox", type=int, default=128)
    ap.add_argument("--n-dirs", type=int, default=36)
    ap.add_argument("--n-pts", type=int, default=200_000)
    ap.add_argument("--fast", action="store_true",
                    help="n_vox 96, n_dirs 24, n_pts 120k, res 384 -- for a "
                         "quick look on a laptop; do NOT mix fast and normal "
                         "runs in one comparison, the numbers are not the same")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--resume", action="store_true",
                    help="reuse models already present in --out")
    a = ap.parse_args()

    if a.fast:
        a.n_vox, a.n_dirs, a.n_pts, a.res = 96, 24, 120_000, 384
    if len(a.stl) != len(a.models):
        raise SystemExit(f"--stl has {len(a.stl)} entries, --models has {len(a.models)}")
    truths = a.truth or [Path(a.data_dir) / TRUTH_REL[m] for m in a.models]
    if len(truths) != len(a.models):
        raise SystemExit("--truth must have one entry per model")
    for t in truths:
        if not Path(t).exists():
            raise SystemExit(f"ground truth not found: {t}\n"
                             f"pass --data-dir or --truth explicitly")

    res = {}
    # resume: a partial --out from a killed run is picked up rather than recomputed, so a
    # machine that dies on model 1 does not cost you models 2 and 3 as well
    if a.out and a.resume and Path(a.out).exists():
        prev = json.loads(Path(a.out).read_text()).get("results", {})
        res = {int(k): v for k, v in prev.items()}
        if res:
            print(f"resuming: {sorted(res)} already in {a.out}", flush=True)

    def _save():
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(
                {"label": a.label, "assd_ref": ASSD_REF,
                 "results": {str(k): v for k, v in res.items()}}, indent=2))

    print(f"[{a.label}]", flush=True)
    for stl, m, t in zip(a.stl, a.models, truths):
        if m in res:
            print(f"  model {m}: cached", flush=True)
            continue
        print(f"  model {m}: {stl}", flush=True)
        r = evaluate(stl, m, t, a.n_vox, a.n_dirs, a.n_pts, a.res)
        res[m] = r
        _save()
        pz = r["pose"]
        if not (pz["touches_planes"] and pz["fits_cylinder"]):
            flag = "   <-- POSE VIOLATION, the organisers would reject this"
        elif not pz["on_axis"]:
            flag = "   <-- off axis (fits once re-centred; translate before submitting)"
        else:
            flag = ""
        print(f"  model {m}: voxel {r['voxel']:.4f}   assd {r['assd_mean']:.4f}   "
              f"worst {r['assd_worst']:.4f}   R {pz['min_enclosing_R']:.3f}"
              f"/{pz['cylinder_R']}{flag}", flush=True)
    print(f"  summed voxel over {len(res)} models: "
          f"{sum(r['voxel'] for r in res.values()):.4f}")
    print(f"  mean composite (UNOFFICIAL normalisation, not a challenge score): "
          f"{np.mean([r['composite_UNOFFICIAL'] for r in res.values()]):.4f}")

    _save()
    if a.out:
        print(f"  wrote {a.out}")

    if not a.baseline:
        print("\n  no --baseline given: reported only, no verdict.")
        return 0

    base_raw = json.loads(Path(a.baseline).read_text())
    base = {int(k): v for k, v in base_raw["results"].items()}
    print(f"\ngate: target model {a.target} must gain >= {a.min_gain:+.4f} voxel and not "
          f"lose outline accuracy;\n      models {a.protect} must not lose more than "
          f"{a.tol_voxel:.4f} voxel / gain {a.tol_assd:.4f} assd")
    print(f"      baseline = {base_raw.get('label', a.baseline)}")
    ok, lines = verdict(res, base, a.min_gain, a.tol_voxel, a.tol_assd,
                        a.target, a.protect)
    print("\n".join(lines))
    bad_pose = [m for m, r in res.items()
                if not (r["pose"]["touches_planes"] and r["pose"]["fits_cylinder"])]
    if bad_pose:
        ok = False
        print(f"  POSE    models {bad_pose} violate the submission pose rules")
    print("\n  GATE " + ("PASS -- keep the change" if ok else
                         "FAIL -- do not ship this corpus/checkpoint"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

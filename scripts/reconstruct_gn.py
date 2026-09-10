#!/usr/bin/env python3
"""Recover one body from its curves by fitting the hull's reshaping and the carve together.

    python scripts/reconstruct_gn.py --model 3 --channel blender --hold-out-geoms 5 \
        --out results/gn/Asteroid03.stl

The convex stage's answer supplies the base support h. It is not the body's hull: a convex
inversion of a non-convex body returns the convex body whose own shadowing best imitates the
concavities, which is larger, and on the one non-convex public model it has 1.37 times the
volume of the true hull. The fit therefore does not treat h as fixed and does not treat the
hull as something to be corrected before carving. It moves nine coefficients that reshape h
and the lattice amplitudes that carve it in one step, because each half raises the misfit or
barely lowers it on its own while the two together lower it by more than half.

The step is damped Gauss-Newton with a secant Jacobian, coarse to fine over the amplitudes
(hac26.solvers.gauss_newton). Nothing differentiates the renderer. The fit is started several
times, once from the convex answer itself and otherwise from a single deep waist, because the
first linearisation from an uncarved body is taken where a carve's effect on the curves has
not yet begun.

--hold-out-geoms keeps cameras out of the fit and reports the written body's misfit on them
beside the convex answer's on the same cameras. That pair is what scripts/select_answers.py
reads, and it is the only test of whether a shape was recovered rather than curves fitted.
On a public model the overlap with the released shape is reported as well.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.conventions import CYLINDER_R, psi_grid                       # noqa: E402
from hac26.data_io import N_CAMS, load_inversion_curves                  # noqa: E402
from hac26.field import (CODE_DIM, EXTRACT_RES, LATTICE_SHAPE, N_RADIAL,  # noqa: E402
                         N_SITES, lattice_kernel)
from hac26.recon import fit_to_cylinder                                  # noqa: E402
from hac26.solvers.gauss_newton import CarveFit, Stage, waist_amplitudes  # noqa: E402
from hac26.solvers.operator import CodeOperator                          # noqa: E402
from hac26.solvers.output import export_stl, restore_constraints         # noqa: E402
from reconstruct import answer_path                                      # noqa: E402
from reconstruct_lpd import (curve_pairs, curve_weight, geometry_mask,   # noqa: E402
                             residual_scale, support_from_convex)
from calibrate import ETA_FLOOR                                          # noqa: E402
from reconstruct_map import EXPORT_RES, convexity, truth_dice            # noqa: E402
from train_lpd import INSTRUMENT, RENDER, _enable_tf32, load_instrument   # noqa: E402

STAGE_SIDES = (6, 12)      # sub-lattices the carve is fitted on before the random subspace
SUBSPACE_DIRS = 192        # directions per iteration of the last stage
STAGE_ITERS = (6, 6, 10)
RESTARTS = 8               # starts of the first stage
RESTART_SCREEN = 2         # iterations every start is judged on before all but the best stop
RESTART_KEEP = 2           # starts carried to the end of the first stage
TARGET_SIGMA = 1.0         # a body that explains the curves to the model error is fitted


def curve_index(weight: torch.Tensor, geoms) -> tuple:
    """(geometries, per-geometry curve mask) of the curves a fit uses, and the flat index of
    those curves in the operator's (G, 2, P) output."""
    g = [int(i) for i in geoms]
    w = weight[g] > 0
    return g, w


def flat_curves(cur: torch.Tensor, keep: torch.Tensor) -> np.ndarray:
    """The kept curves of an operator result (G, 2, P), flattened in a fixed order."""
    return cur.detach().cpu().numpy()[keep.numpy()].ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--channel", choices=("real", "blender"), default="blender")
    ap.add_argument("--calibration", default=None,
                    help=f"instrument file; by channel, {INSTRUMENT}")
    ap.add_argument("--support-from", default=None,
                    help="STL whose support is the base; by default the convex answer")
    ap.add_argument("--phases", type=int, default=48)
    ap.add_argument("--export-phases", type=int, default=96,
                    help="phases the written body's misfits are measured at")
    ap.add_argument("--operator-res", type=int, default=EXTRACT_RES,
                    help="extraction resolution of the fit's operator; it has to resolve "
                         "the correction's kernels")
    ap.add_argument("--export-res", type=int, default=EXPORT_RES)
    ap.add_argument("--hold-out-geoms", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    ap.add_argument("--ridge", type=float, default=1e-2,
                    help="ridge on the amplitudes, as a fraction of the mean curvature of "
                         "the misfit in the carve coordinates")
    ap.add_argument("--step-g", type=float, default=0.20,
                    help="secant step of a carve coordinate, in body units of depth")
    ap.add_argument("--step-c", type=float, default=0.03,
                    help="secant step of a reshaping coefficient, in body units")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    _enable_tf32()
    dev = a.device if torch.cuda.is_available() else "cpu"
    R = CYLINDER_R[a.model]
    inst = load_instrument(a.calibration or INSTRUMENT[a.channel], device=dev)
    op = CodeOperator(inst, psi_grid(a.phases), res=a.operator_res, config=RENDER, device=dev)

    sup_stl = a.support_from or str(answer_path(a.model))
    support = support_from_convex(sup_stl)

    d = load_inversion_curves(a.data_dir, a.model, m=a.phases, channel=a.channel)
    if set(d["files"]) != {"intensity", "binary"}:
        raise SystemExit(f"model {a.model} needs both {a.channel} curve files under "
                         f"{a.data_dir}; found {sorted(d['files'])}")
    data = curve_pairs(d["curves"])                                  # (N_CAMS, 2, P)
    weight = curve_weight(d["mask"])                                 # (N_CAMS, 2)
    scale = residual_scale(d, inst.eta).clamp_min(ETA_FLOOR)         # (N_CAMS, 2)
    present = [i for i in range(N_CAMS) if float(geometry_mask(d["mask"])[0, i]) > 0]
    rng = np.random.default_rng(a.seed)
    held = [present[i] for i in np.unique(np.linspace(0, len(present) - 1, a.hold_out_geoms)
                                          .round().astype(int))] if a.hold_out_geoms else []
    fit_geoms = [g for g in present if g not in held]
    print(f"model {a.model}  R={R}  channel {d['channel']}  h from {Path(sup_stl).name}\n"
          f"  {int(weight.sum())} curves of {len(present)} geometries; fitting on "
          f"{len(fit_geoms)}" + (f", holding out {held}" if held else "")
          + f"; model error median {float(scale.median()):.4f}", flush=True)
    if d["duplicate_columns"] or d["count_curves_refused"]:
        print(f"  {len(d['duplicate_columns'])} repeated columns and "
              f"{len(d['count_curves_refused'])} count curves are not measurements and were "
              f"dropped", flush=True)

    fit_g, keep_fit = curve_index(weight, fit_geoms)
    held_g, keep_held = curve_index(weight, held) if held else ([], None)
    data_fit = flat_curves(data[fit_g], keep_fit)
    scale_fit = np.repeat(scale[fit_g].numpy()[keep_fit.numpy()], data.shape[-1])
    zero_code = torch.zeros(CODE_DIM, device=dev)

    def render(c, g):
        code = zero_code.clone()
        code[-N_SITES:] = torch.tensor(np.asarray(g), dtype=torch.float32, device=dev)
        cur = op.curves(support, code, R, geoms=fit_g,
                        c=torch.tensor(np.asarray(c), dtype=torch.float32, device=dev))
        return None if cur is None else flat_curves(cur, keep_fit)

    kernel = lattice_kernel()
    stages = tuple(Stage(side=s, n_dirs=0, iters=n)
                   for s, n in zip(STAGE_SIDES, STAGE_ITERS)) \
        + (Stage(side=0, n_dirs=SUBSPACE_DIRS, iters=STAGE_ITERS[-1]),)

    def new_fit(seed):
        return CarveFit(render, data_fit, scale_fit, kernel, LATTICE_SHAPE,
                        n_radial=N_RADIAL, ridge_frac=a.ridge, step_g=a.step_g,
                        step_c=a.step_c, seed=seed)

    # Every start is judged on the first few iterations of the coarse stage, which tells a
    # start that is descending from one that is not; the best are then carried to the end.
    t0 = time.time()
    starts, recipes = [(np.zeros(N_RADIAL), np.zeros(N_SITES))], [{"start": "convex answer"}]
    sites = kernel.shape[0]
    site_xyz = None
    if a.restarts > 1:
        from hac26.field import GaussianLattice
        site_xyz = GaussianLattice().p.numpy()
        for _ in range(a.restarts - 1):
            g0, rec = waist_amplitudes(site_xyz, kernel, rng)
            starts.append((np.zeros(N_RADIAL), g0))
            recipes.append({"start": "waist", **rec})
    screened = []
    for i, (c0, g0) in enumerate(starts):
        f = new_fit(a.seed + i)
        c, g, hist = f.run(c0, g0, stages=(Stage(STAGE_SIDES[0], 0, RESTART_SCREEN),),
                           target=TARGET_SIGMA)
        chi = hist[-1]["chi"] if hist else float("inf")
        screened.append({"i": i, "chi": chi, "c": c, "g": g, "renders": f.renders})
        print(f"  start {i} ({recipes[i]['start']}): chi {chi:.4f} after {RESTART_SCREEN} "
              f"coarse steps, {f.renders} renders [{time.time()-t0:.0f}s]", flush=True)
    screened.sort(key=lambda s: s["chi"])

    best = None
    for s in screened[:max(1, RESTART_KEEP)]:
        f = new_fit(a.seed + s["i"] + 100)
        c, g, hist = f.run(s["c"], s["g"], stages=stages, target=TARGET_SIGMA,
                           log=lambda row: print(f"    {row['stage']:>14}  it {row['iteration']}"
                                                 f"  chi {row['chi']:.4f}"
                                                 f"  {'step' if row['accepted'] else 'no step'}"
                                                 f"  [{time.time()-t0:.0f}s]", flush=True))
        chi = hist[-1]["chi"] if hist else float("inf")
        print(f"  start {s['i']} finished at chi {chi:.4f} "
              f"({f.renders + s['renders']} renders)", flush=True)
        if best is None or chi < best["chi"]:
            best = {"chi": chi, "c": c, "g": g, "start": s["i"], "history": hist,
                    "renders": f.renders + s["renders"], "recipe": recipes[s["i"]]}

    if not a.out:
        return

    # the body as it would be submitted, and its misfits, measured at the export resolution
    # and the export phase count, which is what a scored body is
    op_x = CodeOperator(inst, psi_grid(a.export_phases), res=a.export_res, config=RENDER,
                        device=dev)
    d_x = load_inversion_curves(a.data_dir, a.model, m=a.export_phases, channel=a.channel)
    data_x = curve_pairs(d_x["curves"])
    scale_x = residual_scale(d_x, inst.eta).clamp_min(ETA_FLOOR)

    def misfit_at_export(c, g, geoms):
        if not geoms:
            return float("nan")
        gg, keep = curve_index(weight, geoms)
        code = zero_code.clone()
        code[-N_SITES:] = torch.tensor(np.asarray(g), dtype=torch.float32, device=dev)
        cur = op_x.curves(support, code, R, geoms=gg,
                          c=torch.tensor(np.asarray(c), dtype=torch.float32, device=dev))
        if cur is None:
            return float("inf")
        pred = flat_curves(cur, keep)
        obs = flat_curves(data_x[gg], keep)
        sc = np.repeat(scale_x[gg].numpy()[keep.numpy()], data_x.shape[-1])
        return float(np.sqrt(np.mean(((pred - obs) / sc) ** 2)))

    zeros = (np.zeros(N_RADIAL), np.zeros(N_SITES))
    convex_fit = misfit_at_export(*zeros, fit_geoms)
    convex_held = misfit_at_export(*zeros, held)
    fit_x = misfit_at_export(best["c"], best["g"], fit_geoms)
    held_x = misfit_at_export(best["c"], best["g"], held)

    code = zero_code.clone()
    code[-N_SITES:] = torch.tensor(best["g"], dtype=torch.float32, device=dev)
    m = op_x.mesh(support, code, res=a.export_res,
                  c=torch.tensor(best["c"], dtype=torch.float32, device=dev))
    if m is None:
        raise SystemExit("the fitted body is degenerate; nothing written")
    v = fit_to_cylinder(restore_constraints(
        CodeOperator.canonical(m[0], m[1]).cpu().numpy(), R), R)
    f = m[1].cpu().numpy()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rep = export_stl(a.out, v, f)
    meta = {"model": a.model, "channel": d["channel"], "radius": R, "phases": a.phases,
            "export_phases": a.export_phases, "operator_res": a.operator_res,
            "export_res": a.export_res, "held_out": held, "fit_geoms": fit_g,
            "curves_fitted": int(weight.sum()), "ridge": a.ridge,
            "step_g": a.step_g, "step_c": a.step_c, "restarts": a.restarts,
            "start": best["recipe"], "renders": best["renders"],
            "chi_fit": best["chi"], "history": best["history"], "export": rep,
            "chi_fit_convex": convex_fit, "chi_held_convex": convex_held,
            "chi_fit_export": fit_x, "chi_held_export": held_x,
            "reshaping": best["c"].tolist(),
            "carve_depth_max": float(np.abs(kernel @ best["g"]).max()),
            "final_dice": truth_dice(v, f, a.model, a.data_dir),
            "final_convexity": convexity(v, f)}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    np.savez(Path(a.out).with_suffix(".fit.npz"), c=best["c"], g=best["g"],
             support=support.cpu().numpy())
    print(f"  wrote {a.out} ({rep['faces']} faces, volume {rep['volume']:.3f}); at export "
          f"resolution chi_fit {fit_x:.3f} against the convex answer's {convex_fit:.3f}"
          + (f", chi_held {held_x:.3f} against {convex_held:.3f}" if held else "")
          + f"; dice {meta['final_dice']:.4f}, convexity {meta['final_convexity']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recover one body from its curves by fitting the hull's reshaping and the carve together.

    python scripts/reconstruct_gn.py --model 3 --channel blender --hold-out-geoms 5 \
        --out results/gn/Asteroid03.stl

The convex stage's answer supplies the base support h. It is not the body's hull: a convex
inversion of a non-convex body returns the convex body whose own shadowing best imitates the
concavities, which is larger. Of the three public bodies the answer exceeds the body's own
hull on the one that has concavities and falls short of it on the two that do not, which is
the mechanism and not a bias of the network. The fit therefore does not treat h as fixed and
does not treat the hull as something to be corrected before carving. The correction is one
function on the sphere -- a displacement of the core's surface, inward where the body has a
concavity and outward where the convex answer overshot -- and the reshaping of the hull is
simply its first three degrees.

The step is damped Gauss-Newton with a secant Jacobian, coarse to fine in the angular degree of
that function (hac26.solvers.gauss_newton). Nothing differentiates the renderer. The ladder
stops well short of the scale at which a displacement stops costing surface area, because there
the penalty below cannot charge it and the fit would spend its whole budget buying misfit with
texture. The fit is started several times, once from the convex answer itself and otherwise
from a shrunken hull carved by a single spherical cap drawn from a grid over where the cap is,
how wide it is and how deep, because the first linearisation from the convex answer is taken
where neither half of the correction is yet doing anything and the step there goes into the
carve alone, which is the convex inversion's own mistake made once more.

What is minimised carries the body's surface area beside its misfit, because the misfit of a
rendered body reports how finely its surface is resolved almost as strongly as it reports
whether the shape is right. The run is therefore two phases: to convergence under the
penalised objective, then a polish on the misfit alone with the volume still held, which
recovers the misfit without giving the shape back.

A body whose convex answer already explains its curves is left alone. There is no concavity
there for the correction to find, and an objective that charges surface will trade overlap for
a misfit it does not need; that failure raises the misfit's opinion of the body while lowering
its overlap, so it cannot be caught afterwards and is refused before the fit instead.

--hold-out-geoms keeps cameras out of the fit and reports the written body's misfit and
objective on them beside the convex answer's on the same cameras. That pair is what
scripts/select_answers.py reads, and it is the only test of whether a shape was recovered
rather than curves fitted. On a public model the overlap with the released shape is reported
as well.
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
from hac26.field import (CODE_DIM, EXTRACT_RES, N_NODES, N_RADIAL,       # noqa: E402
                         DepthSphere, depth_cap, node_kernel)
from hac26.recon import fit_to_cylinder                                  # noqa: E402
from hac26.solvers.gauss_newton import (AREA_WEIGHT, AREA_WINDOW,    # noqa: E402
                                        DEFAULT_STAGES, N_STARTS, POLISH_STAGES,
                                        SCREEN_STAGES, STEP_C, STEP_G, TARGET_SIGMA,
                                        VOLUME_TRUST, CarveFit, conjunction_start)
from hac26.solvers.operator import CodeOperator                          # noqa: E402
from hac26.solvers.output import export_stl, restore_constraints         # noqa: E402
from reconstruct import answer_path                                      # noqa: E402
from reconstruct_lpd import (curve_pairs, curve_weight, geometry_mask,   # noqa: E402
                             json_default,
                             residual_scale, support_from_convex)
from calibrate import ETA_FLOOR                                          # noqa: E402
from reconstruct_map import (EXPORT_RES, convex_dice, convexity,     # noqa: E402
                             truth_dice)
from train_lpd import INSTRUMENT, RENDER, _enable_tf32, load_instrument   # noqa: E402

RESTARTS = 9               # starts screened, the convex answer and eight of the designed
                           # grid: enough for one sweep of the grid's axes, which is the factor
                           # the ladder's own first stage cannot correct cheaply. Measured, a
                           # screening costs about fifty renders against the ladder's four
                           # thousand, so the sweep is affordable; measured also, the random
                           # draws this grid replaces were worth nothing at all, so it is the
                           # spread and not the count that has to earn its place.
RESTART_KEEP = 2           # starts carried to the end of the ladder
VOLUME_FLOOR = 0.50        # smallest volume an accepted body may have, as a fraction of the
                           # convex answer's. The cheapest surface area in this representation
                           # is a hull shrink and the misfit barely resists one, so without a
                           # floor the fit walks the volume down past the body and loses more
                           # overlap than the carve gains; notes/objective.md measures both
                           # runs. The value is calibrated on the one released non-convex body,
                           # which sits comfortably above it, and it inherits exactly the
                           # weakness --min-convex-sigmas has: what would make it principled is
                           # the distribution of that ratio over the shape library, which is the
                           # same corpus a learned acceptance gate would need.
MIN_CONVEX_SIGMAS = 4.0    # how badly the convex answer must fit before a body is worth
                           # correcting, in model errors. A body whose convex answer already
                           # explains its curves has no concavity for the correction to find,
                           # and an objective that charges surface will then trade overlap it
                           # cannot regain for a misfit it does not need. Measured on the most
                           # nearly convex public body, that costs a sixth of the overlap while
                           # improving the misfit, so no gate that reads a misfit catches it;
                           # this one is read before the fit instead.


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
                    help="extraction resolution of the fit's operator; it has to resolve the "
                         "angular scale of the depth field")
    ap.add_argument("--export-res", type=int, default=EXPORT_RES)
    ap.add_argument("--hold-out-geoms", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    ap.add_argument("--area-weight", type=float, default=AREA_WEIGHT,
                    help="weight of the posed body's surface area in the objective, in "
                         f"inverse area of the canonical pose; measured window {AREA_WINDOW}, "
                         "and zero minimises the misfit alone")
    ap.add_argument("--volume-trust", type=float, default=VOLUME_TRUST,
                    help="largest fractional change of volume an accepted step may make")
    ap.add_argument("--volume-floor", type=float, default=VOLUME_FLOOR,
                    help="smallest volume an accepted body may have, as a fraction of the "
                         "convex answer's; 0 turns the floor off")
    ap.add_argument("--min-convex-sigmas", type=float, default=MIN_CONVEX_SIGMAS,
                    help="leave a body alone whose convex answer already explains its curves "
                         "to fewer than this many model errors")
    ap.add_argument("--step-g", type=float, default=STEP_G,
                    help="secant step of a carve coordinate, in body units of depth")
    ap.add_argument("--step-c", type=float, default=STEP_C,
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
    floor = [0.0]           # set from the convex answer's own volume, once it is rendered

    def render(c, g):
        code = zero_code.clone()
        code[-N_NODES:] = torch.tensor(np.asarray(g), dtype=torch.float32, device=dev)
        out = op.curves_with_shape(support, code, R, geoms=fit_g,
                                   c=torch.tensor(np.asarray(c), dtype=torch.float32,
                                                  device=dev))
        if out is None:
            return None
        cur, area, vol = out
        # A body below the floor is refused here rather than in the solver, because the solver
        # already has a path for a body the forward model will not render and this is the same
        # kind of refusal: the line search sees a trial that did not come back and shortens.
        if vol < floor[0]:
            return None
        return flat_curves(cur, keep_fit), area, vol

    nodes = DepthSphere(N_NODES).u.numpy()
    kernel = node_kernel()
    cap = depth_cap(support.numpy())

    def new_fit(seed):
        return CarveFit(render, data_fit, scale_fit, kernel, nodes,
                        n_radial=N_RADIAL, area_weight=a.area_weight,
                        volume_trust=a.volume_trust, depth_cap=cap,
                        step_g=a.step_g, step_c=a.step_c, seed=seed)

    zeros = (np.zeros(N_RADIAL), np.zeros(N_NODES))
    gate = new_fit(a.seed)
    r0, area0, vol0 = gate._render(*zeros)
    if r0 is None:
        raise SystemExit("the convex answer does not render; nothing to correct")
    floor[0] = float(a.volume_floor) * vol0
    convex_sigmas = float(np.linalg.norm(r0))
    print(f"  the depth may reach {cap:.3f} body units before the body stops containing its "
          f"own centre; below {floor[0]:.3f} of volume a body is refused", flush=True)
    print(f"  the convex answer explains the fitted curves to {convex_sigmas:.2f} model "
          f"errors; area {area0:.3f}, volume {vol0:.3f}", flush=True)
    if convex_sigmas < a.min_convex_sigmas:
        raise SystemExit(
            f"model {a.model}: the convex answer already fits to {convex_sigmas:.2f} model "
            f"errors, under the {a.min_convex_sigmas:g} this correction is worth running at. "
            f"A body with no concavity to find loses overlap to an objective that charges "
            f"surface, so it is left alone and its convex answer stands.")

    # Every start is judged on the first few iterations of the coarse stage, which tells a
    # start that is descending from one that is not; the best are then carried to the end.
    t0 = time.time()
    starts, recipes = [(np.zeros(N_RADIAL), np.zeros(N_NODES))], [{"start": "convex answer"}]
    skipped = 0
    i = 0
    while len(starts) < a.restarts and i < N_STARTS:
        c0, g0, rec = conjunction_start(nodes, i, n_radial=N_RADIAL)
        i += 1
        # A cap of a body that has little room to carve is not a body, so it is passed over
        # rather than shortened: a start clipped to the cap is a different start from the one
        # the grid means, and the grid would then no longer be a spread.
        if float(g0.max()) > cap:
            skipped += 1
            continue
        starts.append((c0, g0))
        recipes.append({"start": "cap", **rec})
    if skipped:
        print(f"  {skipped} of the designed starts carve deeper than this body's own centre "
              f"allows and were passed over", flush=True)
    def last(hist, key):
        rows = [h for h in hist if key in h]
        return rows[-1][key] if rows else float("inf")

    def show(row):
        print(f"    {row['stage']:>14}  it {row['iteration']}  chi {row['chi']:.4f}"
              f"  area {row['area']:.3f}  volume {row['volume']:.3f}"
              f"  {'step' if row['accepted'] else 'no step'}  [{time.time()-t0:.0f}s]",
              flush=True)

    screened = []
    for i, (c0, g0) in enumerate(starts):
        f = new_fit(a.seed + i)
        c, g, hist = f.run(c0, g0, stages=SCREEN_STAGES, target=TARGET_SIGMA)
        screened.append({"i": i, "objective": last(hist, "objective"),
                         "chi": last(hist, "chi"), "c": c, "g": g, "renders": f.renders})
        print(f"  start {i} ({recipes[i]['start']}): objective "
              f"{screened[-1]['objective']:.4f}, chi {screened[-1]['chi']:.4f} after "
              f"{SCREEN_STAGES[0].iters} coarse steps, {f.renders} renders "
              f"[{time.time()-t0:.0f}s]", flush=True)
    # Starts are compared on what is being minimised. A start that has bought misfit with
    # surface is not ahead of one that has not.
    screened.sort(key=lambda s: s["objective"])

    best = None
    for s in screened[:max(1, RESTART_KEEP)]:
        f = new_fit(a.seed + s["i"] + 100)
        c, g, hist = f.run(s["c"], s["g"], stages=DEFAULT_STAGES, target=TARGET_SIGMA,
                           log=show)
        # The penalty has put the shape where it goes and left the misfit above where the
        # data alone would put it. Minimising the misfit alone from there, with the trust
        # region still holding the volume, recovers the misfit without giving the shape back.
        print("    polish, on the misfit alone", flush=True)
        c, g, polish = f.run(c, g, stages=POLISH_STAGES, target=TARGET_SIGMA,
                             area_weight=0.0, log=show)
        hist = hist + polish
        chi, area = last(hist, "chi"), last(hist, "area")
        # Two finished starts are compared under the penalty, not under the misfit the polish
        # was run on. The polish is a refinement inside a start and is minimising the misfit
        # alone by design; between two bodies, the misfit alone prefers the rougher one, which
        # is the comparison notes/objective.md says may never be made and the one
        # select_answers.py is careful to avoid. They are compared on the functional the shape
        # was fitted under, which is also the one the written body is judged by downstream.
        obj = float(np.log(max(chi ** 2, 1e-300)) + a.area_weight * area)
        print(f"  start {s['i']} finished at chi {chi:.4f}, area {area:.3f}, objective "
              f"{obj:.4f} ({f.renders + s['renders']} renders)", flush=True)
        if best is None or obj < best["objective"]:
            best = {"objective": obj, "chi": chi, "c": c, "g": g, "start": s["i"],
                    "history": hist, "renders": f.renders + s["renders"],
                    "recipe": recipes[s["i"]], "refused_depth": f.refused_depth}

    if not a.out:
        return

    # the body as it would be submitted, and its misfits, measured at the export resolution
    # and the export phase count, which is what a scored body is
    op_x = CodeOperator(inst, psi_grid(a.export_phases), res=a.export_res, config=RENDER,
                        device=dev)
    d_x = load_inversion_curves(a.data_dir, a.model, m=a.export_phases, channel=a.channel)
    data_x = curve_pairs(d_x["curves"])
    scale_x = residual_scale(d_x, inst.eta).clamp_min(ETA_FLOOR)

    def measure_at_export(c, g, geoms):
        """(misfit, objective) on those cameras, at the resolution and phase count a written
        body is scored at.

        The objective goes beside the misfit because it is what was minimised. A gate that
        reads the misfit alone prefers a corrugated body to a shaped one, which is the thing
        the penalty exists to stop, so selecting on a different functional from the one that
        was minimised undoes the fit."""
        if not geoms:
            return float("nan"), float("nan")
        gg, keep = curve_index(weight, geoms)
        code = zero_code.clone()
        code[-N_NODES:] = torch.tensor(np.asarray(g), dtype=torch.float32, device=dev)
        out = op_x.curves_with_shape(support, code, R, geoms=gg,
                                     c=torch.tensor(np.asarray(c), dtype=torch.float32,
                                                    device=dev))
        if out is None:
            return float("inf"), float("inf")
        cur, area, _ = out
        pred = flat_curves(cur, keep)
        obs = flat_curves(data_x[gg], keep)
        sc = np.repeat(scale_x[gg].numpy()[keep.numpy()], data_x.shape[-1])
        chi2 = float(np.mean(((pred - obs) / sc) ** 2))
        return float(np.sqrt(chi2)), float(np.log(max(chi2, 1e-300)) + a.area_weight * area)

    convex_fit, convex_fit_obj = measure_at_export(*zeros, fit_geoms)
    convex_held, convex_held_obj = measure_at_export(*zeros, held)
    fit_x, fit_x_obj = measure_at_export(best["c"], best["g"], fit_geoms)
    held_x, held_x_obj = measure_at_export(best["c"], best["g"], held)

    code = zero_code.clone()
    code[-N_NODES:] = torch.tensor(best["g"], dtype=torch.float32, device=dev)
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
            "curves_fitted": int(weight.sum()), "area_weight": a.area_weight,
            "volume_trust": a.volume_trust, "volume_floor": a.volume_floor,
            "convex_volume": vol0, "depth_cap": cap, "convex_sigmas": convex_sigmas,
            "step_g": a.step_g, "step_c": a.step_c, "restarts": a.restarts,
            "start": best["recipe"], "renders": best["renders"],
            "chi_fit": best["chi"], "objective_fit": best["objective"],
            "history": best["history"], "export": rep,
            "chi_fit_convex": convex_fit, "chi_held_convex": convex_held,
            "chi_fit_export": fit_x, "chi_held_export": held_x,
            "objective_fit_convex": convex_fit_obj, "objective_held_convex": convex_held_obj,
            "objective_fit_export": fit_x_obj, "objective_held_export": held_x_obj,
            "reshaping": best["c"].tolist(),
            "carve_depth_max": float(np.abs(kernel @ best["g"]).max()),
            # how often the star-shaped bound refused a trial. A run that never hits it is not
            # held back by it; one that hits it constantly is a body the correction wants to
            # carve past its own centre, and that is worth seeing rather than inferring.
            "depth_refusals": best["refused_depth"],
            "final_dice": truth_dice(v, f, a.model, a.data_dir),
            "convex_dice": convex_dice(sup_stl, a.model, a.data_dir),
            "final_convexity": convexity(v, f)}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2,
                                                           default=json_default))
    np.savez(Path(a.out).with_suffix(".fit.npz"), c=best["c"], g=best["g"],
             support=support.cpu().numpy())
    print(f"  wrote {a.out} ({rep['faces']} faces, volume {rep['volume']:.3f}); at export "
          f"resolution chi_fit {fit_x:.3f} against the convex answer's {convex_fit:.3f}"
          + (f", chi_held {held_x:.3f} against {convex_held:.3f}" if held else "")
          + f"; dice {meta['final_dice']:.4f}, convexity {meta['final_convexity']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Refine the convex answer of one model by descending the exact misfit of one channel.

    python scripts/reconstruct_map.py --model 3 --channel blender --hold-out-geoms 6 \
        --out results/map/Asteroid03.stl

No flow, no training, no corpus. The convex stage's answer is the base support h; the code,
the band-limited correction dh of that support and the lattice amplitudes g that carve, starts
at zero, which is the convex body, and moves by gradient descent on the whitened misfit
through the exact forward model, with a ridge on the amplitudes as the only prior.

A lightcurve misfit is not the score, and a body can fit the curves better while resembling
the truth less, so the run measures both. --hold-out-geoms keeps cameras out of the fit and
reports the misfit on them, which is the difference between recovering a shape and fitting
curves; on a public model the Dice against the released shape is printed at every checkpoint.
The written body is extracted at EXPORT_RES, finer than the grid the descent ran on, and its
own misfits are recorded beside the descent's, since it is the body that would be submitted.
scripts/select_answers.py reads those numbers and decides, per model, whether this body
replaces the convex answer.

The step is a backtracking line search along the sign-normalised gradient, per block. The
objective's curvature varies over orders of magnitude across the coordinates, and dh and g
are in different units, so a fixed step or a single scale lets one block set the step for
both; halving until the objective falls, and growing the step when it does, needs only the
direction.
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

from hac26.conventions import CYLINDER_R, PUBLIC_MODELS, psi_grid        # noqa: E402
from hac26.data_io import N_CAMS, load_inversion_curves, public_stl      # noqa: E402
from hac26.field import CODE_DIM, N_DIR                                  # noqa: E402
from hac26.recon import dice, fit_to_cylinder, mesh_occupancy            # noqa: E402
from hac26.shapes import rescale_touch_z                                 # noqa: E402
from hac26.solvers.operator import CodeOperator                          # noqa: E402
from hac26.solvers.output import export_stl, restore_constraints         # noqa: E402
from reconstruct import answer_path                                      # noqa: E402
from reconstruct_lpd import (curve_pairs, curve_weight, geometry_mask,   # noqa: E402
                             residual_scale, support_from_convex, whitened_misfit)
from train_lpd import INSTRUMENT, RENDER, _enable_tf32, load_instrument   # noqa: E402

OCC_RES = 128         # grid of the Dice reported at the checkpoints
EXPORT_RES = 64       # extraction resolution of the written mesh
TARGET_SIGMA = 1.0    # stop once the answer explains the data to the noise level
MAX_HALVINGS = 8      # trial steps per iteration before declaring convergence
STEP_GROW = 1.6       # a step that works makes the next trial bolder


def truth_dice(verts, faces, model: int, data_dir: str) -> float:
    """Dice of a posed reconstruction against the released truth, or nan when there is none."""
    if model not in PUBLIC_MODELS or not Path(public_stl(data_dir, model)).exists():
        return float("nan")
    import trimesh
    t = trimesh.load(public_stl(data_dir, model), process=False)
    tf = np.asarray(t.faces)
    tv = rescale_touch_z(np.asarray(t.vertices), tf, centre_xy=False)
    rv = rescale_touch_z(np.asarray(verts), np.asarray(faces), centre_xy=False)
    e = max(float(np.abs(rv).max()), float(np.abs(tv).max())) * 1.05
    return float(dice(mesh_occupancy(rv, np.asarray(faces), OCC_RES, e),
                      mesh_occupancy(tv, tf, OCC_RES, e)))


def convexity(verts, faces) -> float:
    """Volume over convex-hull volume; nan for a mesh whose hull cannot be built."""
    import trimesh
    from scipy.spatial import ConvexHull
    m = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=True)
    try:
        return float(abs(m.volume) / ConvexHull(np.asarray(m.vertices)).volume)
    except Exception:                                    # noqa: BLE001
        return float("nan")


def posed_mesh(op: CodeOperator, support, code, radius: float, res: int):
    """(verts, faces) of the code's body in the challenge pose at the physical radius, as
    numpy arrays, or None when the body is degenerate."""
    m = op.mesh(support, code, res=res)
    if m is None:
        return None
    v = fit_to_cylinder(restore_constraints(
        CodeOperator.canonical(m[0], m[1]).cpu().numpy(), radius), radius)
    return v, m[1].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--channel", choices=("real", "blender"), default="blender",
                    help="the released curves to fit and, with it, the instrument fitted to "
                         "them")
    ap.add_argument("--calibration", default=None,
                    help=f"instrument file; by channel, {INSTRUMENT}")
    ap.add_argument("--support-from", default=None,
                    help="STL whose support is the base; by default the convex answer under "
                         "results/")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05,
                    help="first trial step, in RMS code units per coordinate; the line search "
                         "adapts it, so it is a starting scale and not a schedule")
    ap.add_argument("--l2", type=float, default=3e-3,
                    help="ridge on the lattice amplitudes, the only prior: many amplitudes "
                         "against few curves is not obviously determined, and the ridge "
                         "stops the fit spending them on noise")
    ap.add_argument("--phases", type=int, default=96)
    ap.add_argument("--operator-res", type=int, default=32,
                    help="extraction resolution of the descent's operator")
    ap.add_argument("--hold-out-geoms", type=int, default=0,
                    help="cameras kept out of the fit; their misfit is the honest test")
    ap.add_argument("--every", type=int, default=25, help="steps between diagnostics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    _enable_tf32()
    dev = a.device if torch.cuda.is_available() else "cpu"
    R = CYLINDER_R[a.model]
    inst = load_instrument(a.calibration or INSTRUMENT[a.channel], device=dev)
    psi = psi_grid(a.phases)
    op = CodeOperator(inst, psi, res=a.operator_res, config=RENDER, device=dev)
    op_export = CodeOperator(inst, psi, res=EXPORT_RES, config=RENDER, device=dev)

    sup_stl = a.support_from or str(answer_path(a.model))
    support = support_from_convex(sup_stl)

    d = load_inversion_curves(a.data_dir, a.model, m=a.phases, channel=a.channel)
    if set(d["files"]) != {"intensity", "binary"}:
        raise SystemExit(f"model {a.model} needs both {a.channel} curve files under "
                         f"{a.data_dir}; found {sorted(d['files'])}")
    data = curve_pairs(d["curves"])
    weight = curve_weight(d["mask"])
    if d["duplicate_columns"] or d["count_curves_refused"]:
        print(f"  {len(d['duplicate_columns'])} repeated columns and "
              f"{len(d['count_curves_refused'])} count curves dropped; "
              f"{int(weight.sum())} curves fitted", flush=True)
    scale = residual_scale(d, inst.eta)
    print(f"  curves: {d['channel']}   eta median {float(inst.eta.median()):.4f}   "
          f"median scale {float(scale.median()):.4f}", flush=True)
    gmask = geometry_mask(d["mask"])[0] > 0
    present = [i for i in range(N_CAMS) if bool(gmask[i])]

    rng = np.random.default_rng(a.seed)
    held = sorted(rng.choice(present, a.hold_out_geoms, replace=False).tolist()) \
        if a.hold_out_geoms else []
    fit_geoms = [g for g in present if g not in held]
    print(f"model {a.model}  R={R}  h from {Path(sup_stl).name}  "
          f"fit on {len(fit_geoms)} geometries" + (f", holding out {held}" if held else ""),
          flush=True)

    d_fit = data[fit_geoms].to(dev)
    s_fit = scale[fit_geoms].to(dev)

    w_fit = weight[fit_geoms].to(dev)
    n_obs = float(w_fit.sum()) * data.shape[-1]

    def cot_fn(cur):
        return 2.0 * w_fit[..., None] * (cur - d_fit) / (s_fit[..., None] ** 2) / n_obs

    def objective(z):
        cur = op.curves(support, z, R, geoms=fit_geoms)
        if cur is None:
            return float("inf"), None
        chi = whitened_misfit(cur.cpu(), data, scale, fit_geoms, weight)
        ridge = a.l2 * float((z[N_DIR:] ** 2).sum())
        return chi ** 2 + ridge, chi

    def held_misfit(operator, z):
        if not held:
            return float("nan")
        ch = operator.curves(support, z, R, geoms=held)
        return whitened_misfit(ch.cpu(), data, scale, held, weight) if ch is not None \
            else float("inf")

    def export_misfits(z):
        """The misfits of the body as it would be written, extracted at EXPORT_RES."""
        cur = op_export.curves(support, z, R, geoms=fit_geoms)
        fit = whitened_misfit(cur.cpu(), data, scale, fit_geoms, weight) if cur is not None \
            else float("inf")
        return fit, held_misfit(op_export, z)

    code = torch.zeros(CODE_DIM, device=dev)
    hist, best = [], None
    t0 = time.time()
    step = a.lr
    J, chi = objective(code)
    convex_fit, convex_held = export_misfits(code)
    print(f"  convex answer at export resolution: chi_fit {convex_fit:.3f}"
          + (f"  chi_held {convex_held:.3f}" if held else ""), flush=True)
    for it in range(a.steps + 1):
        if it % a.every == 0 or it == a.steps:
            row = {"step": it, "chi_fit": chi, "objective": J, "step_size": step,
                   "seconds": round(time.time() - t0, 1)}
            m = posed_mesh(op, support, code, R, EXPORT_RES)
            if m is not None:
                row["dice"] = truth_dice(m[0], m[1], a.model, a.data_dir)
                row["convexity"] = convexity(m[0], m[1])
            if held:
                row["chi_held"] = held_misfit(op, code)
            if best is None or chi < best["chi_fit"]:
                best = {**row, "code": code.detach().clone()}
            hist.append(row)
            print(f"  step {it:>4}  chi_fit {chi:7.3f}"
                  + (f"  chi_held {row['chi_held']:7.3f}" if held else "")
                  + f"  dice {row.get('dice', float('nan')):.4f}"
                    f"  convexity {row.get('convexity', float('nan')):.3f}"
                    f"  step {step:.4f}  [{row['seconds']:.0f}s]", flush=True)
        if it == a.steps or chi <= TARGET_SIGMA:
            break

        _, grad = op.adjoint(support, code, R, cot_fn, geoms=fit_geoms)
        if grad is None:
            print("  the body has no curves; stopping", flush=True)
            break
        grad = grad.detach()
        grad[N_DIR:] += 2.0 * a.l2 * code[N_DIR:]
        # one unit-RMS direction per block, so that neither the few dh coordinates nor the
        # many amplitudes set the step for the other
        direction = torch.zeros_like(grad)
        for sl in (slice(0, N_DIR), slice(N_DIR, None)):
            r = float(grad[sl].pow(2).mean().sqrt())
            if np.isfinite(r) and r > 0:
                direction[sl] = -grad[sl] / r
        if float(direction.abs().max()) == 0.0:
            print("  the gradient vanished; stopping", flush=True)
            break

        moved = False
        for _ in range(MAX_HALVINGS):
            trial = code + step * direction
            J_t, chi_t = objective(trial)
            if J_t < J:
                code, J, chi = trial, J_t, chi_t
                step *= STEP_GROW
                moved = True
                break
            step *= 0.5
        if not moved:
            print(f"  no step of size >= {step:.2e} lowers the objective; converged at "
                  f"step {it}", flush=True)
            break

    if a.out:
        z = best["code"] if best is not None else code.detach()
        m = posed_mesh(op, support, z, R, EXPORT_RES)
        if m is None:
            raise SystemExit("the final body is degenerate; nothing written")
        v, f = m
        fit_x, held_x = export_misfits(z)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        rep = export_stl(a.out, v, f)
        meta = {"model": a.model, "channel": d["channel"], "radius": R, "steps": a.steps,
                "lr": a.lr, "l2": a.l2, "phases": a.phases, "operator_res": a.operator_res,
                "export_res": EXPORT_RES, "held_out": held, "history": hist, "export": rep,
                "chi_fit_convex": convex_fit, "chi_held_convex": convex_held,
                "chi_fit_export": fit_x, "chi_held_export": held_x,
                "final_dice": truth_dice(v, f, a.model, a.data_dir),
                "final_convexity": convexity(v, f)}
        Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
        np.savez(Path(a.out).with_suffix(".code.npz"), code=z.cpu().numpy(),
                 support=support.cpu().numpy())
        print(f"  wrote {a.out}  ({rep['faces']} faces, volume {rep['volume']:.3f}); "
              f"at export resolution chi_fit {fit_x:.3f}"
              + (f", chi_held {held_x:.3f} against the convex answer's {convex_held:.3f}"
                 if held else ""), flush=True)


if __name__ == "__main__":
    main()

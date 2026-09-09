#!/usr/bin/env python3
"""Reconstruct one model by descending the exact misfit from the convex stage's answer.

    python scripts/reconstruct_map.py --model 3 --steps 300 --out results/map/Asteroid03.stl
    python scripts/reconstruct_map.py --model 3 --steps 300 --hold-out-geoms 6 --l2 3e-3

No flow, no training, no corpus. The convex stage fixes the support h; the 1728 lattice
amplitudes g start at zero -- which *is* the convex body -- and are moved by gradient descent
on the whitened misfit through the exact forward model. Where the amortised route pays the
operator's cost thousands of times to learn a prior it then uses seven times, this pays it a
few hundred times per model and nothing else.

The reason to expect anything from it: fitting the same lattice to Mithra by least squares,
with h pinned to its true hull, reproduces the body to Dice 0.991, and the convex stage's own
answer scores 0.715. The representation can hold the shape; the question this script asks is
whether the data can find it.

The reason to distrust it, and what the diagnostics are for: a lightcurve misfit is not the
score. The genetic-algorithm branch drove a photometric fitness monotonically down while Dice
against truth fell 0.65 -> 0.56. It was scoring a deforming non-convex mesh with a *convex*
forward model, so every shadow the deformation created was invisible to it, and that
particular trap is not this one -- but the general one is. So: `--hold-out-geoms` keeps
cameras out of the fit entirely and reports the misfit on them, which is the difference
between recovering a shape and fitting curves; and on a public model the Dice against truth is
printed every checkpoint, so a run that fits better while resembling the body less says so in
its own log rather than at submission time.
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
from hac26.data_io import N_CAMS, load_model_curves, public_stl          # noqa: E402
from hac26.field import CODE_DIM, N_DIR                                  # noqa: E402
from hac26.recon import dice, fit_to_cylinder, mesh_occupancy            # noqa: E402
from hac26.shapes import rescale_touch_z                                 # noqa: E402
from hac26.solvers.operator import CodeOperator                          # noqa: E402
from hac26.solvers.output import export_stl, restore_constraints         # noqa: E402
from reconstruct_lpd import (curve_pairs, geometry_mask, residual_scale,  # noqa: E402
                             support_from_convex, whitened_misfit)
from train_lpd import CALIBRATION, RENDER, _enable_tf32, load_instrument  # noqa: E402

OCC_RES = 128
EXPORT_RES = 96       # extraction resolution for the written mesh; the fit runs coarser
TARGET_SIGMA = 1.0    # stop once the answer explains the data to the noise level
MAX_HALVINGS = 8      # trial steps per iteration before declaring convergence
STEP_GROW = 1.6       # a step that works makes the next trial bolder


def truth_dice(verts, faces, model: int, data_dir: str) -> float:
    """Dice of a posed reconstruction against the released truth, or nan when there is none."""
    if model not in PUBLIC_MODELS:
        return float("nan")
    import trimesh
    t = trimesh.load(public_stl(data_dir, model), process=False)
    tv = rescale_touch_z(np.asarray(t.vertices), np.asarray(t.faces), centre_xy=False)
    tf = np.asarray(t.faces)
    rv = rescale_touch_z(np.asarray(verts), np.asarray(faces), centre_xy=False)
    e = max(float(np.abs(rv).max()), float(np.abs(tv).max())) * 1.05
    return float(dice(mesh_occupancy(rv, np.asarray(faces), OCC_RES, e),
                      mesh_occupancy(tv, tf, OCC_RES, e)))


def convexity(verts, faces) -> float:
    import trimesh
    from scipy.spatial import ConvexHull
    m = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=True)
    try:
        return float(abs(m.volume) / ConvexHull(np.asarray(m.vertices)).volume)
    except Exception:                                    # noqa: BLE001
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--convex-dir", default="results/convex")
    ap.add_argument("--calibration", default=CALIBRATION)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05,
                    help="first trial step, in RMS code units per coordinate. Backtracking "
                         "adapts it, so this is a starting scale and not a schedule")
    ap.add_argument("--l2", type=float, default=3e-3,
                    help="ridge on g. The only prior here: 1728 amplitudes against 56 curves "
                         "is not obviously determined, and the ridge is what stops the fit "
                         "spending them on noise")
    ap.add_argument("--phases", type=int, default=96)
    ap.add_argument("--operator-res", type=int, default=32)
    ap.add_argument("--hold-out-geoms", type=int, default=0,
                    help="cameras kept out of the fit; their misfit is the honest test")
    ap.add_argument("--every", type=int, default=25, help="steps between diagnostics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--blender", action="store_true",
                    help="fit the Blender curves instead of the real ones. They exist for all "
                         "ten models and nothing in this repository has ever read them; they "
                         "are a render of the true shape by a known camera, so the "
                         "forward-model error against them is a different and much smaller "
                         "quantity than the eta fitted to the lab curves")
    ap.add_argument("--eta", type=float, default=-1.0,
                    help="override the calibration's per-curve model error. The fitted eta "
                         "(~0.07 of the curve mean) is the mismatch against the LAB curves; "
                         "against Blender it is far smaller, and leaving it high makes the "
                         "noise floor swallow the concavity signal")
    ap.add_argument("--dh-weight", type=float, default=1.0,
                    help="relative step for the convex-core block against the carving block")
    ap.add_argument("--uncalibrated", action="store_true",
                    help="run against a default Instrument instead of a fitted one. For "
                         "shaking out the plumbing only: the misfit is then measured against "
                         "an instrument nobody fitted, so its absolute value means nothing "
                         "and neither does a recipe tuned on it")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    _enable_tf32()
    dev = a.device if torch.cuda.is_available() else "cpu"
    R = CYLINDER_R[a.model]

    if a.uncalibrated:
        from hac26.forward.mesh.instrument import Instrument
        inst = Instrument().to(dev).requires_grad_(False)
        print("  WARNING: running --uncalibrated; the misfit is not measured against a "
              "fitted instrument", flush=True)
    else:
        inst = load_instrument(a.calibration, device=dev)
    eta56 = inst.eta
    psi = psi_grid(a.phases)
    op = CodeOperator(inst, psi, res=a.operator_res, config=RENDER, device=dev)

    sup_stl = str(Path(a.convex_dir) / f"Asteroid{a.model:02d}.stl")
    support = support_from_convex(sup_stl)

    d = load_model_curves(a.data_dir, a.model, m=a.phases, use_blender=a.blender)
    if d["mask"].sum() < 2 * N_CAMS:
        raise SystemExit(f"model {a.model}: only {int(d['mask'].sum())} of {2 * N_CAMS} "
                         f"curves present; refusing to fit around missing data")
    data = curve_pairs(d["curves"])
    if a.eta >= 0:
        eta56 = torch.full_like(eta56, a.eta)
    scale = residual_scale(d, eta56)
    print(f"  curves: {'blender' if a.blender else 'real'}   "
          f"eta {float(eta56.median()):.4f}   median scale {float(scale.median()):.4f}",
          flush=True)
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
    n_obs = d_fit.numel()

    def cot_fn(cur):
        return 2.0 * (cur - d_fit) / (s_fit[..., None] ** 2) / n_obs

    code = torch.zeros(CODE_DIM, device=dev)
    hist, best = [], None
    t0 = time.time()

    # Backtracking rather than a fixed step. Adam on these amplitudes overshoots badly -- at
    # lr 0.01 the misfit went 1.85 -> 5.74 in one step -- because the objective's curvature
    # varies over orders of magnitude across the 1728 coordinates and nothing here normalises
    # it. polish() in reconstruct_lpd.py already solved this the robust way: step along the
    # gradient, halve until the objective actually falls, grow the step when it does. That
    # also makes the run insensitive to the binary channel's gradient being about half scale
    # (tests/test_vertex_gradient.py), since a line search only needs the direction.
    def objective(z):
        cur = op.curves(support, z, R, geoms=fit_geoms)
        if cur is None:
            return float("inf"), None
        chi = whitened_misfit(cur.cpu(), data, scale, fit_geoms)
        ridge = a.l2 * float((z[N_DIR:] ** 2).sum())
        return chi ** 2 + ridge, chi

    step = a.lr
    J, chi = objective(code)
    for it in range(a.steps + 1):
        if it % a.every == 0 or it == a.steps:
            m = op.mesh(support, code, res=EXPORT_RES)
            row = {"step": it, "chi_fit": chi, "objective": J, "step_size": step,
                   "seconds": round(time.time() - t0, 1)}
            if m is not None:
                v = fit_to_cylinder(restore_constraints(
                    CodeOperator.canonical(m[0], m[1]).cpu().numpy(), R), R)
                f = m[1].cpu().numpy()
                row["dice"] = truth_dice(v, f, a.model, a.data_dir)
                row["convexity"] = convexity(v, f)
                if held:
                    ch = op.curves(support, code, R, geoms=held)
                    row["chi_held"] = (whitened_misfit(ch.cpu(), data, scale, held)
                                       if ch is not None else float("inf"))
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
        # Per-block normalisation. dh is 128 coordinates band-limited to 36 effective
        # degrees of freedom that shift the convex core; g is 1728 amplitudes that carve.
        # They are in different units, and one global RMS lets whichever block has the larger
        # gradient set the step for both -- which is how a run ends up at convexity 0.999
        # having "converged": the core absorbed the misfit and the amplitudes never moved.
        direction = torch.zeros_like(grad)
        for sl, w in ((slice(0, N_DIR), a.dh_weight), (slice(N_DIR, None), 1.0)):
            r = float(grad[sl].pow(2).mean().sqrt())
            if np.isfinite(r) and r > 0:
                direction[sl] = -w * grad[sl] / r
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
        m = op.mesh(support, z, res=EXPORT_RES)
        if m is None:
            raise SystemExit("the final body is degenerate; nothing written")
        v = fit_to_cylinder(restore_constraints(
            CodeOperator.canonical(m[0], m[1]).cpu().numpy(), R), R)
        f = m[1].cpu().numpy()
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        rep = export_stl(a.out, v, f)
        meta = {"model": a.model, "radius": R, "steps": a.steps, "lr": a.lr, "l2": a.l2,
                "phases": a.phases, "operator_res": a.operator_res, "held_out": held,
                "history": hist, "export": rep,
                "final_dice": truth_dice(v, f, a.model, a.data_dir),
                "final_convexity": convexity(v, f)}
        Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
        np.savez(Path(a.out).with_suffix(".code.npz"), code=z.cpu().numpy(),
                 support=support.cpu().numpy())
        print(f"  wrote {a.out}  ({rep['faces']} faces, volume {rep['volume']:.3f})",
              flush=True)


if __name__ == "__main__":
    main()

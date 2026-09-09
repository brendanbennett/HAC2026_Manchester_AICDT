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
EXPORT_RES = 96      # extraction resolution for the written mesh; the fit runs coarser


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
    ap.add_argument("--lr", type=float, default=0.01, help="Adam step on the raw amplitudes")
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
        eta56 = inst.eta.detach().cpu()
        print("  WARNING: running --uncalibrated; the misfit is not measured against a "
              "fitted instrument", flush=True)
    else:
        inst, eta56 = load_instrument(a.calibration, device=dev)
    psi = psi_grid(a.phases)
    op = CodeOperator(inst, psi, res=a.operator_res, config=RENDER, device=dev)

    sup_stl = str(Path(a.convex_dir) / f"Asteroid{a.model:02d}.stl")
    support = support_from_convex(sup_stl)

    d = load_model_curves(a.data_dir, a.model, m=a.phases)
    if d["mask"].sum() < 2 * N_CAMS:
        raise SystemExit(f"model {a.model}: only {int(d['mask'].sum())} of {2 * N_CAMS} "
                         f"curves present; refusing to fit around missing data")
    data = curve_pairs(d["curves"])
    scale = residual_scale(d, eta56)
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
    opt = torch.optim.Adam([code.requires_grad_(True)], lr=a.lr)
    hist, best = [], None
    t0 = time.time()

    for step in range(a.steps + 1):
        cur, grad = op.adjoint(support, code.detach(), R, cot_fn, geoms=fit_geoms)
        if cur is None:
            print(f"  step {step}: the body has no curves; stopping", flush=True)
            break
        chi_fit = whitened_misfit(cur.cpu(), data, scale, fit_geoms)
        if step % a.every == 0 or step == a.steps:
            m = op.mesh(support, code.detach(), res=EXPORT_RES)
            row = {"step": step, "chi_fit": chi_fit, "seconds": round(time.time() - t0, 1)}
            if m is not None:
                v = fit_to_cylinder(restore_constraints(
                    CodeOperator.canonical(m[0], m[1]).cpu().numpy(), R), R)
                f = m[1].cpu().numpy()
                row["dice"] = truth_dice(v, f, a.model, a.data_dir)
                row["convexity"] = convexity(v, f)
                if held:
                    ch, _ = op.adjoint(support, code.detach(), R,
                                       lambda c: torch.zeros_like(c), geoms=held)
                    row["chi_held"] = (whitened_misfit(ch.cpu(), data, scale, held)
                                       if ch is not None else float("inf"))
                if best is None or chi_fit < best["chi_fit"]:
                    best = {**row, "code": code.detach().clone()}
            hist.append(row)
            print(f"  step {row['step']:>4}  chi_fit {chi_fit:7.3f}"
                  + (f"  chi_held {row['chi_held']:7.3f}" if held else "")
                  + f"  dice {row.get('dice', float('nan')):.4f}"
                    f"  convexity {row.get('convexity', float('nan')):.3f}"
                    f"  [{row['seconds']:.0f}s]", flush=True)
        if step == a.steps:
            break
        # the ridge is applied to the amplitudes only; dh is 36 effective degrees of freedom
        # correcting the convex stage's bias and needs no shrinking
        g = code[N_DIR:]
        opt.zero_grad(set_to_none=True)
        code.grad = grad.detach()
        code.grad[N_DIR:] += 2.0 * a.l2 * g.detach()
        opt.step()

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

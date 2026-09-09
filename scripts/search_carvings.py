#!/usr/bin/env python3
"""Search over whole carved bodies rather than descending toward one.

    python scripts/search_carvings.py --model 3 --candidates 300
    python scripts/search_carvings.py --models 1 2 3 --candidates 300 --out runs/carve_search

Why a search and not an optimiser. Walking the straight line from the convex answer to the
true body (the oracle) shows the misfit going 2.147 -> 2.336 -> 1.898 while Dice climbs
0.690 -> 0.991 the whole way: the truth is a real minimum, and a better one, but it sits
behind a barrier about 9% high. Gradient descent from the convex body walks away from it --
measurably, all the way down to chi 1.11 at Dice 0.665, fitting model error with 1728 free
coordinates. A search that proposes whole bodies never takes that path: each candidate is
evaluated where it stands, and the low-dimensional recipe behind it (a handful of numbers
naming a waist or a basin) leaves nothing to overfit with.

How a candidate is made. The body is f(y) = core(y) + Delta(y) < 0, so a POSITIVE Delta
pushes the surface inward -- Delta is the carve. A candidate is a target carve field T
sampled at the lattice sites and turned into amplitudes by one solve of K g = T, where K is
the site-to-site kernel matrix (factorised once). That gives exact control: ask for a waist
of a given depth and width and get amplitudes that produce it, rather than hoping a random
draw in a 1728-dimensional space lands on something shaped like a body.

The families are the ones real targets come in and the ones the shape library already
generates: a waist or neck (a slab of carve about a plane, which is what makes a contact
binary), basins (blobs bitten out of the surface), and saw cuts (a whole cap removed). One to
three features per candidate, amplitudes and widths drawn over ranges that bracket the
oracle's own (|g|max 0.52, convexity 0.78).

What is accepted. Only a candidate that beats the convex answer's misfit by a clear margin,
because a near-convex body -- Vesta is 0.994 convex -- has nothing to gain and everything to
lose from being carved. The margin is a parameter and is chosen on the public models, where
the Dice is visible; --report-dice prints it per candidate so the rule can be checked rather
than assumed.
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

from hac26.conventions import CYLINDER_R, PUBLIC_MODELS, psi_grid          # noqa: E402
from hac26.data_io import N_CAMS, load_model_curves                        # noqa: E402
from hac26.field import CODE_DIM, N_DIR, ImplicitBody                      # noqa: E402
from hac26.recon import fit_to_cylinder                                    # noqa: E402
from hac26.solvers.operator import CodeOperator                            # noqa: E402
from hac26.solvers.output import export_stl, restore_constraints           # noqa: E402
from reconstruct_lpd import (curve_pairs, geometry_mask, residual_scale,    # noqa: E402
                             support_from_convex, whitened_misfit)
from reconstruct_map import EXPORT_RES, convexity, truth_dice               # noqa: E402
from train_lpd import CALIBRATION, RENDER, _enable_tf32, load_instrument    # noqa: E402


class CarveBasis:
    """Turns a target carve field into lattice amplitudes: g = K^-1 T(sites)."""

    def __init__(self, device: str, ridge: float = 1e-2):
        lat = ImplicitBody().to(device).delta
        self.sites = lat.p.detach()                                  # (N_SITES, 3)
        d2 = (self.sites ** 2 * lat.inv2).sum(1, keepdim=True) + lat.pb[None] \
            - 2.0 * ((self.sites * lat.inv2) @ lat.p.T)
        K = torch.exp(-0.5 * d2.clamp_min(0.0)).double()
        # The sites are 0.2 apart and the kernels are 0.18 wide, so neighbouring columns are
        # nearly parallel and K is numerically singular -- a plain Cholesky fails on it. The
        # ridge is not a numerical afterthought: it is what makes the solve ask for the
        # SMOOTHEST amplitudes that produce the requested carve rather than the exact ones,
        # and the exact ones would be a huge alternating pattern that marching cubes could not
        # resolve anyway. Chosen relative to the diagonal so it does not depend on the
        # lattice size.
        K.diagonal().add_(ridge * K.diagonal().mean())
        self.chol = torch.linalg.cholesky(K)
        self.device = device

    def amplitudes(self, target: torch.Tensor) -> torch.Tensor:
        g = torch.cholesky_solve(target.double()[:, None], self.chol)[:, 0]
        return g.float()


def _unit(rng, equatorial_bias=0.7):
    """A random direction, biased toward the equatorial plane: a waist across the spin axis
    is what a contact binary has, and it is the case the convex stage cannot see."""
    if rng.random() < equatorial_bias:
        a = rng.uniform(0, 2 * np.pi)
        v = np.array([np.cos(a), np.sin(a), rng.normal(0, 0.25)])
    else:
        v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def draw_target(rng, sites: np.ndarray, family: str = "mixed") -> tuple[np.ndarray, dict]:
    """One candidate carve field sampled at the lattice sites, with its recipe.

    `family="waist"` restricts the draw to a single waist -- four numbers: an axis, an offset
    along it, a width and a depth. That is worth having as its own mode because the mixed
    draw spends its candidates on a space far too large to cover: the line probe says a body
    must be about 85% of the way to the truth before its misfit beats the convex answer's, so
    a search only pays off if it can cover its family densely, and only a small family can be
    covered. A contact binary is a waist, so for the one public body that needs concavity this
    family contains the answer.
    """
    T = np.zeros(len(sites))
    recipe = {"features": []}
    if family == "waist":
        n = _unit(rng, equatorial_bias=0.9)
        d = rng.uniform(-0.45, 0.45)
        w = rng.uniform(0.10, 0.50)
        A = rng.uniform(0.10, 1.20)
        T += A * np.exp(-(((sites @ n) - d) / w) ** 2)
        recipe["features"].append({"kind": "waist", "n": n.tolist(), "d": float(d),
                                   "w": float(w), "A": float(A)})
        return T, recipe
    for _ in range(rng.integers(1, 4)):
        kind = rng.choice(["waist", "basin", "cut"], p=[0.5, 0.35, 0.15])
        n = _unit(rng)
        if kind == "waist":
            d = rng.normal(0, 0.25)
            w = rng.uniform(0.12, 0.45)
            A = rng.uniform(0.15, 0.85)
            T += A * np.exp(-(((sites @ n) - d) / w) ** 2)
            f = {"kind": kind, "n": n.tolist(), "d": float(d), "w": float(w), "A": float(A)}
        elif kind == "basin":
            c = _unit(rng, 0.5) * rng.uniform(0.55, 1.05)
            r = rng.uniform(0.18, 0.55)
            A = rng.uniform(0.15, 0.8)
            T += A * np.exp(-((np.linalg.norm(sites - c, axis=1)) / r) ** 2)
            f = {"kind": kind, "c": c.tolist(), "r": float(r), "A": float(A)}
        else:
            d = rng.uniform(0.35, 0.9)
            w = rng.uniform(0.05, 0.2)
            A = rng.uniform(0.2, 0.9)
            T += A / (1.0 + np.exp(-(((sites @ n) - d) / w)))
            f = {"kind": kind, "n": n.tolist(), "d": float(d), "w": float(w), "A": float(A)}
        recipe["features"].append(f)
    return T, recipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", type=int, required=True)
    ap.add_argument("--candidates", type=int, default=300)
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--convex-dir", default="results/convex")
    ap.add_argument("--calibration", default=CALIBRATION)
    ap.add_argument("--phases", type=int, default=48)
    ap.add_argument("--operator-res", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.03,
                    help="relative misfit improvement a carved candidate must show over the "
                         "convex answer before it is accepted")
    ap.add_argument("--hold-out-geoms", type=int, default=0)
    ap.add_argument("--report-dice", action="store_true",
                    help="print Dice per accepted candidate (public models only)")
    ap.add_argument("--out", default="runs/carve_search")
    ap.add_argument("--stl-dir", default="")
    ap.add_argument("--family", choices=["mixed", "waist"], default="mixed",
                    help="'waist' searches a four-parameter family densely instead of "
                         "scattering candidates over everything the library can make")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    _enable_tf32()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inst = load_instrument(a.calibration, device=dev)
    op = CodeOperator(inst, psi_grid(a.phases), res=a.operator_res, config=RENDER, device=dev)
    basis = CarveBasis(dev)
    sites_np = basis.sites.cpu().numpy()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    summary = {}
    for M in a.models:
        R = CYLINDER_R[M]
        support = support_from_convex(str(Path(a.convex_dir) / f"Asteroid{M:02d}.stl"))
        d = load_model_curves(a.data_dir, M, m=a.phases)
        data, scale = curve_pairs(d["curves"]), residual_scale(d, inst.eta)
        gmask = geometry_mask(d["mask"])[0] > 0
        present = [i for i in range(N_CAMS) if bool(gmask[i])]
        rng = np.random.default_rng(a.seed + 1000 * M)
        held = sorted(rng.choice(present, a.hold_out_geoms, replace=False).tolist()) \
            if a.hold_out_geoms else []
        fit_geoms = [g for g in present if g not in held]

        def chi_of(code, geoms=fit_geoms):
            cur = op.curves(support, code, R, geoms=geoms)
            return (whitened_misfit(cur.cpu(), data, scale, geoms)
                    if cur is not None else float("inf"))

        def body_of(code):
            m = op.mesh(support, code, res=EXPORT_RES)
            if m is None:
                return None, None
            v = fit_to_cylinder(restore_constraints(
                CodeOperator.canonical(m[0], m[1]).cpu().numpy(), R), R)
            return v, m[1].cpu().numpy()

        base = torch.zeros(CODE_DIM, device=dev)
        chi0 = chi_of(base)
        v0, f0 = body_of(base)
        dice0 = truth_dice(v0, f0, M, a.data_dir)
        print(f"\nmodel {M}  R={R}  convex answer: chi {chi0:.4f}  dice {dice0:.4f}"
              + (f"  (holding out {held})" if held else ""), flush=True)

        rows, best, t0 = [], None, time.time()
        for i in range(a.candidates):
            T, recipe = draw_target(rng, sites_np, a.family)
            code = torch.zeros(CODE_DIM, device=dev)
            code[N_DIR:] = basis.amplitudes(torch.tensor(T, dtype=torch.float32, device=dev))
            chi = chi_of(code)
            row = {"i": i, "chi": chi, "recipe": recipe,
                   "gmax": float(code[N_DIR:].abs().max())}
            if np.isfinite(chi) and (best is None or chi < best["chi"]):
                v, f = body_of(code)
                if v is not None:
                    row["dice"] = truth_dice(v, f, M, a.data_dir)
                    row["convexity"] = convexity(v, f)
                    if held:
                        row["chi_held"] = chi_of(code, held)
                    best = {**row, "code": code.detach().clone()}
                    print(f"  [{i:>4}] chi {chi:7.4f} ({100*(1-chi/chi0):+5.1f}%)  "
                          f"dice {row.get('dice', float('nan')):.4f}  "
                          f"convexity {row.get('convexity', float('nan')):.3f}  "
                          f"|g|max {row['gmax']:.3f}"
                          + (f"  held {row['chi_held']:.3f}" if held else ""), flush=True)
            rows.append({k: v for k, v in row.items() if k != "code"})

        gain = (chi0 - best["chi"]) / chi0 if best else 0.0
        accept = bool(best and gain >= a.margin)
        print(f"  searched {a.candidates} in {time.time()-t0:.0f}s   best gain {100*gain:+.1f}%"
              f"   -> {'ACCEPT carved' if accept else 'KEEP convex'}", flush=True)
        if best:
            print(f"  convex  chi {chi0:.4f} dice {dice0:.4f}   |   "
                  f"carved  chi {best['chi']:.4f} dice {best.get('dice', float('nan')):.4f}",
                  flush=True)

        summary[M] = {"chi_convex": chi0, "dice_convex": dice0,
                      "chi_best": best["chi"] if best else None,
                      "dice_best": best.get("dice") if best else None,
                      "convexity_best": best.get("convexity") if best else None,
                      "gain": gain, "accepted": accept, "held_out": held,
                      "candidates": a.candidates,
                      "best_recipe": best["recipe"] if best else None}
        Path(a.out, f"model{M:02d}.json").write_text(json.dumps(
            {"summary": summary[M], "rows": rows}, indent=2))
        if a.stl_dir and best is not None and accept:
            v, f = body_of(best["code"])
            Path(a.stl_dir).mkdir(parents=True, exist_ok=True)
            export_stl(str(Path(a.stl_dir) / f"Asteroid{M:02d}.stl"), v, f)

    Path(a.out, "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'model':>6} {'chi_cvx':>9} {'chi_best':>9} {'gain':>7} "
          f"{'dice_cvx':>9} {'dice_best':>10} {'verdict':>14}")
    for M, s in summary.items():
        dn = s["dice_best"] if s["dice_best"] is not None else float("nan")
        print(f"{M:>6} {s['chi_convex']:>9.4f} {s['chi_best']:>9.4f} {100*s['gain']:>6.1f}% "
              f"{s['dice_convex']:>9.4f} {dn:>10.4f} "
              f"{'ACCEPT carved' if s['accepted'] else 'keep convex':>14}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a spherical t-design of N normals and cache it beside hac26/field.py.

    python scripts/make_design.py --n 4096 --device cuda

The design is a fixed asset: generate once, commit the .npy, and every run loads it. The
energy is a pair of N x N Gram matrices per Legendre order, so the cost grows as N^2 and a
large design is worth generating on a GPU.

A t-design integrates every spherical harmonic up to order t exactly, which keeps the
support-function quadrature unbiased; a Fibonacci spiral is only asymptotically uniform and
leaves a low-order residual.

The six axis directions are pinned. The core is an intersection of half-spaces, so it
reproduces a flat face exactly only when that face's normal is present; a design free to
drift leaves the nearest normal a few degrees off and the intersection bulges at the face
centre.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.field import DESIGN_T, design_energy      # noqa: E402


def build(n: int, t: int = DESIGN_T, iters: int = 4000, lr: float = 1e-2,
          device: str = "cpu", seed: int = 0, report: int = 250) -> np.ndarray:
    axes = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0],
                     [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
    m = n - len(axes)
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    tht = np.pi * (1 + 5 ** 0.5) * i           # golden angle: the spiral start point
    free = np.stack([np.cos(tht) * np.sin(phi),
                     np.sin(tht) * np.sin(phi), np.cos(phi)], 1)

    p = torch.tensor(free, dtype=torch.float64, device=device, requires_grad=True)
    fixed = torch.tensor(axes, dtype=torch.float64, device=device)
    opt = torch.optim.Adam([p], lr=lr)
    best, best_x = float("inf"), None
    t0 = time.time()
    for k in range(iters):
        x = torch.cat([fixed, p / p.norm(dim=1, keepdim=True)], 0)
        loss = design_energy(x, t)
        opt.zero_grad(); loss.backward(); opt.step()
        v = float(loss.detach())
        if v < best:
            best, best_x = v, x.detach().clone()
        if report and (k % report == 0 or k == iters - 1):
            print(f"  iter {k:>5}  energy {v:.6e}  best {best:.6e}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    return best_x.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of normals")
    ap.add_argument("--t", type=int, default=DESIGN_T, help="design strength")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out) if a.out else Path(__file__).resolve().parents[1] / "hac26" / \
        f"design{a.n}.npy"
    print(f"building a strength-{a.t} design of {a.n} normals on {a.device}", flush=True)
    x = build(a.n, a.t, a.iters, a.lr, a.device)

    # report the residual the design is supposed to kill, and the facet geometry it implies
    res = float(design_energy(torch.tensor(x, dtype=torch.float64), a.t))
    half = np.degrees(np.arccos(1.0 - 2.0 / a.n))
    print(f"\n  design residual      {res:.3e}")
    print(f"  facet half-angle     {half:.2f} deg")
    print(f"  facet width          {2*np.sin(np.radians(half)):.3f} R")
    print(f"  bulge at face centre {2.0/a.n*100:.2f}% of the support distance")
    np.save(out, x)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

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

from hac26.field import DESIGN_ITERS, DESIGN_T, _design_residual, design_energy      # noqa: E402


def build(n: int, t: int = DESIGN_T, iters: int = DESIGN_ITERS, lr: float = 1e-2,
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
        v = abs(float(loss.detach()))   # the sum cancels to ~1e-15 and can go slightly
        if v < best:                    # negative; selecting on the signed value latches
                                        # onto numerical noise rather than the minimum
            best, best_x = v, x.detach().clone()
        if report and (k % report == 0 or k == iters - 1):
            print(f"  iter {k:>5}  energy {v:.6e}  best {best:.6e}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    return best_x.cpu().numpy()


TOL = 1e-5      # the bound tests/test_field.py asserts on a published design


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of normals")
    ap.add_argument("--t", type=int, default=DESIGN_T, help="design strength")
    ap.add_argument("--iters", type=int, default=DESIGN_ITERS)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None,
                    help="write somewhere other than the hac26/design<n>.npy cache; the "
                         "quality gate below still applies")
    a = ap.parse_args()

    out = Path(a.out) if a.out else Path(__file__).resolve().parents[1] / "hac26" / \
        f"design{a.n}.npy"
    print(f"building a strength-{a.t} design of {a.n} normals on {a.device}", flush=True)
    x = build(a.n, a.t, a.iters, a.lr, a.device)

    # Report the WORST SINGLE-DEGREE residual, which is what defines a t-design and what the
    # tests gate on -- not design_energy, which is the SUM over degrees and was printed here
    # under the label "design residual".
    xt = torch.tensor(x, dtype=torch.float64)
    res = _design_residual(x, a.t)
    energy = float(design_energy(xt, a.t))
    half = np.degrees(np.arccos(1.0 - 2.0 / a.n))
    print(f"\n  worst-degree residual {res:.3e}   (summed energy {energy:.3e})")
    print(f"  facet half-angle     {half:.2f} deg")
    print(f"  facet width          {2*np.sin(np.radians(half)):.3f} R")
    print(f"  bulge at face centre {2.0/a.n*100:.2f}% of the support distance")

    # Refuse to publish a short build. field.py::_write_design_cache already will not cache
    # one, for the same reason: the filename keys on n alone and the design_sha is identical,
    # so nothing downstream could tell a short build from a good one. This script is the only
    # route to design4096.npy, since spherical_design refuses n > 512.
    if res > TOL:
        raise SystemExit(
            f"worst-degree residual {res:.3e} exceeds {TOL:.0e}: this design is not good "
            f"enough to publish as hac26/design{a.n}.npy, and nothing downstream could tell "
            f"it apart from a good one. Raise --iters (default {DESIGN_ITERS}) and rerun, or "
            f"pass --out to write it somewhere that is not the cache.")
    np.save(out, x)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does the trained flow use the curves, or has it memorised the corpus?

The flow sees x_t = (1 - t) x0 + t x1, which reveals x1 more and more as t grows, and with a
small corpus a network can recognise which body it is near without consulting a curve. This
script evaluates the trained flow twice on the same draws: the full velocity, prior plus
data part, and the prior's velocity alone, which reads no curve. If the two losses are close,
the data part is decoration; if dropping it hurts, the curves are contributing by that margin.
The flow and occupancy terms are compared; the data-fit term is left out of both arms.

Both arms come from one call of train_lpd.flow_loss, off one operator call, so they differ by
the data part and nothing else; this script only supplies the draws.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.conventions import cameras, psi_grid          # noqa: E402
from hac26.solvers.lpd_flow import N_MODES, LPDFlow, geometry_tags   # noqa: E402
from hac26.solvers.operator import CodeOperator          # noqa: E402
from train_lpd import (CALIBRATION, CORPUS, OCC_WEIGHT, RENDER, file_digest, flow_loss,   # noqa: E402
                       held_out, load_corpus, load_instrument, model_error_scale,
                       noise_sigma, smooth_noise_like)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/lpd_flow.pt")
    ap.add_argument("--corpus", default=CORPUS, help="must match the training run")
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--calibration", default=CALIBRATION,
                    help="must match the training run")
    ap.add_argument("--val-bodies", type=int, default=8,
                    help="must match the training run, so the same bodies are held out")
    ap.add_argument("--occ-weight", type=float, default=OCC_WEIGHT,
                    help="must match the training run")
    ap.add_argument("--occ-eps", type=float, default=None,
                    help="must match the training run")
    a = ap.parse_args()

    data, meta = load_corpus(a.corpus)
    if meta["calibration"] != file_digest(a.calibration):
        raise SystemExit(f"{a.corpus} was built with another calibration than {a.calibration}")
    phases, op_res = int(meta["phases"]), int(meta["operator_res"])
    M = min(N_MODES, phases // 2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inst = load_instrument(a.calibration, dev)
    eta = model_error_scale(inst)
    op = CodeOperator(inst, psi_grid(phases), res=op_res, config=RENDER, device=dev)
    net = LPDFlow.from_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    net = net.to(dev).eval()
    data = data.to(dev)
    codes = data.codes

    C = len(cameras())
    tag = geometry_tags().to(dev)
    mask = torch.ones(1, C, device=dev)

    # Score the training bodies, split off by the same rule train_lpd.py uses (held_out, by
    # codes-file index). A held-out body cannot have been memorised, so only the training
    # bodies can show the failure this script looks for.
    n_val = max(0, min(a.val_bodies, len(codes) - 1))
    is_val = np.isin(data.index.cpu().numpy(), held_out(int(meta["bodies"]), n_val))
    pool = torch.nonzero(torch.as_tensor(~is_val)).flatten()

    # t is stratified over the bins and continuous within each, as in training. The bins are
    # only for reporting.
    n_bins = 6
    gen = torch.Generator().manual_seed(0)
    idx = pool[torch.randint(0, len(pool), (a.draws,), generator=gen)].to(dev)
    x0 = torch.randn(a.draws, codes.shape[1], dtype=codes.dtype, generator=gen).to(dev)
    t = (((torch.arange(a.draws) % n_bins).to(codes.dtype)
          + torch.rand(a.draws, generator=gen).to(codes.dtype)) / n_bins).to(dev)
    sigma = noise_sigma(a.draws, generator=gen).to(dev)
    xi = torch.randn(a.draws, C, 2, phases, generator=gen).to(dev)
    zeta = smooth_noise_like(data.curves[idx].cpu(), generator=gen).to(dev)

    real_loss, zero_loss, n_bad = [], [], 0
    per_t = {k: ([], []) for k in range(n_bins)}
    for d in range(a.draws):
        sl = slice(d, d + 1)
        with torch.no_grad():
            r_, z_, bad = flow_loss(net, op, data, eta, idx[sl], x0[sl], t[sl], M, tag,
                                    mask, sigma=sigma[sl], xi=xi[sl], zeta=zeta[sl],
                                    ablate=True, occ_weight=a.occ_weight, occ_eps=a.occ_eps)
        r_, z_ = float(r_), float(z_); n_bad += int(bad)
        real_loss.append(r_); zero_loss.append(z_)
        kb = d % n_bins
        per_t[kb][0].append(r_); per_t[kb][1].append(z_)
        print(f"  draw {d:>3}  t={float(t[d]):.3f}  full {r_:.5f}   prior only {z_:.5f}",
              flush=True)

    print("\n  per t bin:")
    for k in range(n_bins):
        rr, zz = per_t[k]
        if rr:
            print(f"    t={(k+0.5)/n_bins:.3f}   full {np.mean(rr):.5f}   "
                  f"prior only {np.mean(zz):.5f}")
    r_, z_ = float(np.mean(real_loss)), float(np.mean(zero_loss))
    print(f"\n  prior plus data part : {r_:.5f}")
    print(f"  prior only           : {z_:.5f}")
    print(f"  the data part is worth : {z_ - r_:+.5f}  ({100*(z_-r_)/max(z_,1e-12):+.1f}%)")
    if n_bad:
        print(f"\n  WARNING: {n_bad}/{a.draws} draws had no curves (degenerate mesh or "
              f"\n  unusable patches), so both arms saw a body without data on those draws, "
              f"\n  which pulls the margin toward zero.")
    print("\n  A margin near zero means the curves are decoration and the flow is"
          "\n  identifying corpus bodies from x_t and the prior alone.")


if __name__ == "__main__":
    main()

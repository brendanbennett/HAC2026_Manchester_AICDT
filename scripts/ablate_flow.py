#!/usr/bin/env python3
"""Does the trained flow actually USE the lightcurves, or has it memorised the corpus?

Beating the data-free bar is necessary evidence that the data contributes, and not
sufficient. The flow sees x_t = (1-t) x0 + t x1, which leaks x1 directly as t grows, and with
a corpus of only 40 bodies a network can identify WHICH body it is near without ever
consulting a curve. This ablation separates the two: evaluate the trained flow twice on the
same draws, once with the real residual and once with the residual channels zeroed.

    if the two are close      the curves are decoration and the flow is memorising
    if zeroing hurts          the operator is contributing, by that margin
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
from solvers.lpd_flow import N_MODES, N_STEPS, LPDFlow     # noqa: E402
from forward_models.learned_surrogate import Surrogate                    # noqa: E402
from train_lpd import curves_from_code                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/lpd_flow.pt")
    ap.add_argument("--corpus", default="/tmp/lpd_corpus_96_g28_h.npz")
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--phases", type=int, default=96)
    a = ap.parse_args()

    z = np.load(a.corpus)
    x1 = torch.tensor(z["codes"]); curves = torch.tensor(z["curves"])
    sup = torch.tensor(z["support"])
    psi = psi_grid(a.phases); M = min(N_MODES, a.phases // 2)

    gdev = "cuda" if torch.cuda.is_available() else "cpu"
    surro = Surrogate(width=96, modes=8, blocks=3)
    surro.load_state_dict(torch.load("model/surrogate.pt", map_location="cpu"))
    surro = surro.to(gdev).eval()
    net = LPDFlow(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu")); net.eval()

    tag = torch.zeros(1, 28, 4)
    for i, cam in enumerate(cameras()):
        tag[0, i] = torch.tensor([np.cos(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.elevation_deg)), 1.0])
    mask = torch.ones(1, 28)

    # STRATIFIED OVER t, not sampled. The target is u = (x1 - x_t)/(1 - t), so a draw at
    # k = 5 carries a gain of 6 and contributes 36x the squared error of one at k = 0. A loss
    # averaged over randomly drawn k therefore reports mostly WHICH k was drawn -- which is
    # why the training log swings between 0.002 and 0.2 from step to step and single values
    # cannot be compared. Every k is evaluated the same number of times here, and reported
    # separately as well as pooled.
    torch.manual_seed(0)
    real_loss, zero_loss = [], []
    per_t = {k: ([], []) for k in range(N_STEPS)}
    for d in range(a.draws):
        i = torch.randint(0, len(x1), (1,))
        y = x1[i]; x0 = torch.randn_like(y)
        k = torch.tensor([d % N_STEPS]); t = k.float() / N_STEPS
        xt = (1 - t[:, None]) * x0 + t[:, None] * y
        cur = curves_from_code(xt[0], 1.0, surro, psi, support=sup[i[0]])
        if cur is None:
            continue
        g_dat = torch.fft.rfft(curves[i], dim=-1)[..., 1:M + 1]
        g_cur = torch.fft.rfft(cur[None], dim=-1)[..., 1:M + 1]
        r = g_dat - g_cur
        rp = r.clone(); rp[..., :4] = 0
        feats = torch.zeros(1, 28, N_MODES, 6); perp = torch.zeros(1, 28, N_MODES, 6)
        for ch in range(2):
            feats[:, :, :M, 2 * ch] = r[:, :, ch].real
            feats[:, :, :M, 2 * ch + 1] = r[:, :, ch].imag
            perp[:, :, :M, 2 * ch] = rp[:, :, ch].real
            perp[:, :, :M, 2 * ch + 1] = rp[:, :, ch].imag
        feats[:, :, :M, 4] = g_dat[:, :, 0].real
        feats[:, :, :M, 5] = g_dat[:, :, 1].real
        zed = torch.zeros_like(feats)
        with torch.no_grad():
            u_real = net.velocity(xt, feats, perp, tag, mask, t)
            u_zero = net.velocity(xt, zed, zed, tag, mask, t)
        tgt = y - x0
        real_loss.append(float(((u_real - tgt) ** 2).mean()))
        zero_loss.append(float(((u_zero - tgt) ** 2).mean()))
        per_t[int(k)][0].append(real_loss[-1]); per_t[int(k)][1].append(zero_loss[-1])
        print(f"  draw {d:>3}  t={float(t):.3f}  real {real_loss[-1]:.5f}   "
              f"zeroed {zero_loss[-1]:.5f}", flush=True)

    print("\n  per step time (the 1/(1-t) gain makes these incomparable to each other):")
    for k in range(N_STEPS):
        rr, zz = per_t[k]
        if rr:
            print(f"    k={k}  t={k/N_STEPS:.3f}  gain {1/(1-k/N_STEPS):.1f}x   "
                  f"real {np.mean(rr):.5f}   zeroed {np.mean(zz):.5f}")
    r_, z_ = float(np.mean(real_loss)), float(np.mean(zero_loss))
    print(f"\n  with the real residual : {r_:.5f}")
    print(f"  residual zeroed        : {z_:.5f}")
    print(f"  the operator is worth  : {z_ - r_:+.5f}  ({100*(z_-r_)/max(z_,1e-12):+.1f}%)")
    print("\n  A margin near zero means the curves are decoration and the flow is"
          "\n  identifying corpus bodies from x_t alone.")


if __name__ == "__main__":
    main()

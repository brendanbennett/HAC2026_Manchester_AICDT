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
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.conventions import cameras, psi_grid          # noqa: E402
from hac26.field import DESIGN_N                         # noqa: E402
from hac26.solvers.lpd_flow import CODE_DIM, N_DIR, N_MODES, LPDFlow   # noqa: E402
from hac26.forward.learned_surrogate import Surrogate                    # noqa: E402
from train_lpd import (cond_channels, corpus_cache_path,   # noqa: E402
                       curves_from_code)                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/lpd_flow.pt")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--phases", type=int, default=96)
    ap.add_argument("--operator-res", type=int, default=32)
    ap.add_argument("--cache-tag", default="shared")
    a = ap.parse_args()

    corpus = a.corpus or corpus_cache_path(a.phases, len(cameras()), a.operator_res,
                                           a.cache_tag)
    z = np.load(corpus)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else None
    if meta is None:
        print(f"  WARNING: {corpus} has no metadata; make sure it is not a stale cache",
              flush=True)
    else:
        expected = {
            "phases": int(a.phases),
            "n_geoms": int(len(cameras())),
            "operator_res": int(a.operator_res),
            "design_n": int(DESIGN_N),
            "code_dim": int(CODE_DIM),
        }
        bad = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
        if bad:
            raise SystemExit(f"{corpus} metadata does not match this ablation: {bad}")
    x1 = torch.tensor(z["codes"]); curves = torch.tensor(z["curves"])
    sup = torch.tensor(z["support"])
    if x1.ndim != 2 or x1.shape[1] != CODE_DIM:
        raise SystemExit(f"{corpus} codes have shape {tuple(x1.shape)}, but "
                         f"CODE_DIM={CODE_DIM}; rebuild the corpus")
    if sup.ndim != 2 or sup.shape[1] != DESIGN_N:
        raise SystemExit(f"{corpus} support has shape {tuple(sup.shape)}, "
                         f"but DESIGN_N={DESIGN_N}")
    if curves.ndim != 4 or curves.shape[1:] != (len(cameras()), 2, a.phases):
        raise SystemExit(f"{corpus} curves have shape {tuple(curves.shape)}, "
                         "which does not match the requested phase/operator grid")
    psi = psi_grid(a.phases); M = min(N_MODES, a.phases // 2)

    gdev = "cuda" if torch.cuda.is_available() else "cpu"
    surro = Surrogate(width=96, modes=8, blocks=3)
    surro.load_state_dict(torch.load("runs/surrogate.pt", map_location="cpu"))
    surro = surro.to(gdev).eval()
    net = LPDFlow(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu")); net.eval()

    tag = torch.zeros(1, 28, 4)
    for i, cam in enumerate(cameras()):
        tag[0, i] = torch.tensor([np.cos(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.elevation_deg)), 1.0])
    mask = torch.ones(1, 28)

    # STRATIFIED OVER t, not sampled, and t is now CONTINUOUS -- the flow is trained that
    # way, so scoring it on six discrete times would score a different objective.
    #
    # A previous version of this comment claimed the target is u = (x1 - x_t)/(1 - t) so that
    # a draw near t = 1 "carries a gain of 6 and contributes 36x the squared error". In this
    # parameterisation that expression is IDENTICALLY x1 - x0: the target does not depend on
    # t at all, and the printed gain was not a property of it. What does vary with t is how
    # informative the operator call is, which is why the bins are still reported separately.
    torch.manual_seed(0)
    real_loss, zero_loss = [], []
    n_bins = 6
    per_t = {k: ([], []) for k in range(n_bins)}
    for d in range(a.draws):
        i = torch.randint(0, len(x1), (1,))
        y = net.codec.encode(x1[i]); x0 = torch.randn_like(y)
        kb = d % n_bins
        t = torch.tensor([(kb + 0.5) / n_bins])
        xt = (1 - t[:, None]) * x0 + t[:, None] * y
        raw = net.codec.decode(xt)
        h = sup[i[0]]
        cur = curves_from_code(raw[0], 1.0, surro, psi, support=h)
        if cur is None:
            continue
        g_dat = torch.fft.rfft(curves[i], dim=-1)[..., 1:M + 1]
        g_cur = torch.fft.rfft(cur[None], dim=-1)[..., 1:M + 1]
        r = g_dat - g_cur
        feats = torch.zeros(1, 28, N_MODES, 6)
        for ch in range(2):
            feats[:, :, :M, 2 * ch] = r[:, :, ch].real
            feats[:, :, :M, 2 * ch + 1] = r[:, :, ch].imag
        feats[:, :, :M, 4] = g_dat[:, :, 0].real
        feats[:, :, :M, 5] = g_dat[:, :, 1].real
        sph, vol = cond_channels(h[None])
        zed = torch.zeros_like(feats)
        with torch.no_grad():
            u_real = net.velocity(xt, feats, tag, mask, t, sph, vol)
            u_zero = net.velocity(xt, zed, tag, mask, t, sph, vol)
        tgt = y - x0
        # the same per-block weighting the training loss uses, or the two are not comparable
        def _l(u):
            e = (u - tgt) ** 2
            return 0.5 * float(e[:, :N_DIR].mean() + e[:, N_DIR:].mean())
        real_loss.append(_l(u_real)); zero_loss.append(_l(u_zero))
        per_t[kb][0].append(real_loss[-1]); per_t[kb][1].append(zero_loss[-1])
        print(f"  draw {d:>3}  t={float(t):.3f}  real {real_loss[-1]:.5f}   "
              f"zeroed {zero_loss[-1]:.5f}", flush=True)

    print("\n  per t bin:")
    for k in range(n_bins):
        rr, zz = per_t[k]
        if rr:
            print(f"    t={(k+0.5)/n_bins:.3f}   real {np.mean(rr):.5f}   "
                  f"zeroed {np.mean(zz):.5f}")
    r_, z_ = float(np.mean(real_loss)), float(np.mean(zero_loss))
    print(f"\n  with the real residual : {r_:.5f}")
    print(f"  residual zeroed        : {z_:.5f}")
    print(f"  the operator is worth  : {z_ - r_:+.5f}  ({100*(z_-r_)/max(z_,1e-12):+.1f}%)")
    print("\n  A margin near zero means the curves are decoration and the flow is"
          "\n  identifying corpus bodies from x_t alone.")


if __name__ == "__main__":
    main()

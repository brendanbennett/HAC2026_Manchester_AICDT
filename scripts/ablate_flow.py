#!/usr/bin/env python3
"""Does the trained flow actually USE the lightcurves, or has it memorised the corpus?

Beating the data-free bar is necessary evidence that the data contributes, and not
sufficient. The flow sees x_t = (1-t) x0 + t x1, which leaks x1 directly as t grows, and with
a small corpus a network can identify WHICH body it is near without ever
consulting a curve. This ablation separates the two: evaluate the trained flow twice on the
same draws, once with the curves and once without them.

    if the two are close      the curves are decoration and the flow is memorising
    if zeroing hurts          the operator is contributing, by that margin

"Without" means every channel that carries curve information: the Fourier residual, the raw
data coefficients sharing the same tensor (feats channels 4 and 5 are g_dat, not r), and the
adjoint channel on the sphere branch. An arm that keeps the raw coefficients is not data-free.

Both arms come from ONE flow_loss call off ONE operator call, so they differ by the switch and
nothing else. This file used to re-implement the objective and drifted from it; there is now
one implementation, in train_lpd.flow_loss, and this file only supplies the draws.
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
from hac26.solvers.lpd_flow import CODE_DIM, N_MODES, LPDFlow   # noqa: E402
from hac26.forward.learned_surrogate import load_surrogate       # noqa: E402
from train_lpd import (_dh_perturbation, corpus_cache_path,      # noqa: E402
                       flow_loss)                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/lpd_flow.pt")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--phases", type=int, default=96)
    ap.add_argument("--operator-res", type=int, default=32)
    ap.add_argument("--cache-tag", default="shared")
    ap.add_argument("--val-bodies", type=int, default=8,
                    help="must match the training run: the held-out split is reproduced here "
                         "so the ablation scores the bodies the flow could have memorised")
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
            "schema": 5,            # A(x) changed; a schema-4 cache is a different operator
            "phases": int(a.phases),
            "n_geoms": int(len(cameras())),
            "operator_res": int(a.operator_res),
            "design_n": int(DESIGN_N),
            "code_dim": int(CODE_DIM),
        }
        bad = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
        if bad:
            raise SystemExit(f"{corpus} metadata does not match this ablation: {bad}")
    codes = torch.tensor(z["codes"]); curves = torch.tensor(z["curves"])
    sup = torch.tensor(z["support"])
    if codes.ndim != 2 or codes.shape[1] != CODE_DIM:
        raise SystemExit(f"{corpus} codes have shape {tuple(codes.shape)}, but "
                         f"CODE_DIM={CODE_DIM}; rebuild the corpus")
    if sup.ndim != 2 or sup.shape[1] != DESIGN_N:
        raise SystemExit(f"{corpus} support has shape {tuple(sup.shape)}, "
                         f"but DESIGN_N={DESIGN_N}")
    if curves.ndim != 4 or curves.shape[1:] != (len(cameras()), 2, a.phases):
        raise SystemExit(f"{corpus} curves have shape {tuple(curves.shape)}, "
                         "which does not match the requested phase/operator grid")
    psi = psi_grid(a.phases); M = min(N_MODES, a.phases // 2)

    gdev = "cuda" if torch.cuda.is_available() else "cpu"
    surro, smeta = load_surrogate("runs/surrogate.pt", phases=a.phases, device=gdev)
    print(f"  surrogate {smeta}", flush=True)
    net = LPDFlow(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu")); net.eval()

    C = len(cameras())
    tag = torch.zeros(1, C, 4)
    for i, cam in enumerate(cameras()):
        tag[0, i] = torch.tensor([np.cos(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.elevation_deg)), 1.0])
    mask = torch.ones(1, C)

    # SCORE THE TRAINING BODIES, taken from the same fixed permutation train_lpd.py splits on.
    # A held-out body cannot be memorised, so a margin there proves nothing about the failure
    # this test exists to catch; the training bodies are the population where "identify the
    # body from x_t alone" is actually available to the network.
    perm = torch.randperm(len(codes), generator=torch.Generator().manual_seed(0))
    pool = perm[max(0, min(a.val_bodies, len(codes) - 1)):]

    # t stratified over the bins and continuous within each, matching how the flow is
    # trained. The bins are only for reporting: what varies with t is how informative the
    # operator call is.
    n_bins = 6
    gen = torch.Generator().manual_seed(0)
    idx = pool[torch.randint(0, len(pool), (a.draws,), generator=gen)]
    x0 = torch.randn(a.draws, codes.shape[1], dtype=codes.dtype, generator=gen)
    t = ((torch.arange(a.draws) % n_bins).to(codes.dtype)
         + torch.rand(a.draws, generator=gen).to(codes.dtype)) / n_bins
    eps = _dh_perturbation(a.draws, generator=gen)

    real_loss, zero_loss, n_bad = [], [], 0
    per_t = {k: ([], []) for k in range(n_bins)}
    for d in range(a.draws):
        sl = slice(d, d + 1)
        with torch.no_grad():
            r_, z_, bad = flow_loss(net, surro, psi, codes, curves, sup, idx[sl], x0[sl],
                                    t[sl], M, tag, mask, op_res=a.operator_res,
                                    eps=eps[sl], ablate=True)
        r_, z_ = float(r_), float(z_); n_bad += int(bad)
        real_loss.append(r_); zero_loss.append(z_)
        kb = d % n_bins
        per_t[kb][0].append(r_); per_t[kb][1].append(z_)
        print(f"  draw {d:>3}  t={float(t[d]):.3f}  real {r_:.5f}   zeroed {z_:.5f}",
              flush=True)

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
    if n_bad:
        print(f"\n  WARNING: {n_bad}/{a.draws} draws decoded to under 8 faces, so A(x) was "
              f"\n  unavailable and the residual was the raw data. Both arms saw the same "
              f"\n  thing on those draws, which pulls the margin toward zero.")
    print("\n  A margin near zero means the curves are decoration and the flow is"
          "\n  identifying corpus bodies from x_t alone.")


if __name__ == "__main__":
    main()

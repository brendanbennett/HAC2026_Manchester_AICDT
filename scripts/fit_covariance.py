#!/usr/bin/env python3
"""Fit the measurement covariance from the public triplet, and save it.

    sigma^2_{c,m}  photon noise, from the co-located horizontal replicate pairs
    eta^2_{c,m}    model error, from real - forward(true STL) under the calibrated nuisance
                   parameters, fitted as gamma_c tau0^2 (1 + m/m0)^(-2p) by maximum likelihood
    s^2            their sum, the only weight any objective may carry

The residual is formed exactly as the calibration forms its own: radiance rendered once per
body, then the sensor chain, pedestal and thresholds applied to the cached images, then the
per-curve mean normalisation. Anything else would fit the difference between two reduction
paths rather than the model error.

    python scripts/fit_covariance.py --phases 360

Rendering dominates the cost and needs a GPU rasteriser.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate import render_body                              # noqa: E402
from hac26.covariance import (fit_model_error, photon_modes,   # noqa: E402
                              psi_dft, save_covariance)
from hac26.data_io import load_model_curves                    # noqa: E402
from hac26.noise import sigma_from_replicates                  # noqa: E402
from hac26.forward.mesh.sensor import SensorModel             # noqa: E402


def predicted_curves(imgs, ras, cal, dev, chunk: int = 4) -> torch.Tensor:
    """Cached radiance -> the 56 mean-normalised curves, under the calibrated parameters."""
    sensor = SensorModel(quantise=True).to(dev)
    sensor.load_state_dict(cal["sensor"])
    ti = torch.sigmoid(cal["raw_tau_i"].to(dev))
    tb = torch.sigmoid(cal["raw_tau_b"].to(dev))
    ped = cal["pedestal"].to(dev)
    C, P, H, W = imgs.shape
    cos_off, rad = ras.pixel_geometry(np.radians(20.0))
    inten, binar = [], []
    with torch.no_grad():
        for c0 in range(0, C, chunk):
            im = imgs[c0:c0 + chunk].to(dev, non_blocking=True)
            n = im.shape[0]
            val = sensor(im.reshape(n * P, H, W), cos_off.expand(n * P, H, W),
                         rad.expand(n * P, H, W), supersample=ras.ss)
            val = val.reshape(n, P, val.shape[-2], val.shape[-1])
            val = val + ped[c0:c0 + n].reshape(n, 1, 1, 1)
            inten.append((val * (val > ti[c0:c0 + n].reshape(n, 1, 1, 1))).sum(dim=(-2, -1)))
            binar.append((val > tb[28 + c0:28 + c0 + n].reshape(n, 1, 1, 1))
                         .to(val.dtype).sum(dim=(-2, -1)))
            del im, val
    cur = torch.cat(inten + binar, 0)
    return (cur / cur.mean(1, keepdim=True).clamp_min(1e-9)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", type=int, default=360,
                    help="N in the DFT convention; sigma^2_m = sigma^2_c / N uses the same N")
    ap.add_argument("--modes", type=int, default=40)
    ap.add_argument("--height", type=int, default=108)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--raster-faces", type=int, default=120000)
    ap.add_argument("--calibration", default="models/instrument_calibration.pt")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--out", default="models/data_covariance.pt")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cal = torch.load(a.calibration, map_location="cpu", weights_only=False)
    rho = float(cal["rho"]); delta = float(np.radians(float(cal["delta_deg"])))
    print(f"calibration: rho {rho:.3f}, source radius {float(cal['delta_deg']):.2f} deg",
          flush=True)

    resid, sig = [], None
    for M in (1, 2, 3):
        d = load_model_curves(a.data_dir, M, m=a.phases)
        real = torch.tensor(d["curves"], dtype=torch.float32)
        if sig is None:
            sig = np.zeros((3, 56))
        sig[M - 1] = sigma_from_replicates(d["curves"], d["mask"])
        print(f"[render] model {M} at {a.phases} phases", flush=True)
        imgs, ras, _ = render_body(M, a.phases, (a.height, a.width), a.ss, rho, delta,
                                   device=dev, raster_faces=a.raster_faces)
        pred = predicted_curves(imgs, ras, cal, dev)
        r = real - pred
        print(f"           residual rms {float(r.pow(2).mean().sqrt()):.5f}", flush=True)
        resid.append(r)
        del imgs
        if dev == "cuda":
            torch.cuda.empty_cache()

    sigma_c = np.median(sig, axis=0)                       # (56,), across the three bodies
    sigma2_m = photon_modes(sigma_c, a.phases, a.modes)
    rhat = torch.stack([psi_dft(r, a.modes) for r in resid])       # (3, 56, M) complex
    print(f"[fit] {rhat.shape[0]} x {rhat.shape[1]} x {rhat.shape[2]} coefficients, "
          f"{56 + 3} parameters", flush=True)
    fit = fit_model_error(rhat, sigma2_m, steps=a.steps)

    eta2 = fit["eta2"]
    print(f"\n  tau0 {float(fit['tau0']):.5f}   m0 {float(fit['m0']):.3f}   "
          f"p {float(fit['p']):.3f}")
    print(f"  median sigma^2_m {float(sigma2_m.median()):.3e}   "
          f"median eta^2 at m=1 {float(eta2[:, 0].median()):.3e}   "
          f"at m={a.modes} {float(eta2[:, -1].median()):.3e}")
    print(f"  model error dominates photon noise by "
          f"{float((eta2 / sigma2_m).median()):.1f}x at the median coefficient")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_covariance(fit | {"sigma_c": torch.tensor(sigma_c, dtype=torch.float32),
                           "n_phase": torch.tensor(a.phases)}, a.out)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()

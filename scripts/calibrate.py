#!/usr/bin/env python3
"""Fit the instrument against the real curves of the public models, whose shapes are released.

What is fitted, all at once and all by gradient: the albedo rho, the source radius delta, the
camera distance, the intensity threshold tau_i, the per-curve pedestal, the per-curve model
error eta, the sensor chain (PSF width, vignetting, OETF knots, saturation), and one start
phase psi0 per public body. The thresholds pass a gradient through the coarea formula, the
rest through the rendering itself. The objective is the Gaussian log-likelihood of the real
curves given the rendered ones,

    sum over present curves and phases of  (pred - real)^2 / s^2 + log s^2,
    s_c^2 = sigma_c^2 + eta_c^2,

with sigma_c the measurement noise of each curve, estimated from its own high-frequency
content at the files' native frame rate (hac26.noise). The log term is what stops the fit
from explaining every residual by a larger eta, and eta is where the A/B mounting mismatch
between the two columns of a geometry belongs.

psi0 is first found by a search over whole-frame shifts of the rendered curves against the
data, within an eighth of a turn either way, then refined with everything else. Its fitted
values are the check on hac26.conventions.PSI0: if they agree with each other and differ
from PSI0, PSI0 is wrong.

The released meshes are decimated to TRUTH_FACES faces before rendering; the interreflection
runs on the operator's usual patches.

Writes the Instrument to models/instrument_calibration.pt, and the fitted psi0 per body with
the per-geometry residual report to models/instrument_calibration.json. The residual at the
true shape divided by the noise, per geometry, is the number that says whether the forward
model reproduces the organisers' processing; everything downstream rests on it.
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

from hac26.conventions import PUBLIC_MODELS, SENSE, cameras, psi_grid   # noqa: E402
from hac26.data_io import (N_CAMS, load_model_curves, native_sigma,    # noqa: E402
                           public_stl)
from hac26.forward.mesh.exact import (ExactForward, RenderConfig, decimate, normalise,   # noqa: E402
                                      normalise_vjp)
from hac26.forward.mesh.instrument import Instrument                  # noqa: E402
from hac26.noise import ab_mismatch                                   # noqa: E402
from hac26.shapes import rescale_touch_z                              # noqa: E402
from hac26.stl_io import load_stl                                     # noqa: E402

TRUTH_FACES = 20000       # faces the released meshes are decimated to before rendering
PSI0_SEARCH = 1.0 / 8.0   # the start phase is searched within this fraction of a turn each way
OUT_INSTRUMENT = "models/instrument_calibration.pt"
OUT_REPORT = "models/instrument_calibration.json"


def load_truth(data_dir: str, model: int, device: str):
    """The released mesh of a public model, posed and decimated, as torch tensors."""
    v, f = load_stl(public_stl(data_dir, model))
    v = rescale_touch_z(v, f)
    v, f = decimate(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64), TRUTH_FACES)
    return (torch.tensor(v, dtype=torch.float32, device=device),
            torch.tensor(f, dtype=torch.long, device=device))


def load_data(data_dir: str, model: int, phases: int, device: str):
    """The real mean-normalised curves (N_CAMS, 2, P), which geometries are present (N_CAMS,)
    and the measured noise per curve (N_CAMS, 2).

    The noise comes from the high-frequency content of each curve at the files' own frame
    rate, not from the difference of the two columns of a geometry: those two columns are
    separate recordings of the body in its two mountings, so their difference is dominated by
    the A/B mismatch and runs 3-20x the actual noise (hac26.noise). That difference is
    reported beside sigma as a diagnostic and is left for eta to absorb.
    """
    d = load_model_curves(data_dir, model, m=phases)
    pairs = np.stack([d["curves"][:N_CAMS], d["curves"][N_CAMS:]], axis=1)
    present = (d["mask"][:N_CAMS] > 0) & (d["mask"][N_CAMS:] > 0)
    sigma = native_sigma(d).reshape(2, N_CAMS).T
    mismatch = ab_mismatch(d["curves"], d["mask"]).reshape(2, N_CAMS).T
    return (torch.tensor(pairs, dtype=torch.float32, device=device),
            torch.tensor(present, device=device),
            torch.tensor(sigma, dtype=torch.float32, device=device),
            torch.tensor(mismatch, dtype=torch.float32, device=device))


def initial_psi0(fwd: ExactForward, verts, faces, real, present, sigma) -> float:
    """The whole-frame shift of the rendered curves that fits the data best, as a start
    phase. Shifting psi0 by one grid step moves every curve by one frame, so one rendering
    serves every candidate."""
    P = real.shape[-1]
    pred = normalise(fwd.raw_curves(verts, faces, psi0=0.0))
    best, best_j = float("inf"), 0
    for j in range(-int(P * PSI0_SEARCH), int(P * PSI0_SEARCH) + 1):
        r = (torch.roll(pred, -j, dims=-1) - real) / sigma[..., None]
        m = float((r ** 2)[present].mean())
        if m < best:
            best, best_j = m, j
    return float(SENSE * 2.0 * np.pi * best_j / P)


def nll(pred, real, present, sigma, eta):
    """The Gaussian negative log-likelihood per present curve and phase, and its derivative
    with respect to the prediction."""
    s2 = sigma ** 2 + eta ** 2
    r = pred - real
    n = float(present.sum()) * real.shape[-1]
    loss = (((r ** 2 / s2[..., None]) + torch.log(s2)[..., None]) * present[:, None, None]).sum() / n
    cot = 2.0 * r / s2[..., None] * present[:, None, None] / n
    return loss, cot


def likelihood_cotangent(raw, body, eta):
    """The cotangent on the unnormalised curves: the likelihood's derivative with respect to
    the normalised curves, taken back through the normalisation."""
    _, cot = nll(normalise(raw), body["real"], body["present"], body["sigma"], eta)
    return normalise_vjp(raw, cot)


def residual_report(pred, real, present, sigma, eta) -> dict:
    """RMS residual per geometry over the phases, divided by the noise alone and by the total
    scale sqrt(sigma^2 + eta^2), for the intensity and the binary curves."""
    r = (pred - real)
    per_sigma = (r / sigma[..., None]).pow(2).mean(-1).sqrt()
    per_s = (r / torch.sqrt(sigma ** 2 + eta ** 2)[..., None]).pow(2).mean(-1).sqrt()
    keep = present.cpu().numpy()
    out = {}
    for k, name in enumerate(("intensity", "binary")):
        a = np.where(keep, per_sigma[:, k].cpu().numpy(), np.nan)
        b = np.where(keep, per_s[:, k].cpu().numpy(), np.nan)
        out[name] = {"per_sigma": a.tolist(), "per_s": b.tolist()}
    return out


def print_report(model: int, rep: dict) -> None:
    """One row per camera kind, one column per azimuth, for each curve type and each
    denominator; nothing is aggregated except the medians on the last line."""
    cams = cameras()
    kinds = ("hor_a", "hor_b", "top", "bottom")
    azimuths = sorted({c.azimuth_deg for c in cams})
    print(f"  model {model}:" + " " * 22 + " ".join(f"{z:>6.0f}" for z in azimuths))
    for name in ("intensity", "binary"):
        for denom in ("per_sigma", "per_s"):
            vals = np.asarray(rep[name][denom])
            for kind in kinds:
                row = [vals[i] for i, c in enumerate(cams) if c.kind == kind]
                print(f"    {name:<9} /{denom[4:]:<5} {kind:<7} "
                      + " ".join(f"{v:>6.1f}" for v in row))
        a = np.asarray(rep[name]["per_sigma"]); b = np.asarray(rep[name]["per_s"])
        print(f"    {name:<9} median /sigma {np.nanmedian(a):.2f}, median /s "
              f"{np.nanmedian(b):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", type=int, default=48)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--out", default=OUT_INSTRUMENT)
    ap.add_argument("--report", default=OUT_REPORT)
    render = RenderConfig()
    ap.add_argument("--phase-chunk", type=int, default=render.phase_chunk,
                    help="phases per rendering batch; with --geom-chunk it sets the GPU "
                         "memory the rendering takes, not the result")
    ap.add_argument("--geom-chunk", type=int, default=render.geom_chunk,
                    help="geometries per rendering batch")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    inst = Instrument().to(dev)
    render = RenderConfig(phase_chunk=a.phase_chunk, geom_chunk=a.geom_chunk)
    fwd = ExactForward(inst, psi_grid(a.phases), render, device=dev)
    fit_params = [p for n, p in inst.named_parameters() if n != "raw_eta"]

    bodies = {}
    for M in PUBLIC_MODELS:
        t0 = time.time()
        verts, faces = load_truth(a.data_dir, M, dev)
        real, present, sigma, mismatch = load_data(a.data_dir, M, a.phases, dev)
        with torch.no_grad():
            psi0 = initial_psi0(fwd, verts, faces, real, present, sigma)
        bodies[M] = dict(verts=verts, faces=faces, real=real, present=present, sigma=sigma,
                         psi0=torch.tensor(psi0, device=dev, requires_grad=True))
        print(f"  model {M}: {len(faces)} faces, {int(present.sum())}/{N_CAMS} geometries, "
              f"noise median {float(sigma.median()):.4f} (A/B mismatch "
              f"{float(mismatch.median()):.4f}, {float(mismatch.median()/sigma.median()):.0f}x), "
              f"start phase {np.degrees(psi0):+.1f} deg ({time.time()-t0:.0f}s)", flush=True)

    opt = torch.optim.Adam([{"params": fit_params + [inst.raw_eta]},
                            {"params": [b["psi0"] for b in bodies.values()], "lr": a.lr / 10}],
                           lr=a.lr)
    print(f"[fit] {a.steps} steps over {len(bodies)} bodies at {a.phases} phases", flush=True)
    for step in range(a.steps):
        opt.zero_grad()
        total = 0.0
        for M, b in bodies.items():
            eta = inst.eta.reshape(2, N_CAMS).T
            # the curves and the gradient of the likelihood with respect to every instrument
            # parameter and this body's start phase, through the rendering
            raw, _, grads = fwd.vjp(
                b["verts"], b["faces"], lambda r: likelihood_cotangent(r, b, eta.detach()),
                psi0=b["psi0"], params=fit_params + [b["psi0"]])
            for p, g in zip(fit_params + [b["psi0"]], grads):
                p.grad = g if p.grad is None else p.grad + g
            # eta enters only through the likelihood, so its gradient is direct
            loss, _ = nll(normalise(raw), b["real"], b["present"], b["sigma"], eta)
            loss.backward()
            total += float(loss)
        opt.step()
        if step % 10 == 0 or step == a.steps - 1:
            print(f"  step {step:>4}  -logL {total / len(bodies):.4f}  {inst.summary()}; "
                  f"psi0 " + ", ".join(f"{np.degrees(float(b['psi0'])):+.1f}"
                                       for b in bodies.values()) + " deg", flush=True)

    print("\n[report] RMS residual at the true shape, per geometry (azimuth:value)")
    print("    /sigma  against the measurement noise alone")
    print("    /s      against sqrt(sigma^2 + eta^2), eta being the model error the fit admits")
    report = {"psi0_deg": {}, "residual": {}, "instrument": inst.summary()}
    with torch.no_grad():
        eta = inst.eta.reshape(2, N_CAMS).T
        for M, b in bodies.items():
            pred = normalise(fwd.raw_curves(b["verts"], b["faces"], psi0=float(b["psi0"])))
            rep = residual_report(pred, b["real"], b["present"], b["sigma"], eta)
            print_report(M, rep)
            report["residual"][M] = rep
            report["psi0_deg"][M] = float(np.degrees(float(b["psi0"])))
    print(f"  fitted: {inst.summary()}")
    print("  start phases: " + ", ".join(f"model {M} {v:+.2f} deg"
                                          for M, v in report["psi0_deg"].items()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    inst.save(a.out)
    Path(a.report).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out} and {a.report}")


if __name__ == "__main__":
    main()

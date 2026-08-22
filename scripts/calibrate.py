#!/usr/bin/env python3
"""-- fit the calibration against the real curves of models 1, 2 and 3.

STRUCTURE OF THE FIT, and why it is affordable at all. rho and the source radius enter
through the radiosity solve, so changing them means re-rendering. Everything else -- the
sensor chain, the pedestals, psi0 -- acts on the VALUE IMAGE and downstream
of it. So the radiance images are rendered once and cached, and the remaining parameters
are fitted on the cache.

psi0, the per-body phase offset, is NOT fitted here and is not applied: every call leaves it
at zero.

Radiosity runs on a decimated mesh, necessarily -- a dense form-factor matrix on a full
ground-truth mesh does not fit in memory. Its per-facet radiance is transferred by nearest
centroid, so the SILHOUETTE -- which is what the binary channel measures -- still comes from
the original geometry.

Reports residual-to-noise per geometry and never aggregates it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.calibrate import body_radiance, decimate           # noqa: E402
from hac26.conventions import cameras, psi_grid               # noqa: E402
from hac26.data_io import load_model_curves                   # noqa: E402
from hac26.noise import sigma_from_replicates                 # noqa: E402
from hac26.forward.mesh.radiosity import facet_geometry                    # noqa: E402
from hac26.forward.mesh.raster import Rasteriser           # noqa: E402
from hac26.forward.mesh.sensor import SensorModel                          # noqa: E402
from hac26.shapes import rescale_touch_z                      # noqa: E402
from hac26.stl_io import load_stl                             # noqa: E402


def transfer_radiance(full_v, full_f, dec_v, dec_f, L_dec):
    """Map per-facet radiance from the decimated mesh onto the full mesh by nearest centroid."""
    from scipy.spatial import cKDTree
    cf, _, _ = facet_geometry(full_v, full_f)
    cd, _, _ = facet_geometry(dec_v, dec_f)
    idx = cKDTree(cd).query(cf)[1]
    return idx


def flat_arrays(verts, faces):
    """Per-face vertex duplication so each facet can carry a constant radiance."""
    v = verts[faces].reshape(-1, 3).astype(np.float32)
    f = np.arange(len(v), dtype=np.int32).reshape(-1, 3)
    return v, f


def render_body(model: int, n_phase: int, res, ss, rho, delta, device="cuda",
                raster_faces: int | None = None):
    """Cached radiance images: (28 cameras, n_phase, H, W) at supersampled resolution."""
    v, f = load_stl(f"data/raw/AsteroidModel0{model}_shape_public/asteroid{model}.stl")
    v = rescale_touch_z(v)
    if raster_faces and len(f) > raster_faces:
        v, f = decimate(v, f, raster_faces)
    psi = psi_grid(n_phase)
    dv, df, L = body_radiance(v, f, rho=rho, delta_rad=delta, psi=psi, target_faces=800)
    idx = transfer_radiance(v, f, dv, df, L)
    fv, ff = flat_arrays(v, f)
    fv_t = torch.tensor(fv, device=device)
    ff_t = torch.tensor(ff, device=device)
    ras = Rasteriser(res[0], res[1], ss, device=device)
    ext = float(np.abs(v).max())
    fov = 2.0 * np.arctan(1.6 * ext / 8.0)
    cams = cameras()
    out = torch.zeros(28, n_phase, ras.resolution[0], ras.resolution[1],
                      dtype=torch.float32, device=device)
    eyes = np.stack([np.asarray(c.v) for c in cams]) * 8.0
    for j in range(n_phase):
        vr = torch.tensor(np.repeat(L[j][idx], 3).astype(np.float32), device=device)
        img, _ = ras.render_batch(fv_t, ff_t, vr, eyes, fov_y_rad=fov)   # all 28 in one call
        out[:, j] = img
    return out, ras, fov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", type=int, default=24)
    ap.add_argument("--height", type=int, default=108)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--rho", type=float, default=0.85)
    ap.add_argument("--delta-deg", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--raster-faces", type=int, default=120000)
    ap.add_argument("--out", default="calibration.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    sensor = SensorModel(quantise=True).to(dev)
    # The two thresholds are CONSTANTS, not fitted. Both enter only through a hard
    # comparison, `val > tau`, whose derivative is zero everywhere it is defined, so no
    # gradient reaches them and an optimiser holds them at their initial values. They were
    # in the parameter group and had no effect there. Fitting them needs a relaxation of the
    # comparison, which is a modelling choice and not made here.
    raw_tau_i = torch.full((56,), -2.0, device=dev)
    raw_tau_b = torch.full((56,), -1.0, device=dev)
    # The pedestal IS fitted: it shifts the value image before the reduction, so the
    # intensity channel is differentiable in it even though the mask is not.
    pedestal = torch.zeros(56, device=dev, requires_grad=True)
    # eta is the FITTED residual model error. Fixing it by hand silently sets the
    # denominator of every reported ratio -- pick it too large and it swamps the measured
    # sigma and the numbers stop being residual-to-NOISE at all. Per curve, in log space so
    # it stays positive.
    raw_eta = torch.full((56,), float(np.log(np.expm1(0.02))), device=dev,
                         requires_grad=True)
    opt = torch.optim.Adam([{"params": sensor.parameters()},
                            {"params": [pedestal, raw_eta]}],
                           lr=0.03)

    bodies = {}
    for M in (1, 2, 3):
        print(f"[render] model {M} ...", flush=True)
        imgs, ras, fov = render_body(M, a.phases, (a.height, a.width), a.ss,
                                     a.rho, np.radians(a.delta_deg), dev, a.raster_faces)
        d = load_model_curves("dataset/raw", M, m=a.phases, use_blender=False)
        keep = d["mask"] > 0
        real = torch.tensor(d["curves"], dtype=torch.float32, device=dev)
        real = real / real.mean(1, keepdim=True).clamp_min(1e-9)
        sig = sigma_from_replicates(
            (d["curves"] / np.maximum(d["curves"].mean(1, keepdims=True), 1e-9)), d["mask"])
        bodies[M] = dict(imgs=imgs.cpu(), real=real, keep=keep,
                         sigma=torch.tensor(sig, dtype=torch.float32, device=dev), ras=ras)
        print(f"  model {M}: images {tuple(imgs.shape)}, "
              f"radiance max {float(imgs.max()):.4f}", flush=True)

    cg, rg = None, None

    def curves_for(M, chunk: int = 4):
        """Reduced curves for one body, chunked over cameras.

        The cached radiance lives on the CPU and only a few cameras at a time are moved to
        the GPU. All three bodies at every camera and phase does not fit on the card
        alongside the sensor chain's intermediates.
        """
        b = bodies[M]
        C, P, H, W = b["imgs"].shape
        cos_off, rad = b["ras"].pixel_geometry(np.radians(20.0))
        ti = torch.sigmoid(raw_tau_i)
        tb = torch.sigmoid(raw_tau_b)
        inten, binar = [], []
        for c0 in range(0, C, chunk):
            im = b["imgs"][c0:c0 + chunk].to(dev, non_blocking=True)
            n = im.shape[0]
            val = sensor(im.reshape(n * P, H, W), cos_off.expand(n * P, H, W),
                         rad.expand(n * P, H, W), supersample=b["ras"].ss)
            val = val.reshape(n, P, val.shape[-2], val.shape[-1])
            val = val + pedestal[c0:c0 + n].reshape(n, 1, 1, 1)
            inten.append((val * (val > ti[c0:c0 + n].reshape(n, 1, 1, 1))).sum(dim=(-2, -1)))
            binar.append((val > tb[28 + c0:28 + c0 + n].reshape(n, 1, 1, 1))
                         .to(val.dtype).sum(dim=(-2, -1)))
            del im, val
        cur = torch.cat(inten + binar, 0)
        return cur / cur.mean(1, keepdim=True).clamp_min(1e-9)

    print("[fit] optimising sensor + pedestals + eta on cached radiance", flush=True)
    for step in range(a.steps):
        opt.zero_grad()
        tot = 0.0
        for M in (1, 2, 3):                      # backward per body, so only one is resident
            b = bodies[M]
            pred = curves_for(M)
            k = torch.tensor(b["keep"], device=dev)
            eta = torch.nn.functional.softplus(raw_eta).reshape(-1, 1)
            s2 = b["sigma"].reshape(-1, 1) ** 2 + eta ** 2
            # the log|s| term is what stops the fit from simply inflating eta
            loss = ((((pred - b["real"]) ** 2 / s2) + torch.log(s2))[k]).mean() / 3.0
            loss.backward()
            tot += float(loss)
            del pred, loss
            torch.cuda.empty_cache()
        opt.step()
        if step % 20 == 0 or step == a.steps - 1:
            print(f"  step {step:>4}  loss {tot:.5f}", flush=True)

    print("\n[report] residual PER GEOMETRY (never aggregated)")
    print("  two denominators, because they answer different questions:")
    print("    /sigma      : residual against MEASUREMENT noise alone")
    print("    /s          : against sqrt(sigma^2 + eta^2) with eta FITTED, i.e. against")
    print("                  measurement noise plus the model error the fit itself admits")
    report = {}
    eta_f = torch.nn.functional.softplus(raw_eta).detach()
    for M in (1, 2, 3):
        b = bodies[M]
        with torch.no_grad():
            pred = curves_for(M)
        res = (pred - b["real"])
        rn_sig = torch.sqrt(((res / b["sigma"].reshape(-1, 1)) ** 2).mean(1)).cpu().numpy()
        s_full = torch.sqrt(b["sigma"] ** 2 + eta_f ** 2).reshape(-1, 1)
        rn_s = torch.sqrt(((res / s_full) ** 2).mean(1)).cpu().numpy()
        keep = b["keep"]
        rn_sig = np.where(keep, rn_sig, np.nan)
        rn_s = np.where(keep, rn_s, np.nan)
        report[M] = {"per_sigma": rn_sig.tolist(), "per_s": rn_s.tolist()}
        az = [c.azimuth_deg for c in cameras()]
        print(f"  model {M}:")
        for half, name in ((slice(0, 28), "intensity"), (slice(28, 56), "binary")):
            print(f"    {name:<9} /sigma " + " ".join(
                f"{a_:.0f}:{v:.1f}" for a_, v in zip(az[::4], rn_sig[half][::4])))
            print(f"    {'':<9} /s     " + " ".join(
                f"{a_:.0f}:{v:.1f}" for a_, v in zip(az[::4], rn_s[half][::4])))
        print(f"    median /sigma {np.nanmedian(rn_sig):.2f}   median /s "
              f"{np.nanmedian(rn_s):.2f}   fitted eta median "
              f"{float(eta_f.median()):.4f}  (sigma median "
              f"{float(b['sigma'].median()):.4f})")
    torch.save({"sensor": sensor.state_dict(), "raw_tau_i": raw_tau_i.detach(),
                "raw_tau_b": raw_tau_b.detach(), "pedestal": pedestal.detach(),
                "raw_eta": raw_eta.detach(), "rho": a.rho, "delta_deg": a.delta_deg},
               "models/instrument_calibration.pt")
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out} and models/instrument_calibration.pt")


if __name__ == "__main__":
    main()

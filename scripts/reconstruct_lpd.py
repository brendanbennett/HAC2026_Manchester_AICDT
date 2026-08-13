#!/usr/bin/env python3
"""Reconstruct a competition model with the trained flow, then finish it.

Six Euler steps with the operator re-applied at each, several independent draws from x0, and
the metric medoid of those draws as the answer. The medoid rather than the mean because
thresholding a mean of occupancy grids erases concavity: a feature present in most samples but
at slightly different places averages to below the threshold everywhere.

Dice is measured on a voxel grid after posing both meshes with rescale_touch_z, so they are
compared in the frame the challenge defines.
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

from hac26.conventions import cameras, psi_grid                        # noqa: E402
from hac26.data_io import load_model_curves                            # noqa: E402
from hac26.field import ImplicitBody, apply_constraints, extract_mesh  # noqa: E402
from hac26.solvers.lpd_flow import N_MODES, N_STEPS, LPDFlow         # noqa: E402
from hac26.solvers.output import (export_stl, metric_medoid, planar_snap,      # noqa: E402
                          ransac_planes, restore_constraints)
from hac26.covariance import load_covariance, whitened_misfit          # noqa: E402
from hac26.recon import dice                                           # noqa: E402
from hac26.scoring.voxel import occupancy as _occupancy           # noqa: E402
from hac26.shapes import mesh_support, rescale_touch_z                 # noqa: E402
from hac26.forward.learned_surrogate import Surrogate                                  # noqa: E402
from train_lpd import (curves_from_code, curves_from_mesh,             # noqa: E402
                       load_decoder, set_code)                        # noqa: E402

CYLINDER_R = {1: 1.12, 2: 1.42, 3: 0.88, 4: 1.475, 5: 1.22,
              6: 0.925, 7: 1.205, 8: 1.24, 9: 0.67, 10: 3.95}
PUBLIC = {1: "AsteroidModel01_shape_public/asteroid1.stl",
          2: "AsteroidModel02_shape_public/asteroid2.stl",
          3: "AsteroidModel03_shape_public/asteroid3.stl"}


def data_modes(curves56: np.ndarray, n_modes: int):
    """(56, P) real curves -> (1, 28, 2, M) complex Fourier content, intensity then binary."""
    g = np.stack([curves56[:28], curves56[28:]], axis=1)        # (28, 2, P)
    f = torch.fft.rfft(torch.tensor(g, dtype=torch.float32), dim=-1)
    return f[None, ..., 1:n_modes + 1]


def geom_tag_and_mask(mask56: np.ndarray):
    tag = torch.zeros(1, 28, 4)
    for i, cam in enumerate(cameras()):
        tag[0, i] = torch.tensor([np.cos(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.elevation_deg)), 1.0])
    # a geometry counts as present only if BOTH its channels are
    m = torch.tensor((mask56[:28] > 0) & (mask56[28:] > 0), dtype=torch.float32)[None]
    return tag, m


def support_from_convex(stl: str) -> torch.Tensor:
    """h for a competition body, taken from the convex stage's own reconstruction.

    The flow generates the token correction, not h.
    It does not follow that h is a sphere: it has to come from somewhere, and the convex
    stage already predicts it from these same 56 curves. Evaluated on the design normals,
    which is the basis ConvexCore stores its support in.
    """
    import trimesh
    m = trimesh.load(stl, process=False)
    n = ImplicitBody(radius=1.0).core.n.detach().cpu().numpy()
    h = mesh_support(np.asarray(m.vertices), n)
    return torch.tensor(np.maximum(h, 1e-3), dtype=torch.float32)


def make_resid_fn(g_dat, surro, psi, M, radius, support=None):
    """resid_fn(code) -> (residual, convex complement), with the operator actually applied."""
    def fn(x):
        preds = []
        for b in range(len(x)):
            cur = curves_from_code(x[b].detach(), radius, surro, psi, support=support)
            preds.append(torch.zeros(28, 2, len(psi)) if cur is None else cur)
        g_cur = torch.fft.rfft(torch.stack(preds), dim=-1)[..., 1:M + 1]
        r = g_dat.expand_as(g_cur) - g_cur
        r_perp = r.clone()
        r_perp[..., :4] = 0            # the convex operator explains the low orders best
        B = len(x)
        feats = torch.zeros(B, 28, N_MODES, 6)
        perp = torch.zeros(B, 28, N_MODES, 6)
        for ch in range(2):
            feats[:, :, :M, 2 * ch] = r[:, :, ch].real
            feats[:, :, :M, 2 * ch + 1] = r[:, :, ch].imag
            perp[:, :, :M, 2 * ch] = r_perp[:, :, ch].real
            perp[:, :, :M, 2 * ch + 1] = r_perp[:, :, ch].imag
        feats[:, :, :M, 4] = g_dat[:, :, 0].real.expand(B, -1, -1)
        feats[:, :, :M, 5] = g_dat[:, :, 1].real.expand(B, -1, -1)
        return feats, perp
    return fn


def decode(code, radius, res=64, misfit_fn=None, eta=None, support=None):
    """Token code -> constrained mesh: field extraction, constraints, then the planar snap.

    The snap is gated on the data rather than applied because a plane was found: RANSAC will
    always return something on a noisy mesh, and snapping to it flattens curvature that was
    never measured. A plane survives only if flattening to it does not raise the residual past
    the model error from the calibration.
    """
    body = ImplicitBody(radius=radius)
    body.core.set_support(torch.full((body.core.n.shape[0],), 0.8 * radius)
                          if support is None else support)
    load_decoder(body)
    set_code(body, code)
    v, f = extract_mesh(lambda y: body(y), radius * 1.6, res=res, device="cpu")
    if len(f) < 8:
        return None, None, 0
    v = apply_constraints(v, radius)
    kept = 0
    planes = ransac_planes(v, f)
    if len(planes):
        v, kept = planar_snap(v, f, planes,
                              misfit_fn=(None if misfit_fn is None
                                         else lambda w: misfit_fn(w, f)),
                              eta=eta)
        v = restore_constraints(v, radius)
    return v, f, kept


def occupancy(v, f, n=64, extent=None):
    return _occupancy(v, f, n, extent or float(np.abs(v).max()) * 1.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--ckpt", default="runs/lpd_flow.pt")
    ap.add_argument("--surrogate", default="runs/surrogate.pt")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--phases", type=int, default=96)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--support-from", default=None,
                    help="STL whose support function supplies h; defaults to the convex "
                         "stage's reconstruction of this model")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    R = CYLINDER_R[a.model]
    psi = psi_grid(a.phases)
    M = min(N_MODES, a.phases // 2)

    gdev = "cuda" if torch.cuda.is_available() else "cpu"
    surro = Surrogate(width=96, modes=8, blocks=3)
    surro.load_state_dict(torch.load(a.surrogate, map_location="cpu"))
    surro = surro.to(gdev).eval()

    net = LPDFlow()
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    net.eval()

    sup_stl = a.support_from or f"results/convex/Asteroid{a.model:02d}.stl"
    support = support_from_convex(sup_stl)
    print(f"  h from {sup_stl}: {float(support.min()):.3f}-{float(support.max()):.3f}",
          flush=True)

    d = load_model_curves(a.data_dir, a.model, m=a.phases)
    g_dat = data_modes(d["curves"], M)
    tag, mask = geom_tag_and_mask(d["mask"])
    print(f"model {a.model}: R = {R}, {int(mask.sum())}/28 geometries present", flush=True)

    t0 = time.time()
    codes = net.sample(make_resid_fn(g_dat, surro, psi, M, R, support=support),
                       tag.expand(a.samples, -1, -1), mask.expand(a.samples, -1),
                       batch=a.samples)
    print(f"  {a.samples} draws x {N_STEPS} steps in {time.time()-t0:.0f}s", flush=True)

    # The snap is gated on the data: the misfit of a candidate mesh against the real curves,
    # with the per-curve model error from the calibration as the tolerance it may not exceed.
    cov_path = Path("models/data_covariance.pt")
    COV = load_covariance(str(cov_path)) if cov_path.exists() else None
    if COV is None:
        print("  no fitted covariance yet; the snap gate falls back to an unweighted RMS",
              flush=True)
    real = torch.tensor(d["curves"], dtype=torch.float32)
    real_g = torch.stack([real[:28], real[28:]], dim=1)              # (28, 2, P)
    eta = float(torch.nn.functional.softplus(
        torch.load("models/instrument_calibration.pt", map_location="cpu",
                   weights_only=False)["raw_eta"]).mean())

    def misfit(w, faces):
        """Whitened by the measured covariance, which is the only weight in play here.

        An unweighted curve-space RMS would weight every curve and every rotation order
        equally, which is a prior on the data that was never measured.
        """
        c = curves_from_mesh(np.asarray(w), np.asarray(faces), surro, psi)
        pred56 = torch.cat([c[:, 0], c[:, 1]], dim=0)
        if COV is None:
            return float(((c - real_g) ** 2).mean().sqrt())
        return float(whitened_misfit(pred56, real, COV["s2"]))

    meshes, occs, snaps = [], [], []
    for i in range(a.samples):
        v, f, kept = decode(codes[i], R, res=a.res, misfit_fn=misfit, eta=eta,
                            support=support)
        if v is None:
            print(f"  draw {i}: degenerate, dropped", flush=True); continue
        meshes.append((v, f)); occs.append(occupancy(v, f)); snaps.append(kept)
    if not meshes:
        raise SystemExit("every draw was degenerate")
    k = metric_medoid(occs)
    v, f = meshes[k]
    print(f"  planes accepted per draw (eta = {eta:.4f}): {snaps}", flush=True)
    print(f"  medoid = draw {k} of {len(meshes)}; "
          f"mean pairwise Dice {np.mean([dice(occs[k], o) for o in occs]):.4f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    info = export_stl(a.out, v, f)
    res = {"model": a.model, "radius": R, "draws": len(meshes), "medoid": int(k),
           "spread": float(np.mean([dice(occs[k], o) for o in occs])),
           "eta": eta, "planes_accepted": snaps, **info}

    if a.model in PUBLIC:
        import trimesh
        t = trimesh.load(Path(a.data_dir) / PUBLIC[a.model], process=False)
        tv = rescale_touch_z(np.asarray(t.vertices))
        rv = rescale_touch_z(v)
        e = max(float(np.abs(tv).max()), float(np.abs(rv).max())) * 1.05
        res["dice"] = float(dice(occupancy(tv, np.asarray(t.faces), 128, e),
                                 occupancy(rv, f, 128, e)))
        print(f"  DICE vs truth: {res['dice']:.4f}", flush=True)

    print(json.dumps(res))
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

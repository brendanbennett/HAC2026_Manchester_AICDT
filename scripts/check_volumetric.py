#!/usr/bin/env python3
"""Confirm the two non-convex channels of the volumetric forward model.

They are independent and fail differently.

  Self-occlusion (view side). T = Phi_s(phi) drives the contribution of anything behind the
  first crossing to zero. Checked by placing a small lobe entirely behind a larger one, so
  every ray reaching the far lobe must first cross the near one, and measuring the far lobe's
  share of the total rendering weight.

  Cast shadow (illumination side). Checked through
  D_shadow = sum over pixels of [ L(shadows=False) - L(shadows=True) ], which must be
  non-negative, vanish at azimuth 0 elevation 0 where the view and source directions
  coincide, and vary with psi for a non-convex body.

The test body for the psi check must NOT be axisymmetric about z: a body of revolution about
the rotation axis gives an identical image at every phase, so a flat D_shadow(psi) would be
correct for it and would prove nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forward_models import sdf_volumetric as V          # noqa: E402
from forward_models.common import camera_basis          # noqa: E402
from hac26.conventions import S_LAB, cameras, to_body   # noqa: E402


def two_lobes(r_a: float, r_b: float, sep: float, axis=(0.0, 0.0, 1.0)):
    """Union of two spheres of radii r_a (at +axis*sep/2) and r_b (at -axis*sep/2)."""
    a = torch.tensor(axis, dtype=torch.float32); a = a / a.norm()
    ca, cb = 0.5 * sep * a, -0.5 * sep * a

    def phi(x):
        da = (x - ca.to(x.device)).norm(dim=-1) - r_a
        db = (x - cb.to(x.device)).norm(dim=-1) - r_b
        return torch.minimum(da, db)
    return phi, ca, cb


def weight_split(phi, view, w, c_near, c_far, res=48, extent=1.3):
    """Total rendering weight, split by which lobe each sample sits nearer to."""
    ex, ey = camera_basis(view)
    a = torch.linspace(-extent, extent, res)
    gx, gy = torch.meshgrid(a, a, indexing="ij")
    o = (gx.reshape(-1, 1) * ex + gy.reshape(-1, 1) * ey) + 2.0 * extent * view
    d = (-view)[None, :].expand(o.shape[0], 3)

    t_in = V.shell_entry(o, d, phi, w)
    n_e = V.field_normal(phi, o + t_in[..., None] * d, create_graph=False)
    cosang = (n_e * d).sum(-1).abs().clamp_min(0.05)
    dt = (2.0 * V.SHELL * w) / cosang / V.N_NODES

    log_T = torch.zeros(o.shape[0])
    w_near = torch.zeros(o.shape[0]); w_far = torch.zeros(o.shape[0])
    phi_prev = phi(o + t_in[..., None] * d)
    for i in range(V.N_NODES):
        x = o + (t_in + (i + 1) * dt)[..., None] * d
        phi_cur = phi(x)
        dlog = (torch.nn.functional.logsigmoid(phi_cur / w)
                - torch.nn.functional.logsigmoid(phi_prev / w)).clamp(max=0.0)
        wgt = torch.exp(log_T) * (-torch.expm1(dlog))
        far = (x - c_far).norm(dim=-1) < (x - c_near).norm(dim=-1)
        w_far = w_far + wgt * far
        w_near = w_near + wgt * (~far)
        log_T = log_T + dlog
        phi_prev = phi_cur
    return float(w_near.sum()), float(w_far.sum())


def main():
    torch.manual_seed(0)
    w = 0.02

    print("1.1  self-occlusion (view side)")
    # camera along +z, large lobe at +z (near), small lobe at -z (far and fully behind it)
    phi, c_near, c_far = two_lobes(r_a=0.45, r_b=0.25, sep=0.65, axis=(0.0, 0.0, 1.0))
    print(f"     near lobe r=0.45 at z=+0.325, far lobe r=0.25 at z=-0.325")
    near, far = weight_split(phi, torch.tensor([0.0, 0.0, 1.0]), w, c_near, c_far)
    ratio = far / max(near, 1e-12)
    print(f"     near-lobe weight {near:.4f}   far-lobe weight {far:.4e}   ratio {ratio:.3e}")
    print(f"     {'PASS' if ratio < 1e-3 else 'FAIL'}  (far lobe below 1e-3 of the near lobe)")

    print("\n1.2  cast shadow (illumination side)")
    # lobes along x, so the body is NOT axisymmetric about the rotation axis z
    phi, _, _ = two_lobes(r_a=0.42, r_b=0.34, sep=0.62, axis=(1.0, 0.0, 0.0))
    kw = dict(res=40, supersample=1, bounce=False, extent=1.3)

    cams = list(cameras())
    c0 = cams[0]
    v0 = torch.tensor(np.asarray(c0.v), dtype=torch.float32)
    s0 = torch.tensor(np.asarray(S_LAB), dtype=torch.float32)[None]
    lit = V.render_frame(phi, v0, s0, w, shadows=False, **kw).sum().detach()
    d0 = float(V.shadow_deficit(phi, v0, s0, w, **kw).detach())
    print(f"     camera 0: azimuth {c0.azimuth_deg:.0f}, elevation {c0.elevation_deg:.0f}; "
          f"v.s = {float(v0 @ s0[0]):+.3f}")
    print(f"     D_shadow = {d0:+.6e}   total lit = {float(lit):.4f}   "
          f"relative = {d0/max(float(lit),1e-12):.3e}")

    c8 = cams[8]
    print(f"     camera 8: azimuth {c8.azimuth_deg:.0f}, elevation {c8.elevation_deg:.0f}")
    psis = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ds = []
    for p in psis:
        vb = to_body(np.asarray(c8.v), np.array([p]))[0]
        sb = to_body(np.asarray(S_LAB), np.array([p]))[0]
        ds.append(float(V.shadow_deficit(phi, torch.tensor(vb, dtype=torch.float32),
                                         torch.tensor(sb, dtype=torch.float32)[None],
                                         w, **kw).detach()))
    ds = np.asarray(ds)
    print("     D_shadow(psi):", np.array2string(ds, precision=5))
    print(f"     min {ds.min():+.3e}  "
          f"{'PASS (>= 0)' if ds.min() >= -1e-9 else 'FAIL (negative)'}")
    rel = ds.std() / max(abs(ds.mean()), 1e-12)
    print(f"     std/|mean| over psi = {rel:.4f}  "
          f"{'PASS (varies)' if rel > 1e-3 else 'FAIL (flat in psi)'}")


if __name__ == "__main__":
    main()

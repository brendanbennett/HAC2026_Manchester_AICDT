"""The contract every forward model in this folder obeys, and the two reductions.

A forward model here answers ONE question: given a body and a viewing/lighting geometry,
what are the two numbers the challenge measures? Everything else -- how the body is
parameterised, whether the surface is a mesh, a level set or a density -- is the model's own
business, and the models differ precisely there.

    render(shape, view, sun, **kw) -> (intensity, lit_area)

`view` is omega_c, the direction from the BODY TO THE CAMERA, and `sun` is the direction from
the body to the source. Both are unit vectors already carried into the BODY frame; use
hac26.conventions.to_body to get them there, which is where the rotation sense lives.

THE TWO REDUCTIONS, and why they are not the same integral. The organisers' pipeline
thresholds the image and then

    intensity  I = sum over pixels of val * 1[val > tau_I]        a SUM of values
    lit area   N = count of pixels with  val > tau_B              a COUNT of pixels

so the two differ by more than a constant: I carries the radiance, N carries only the
geometry of the lit region. Writing them as one integral with different weights is the single
most common way to get this wrong.

RADIANCE, NOT RADIANCE x mu. A Lambertian facet's radiance is (rho/pi) mu0 and carries no mu:
the viewing obliquity enters through the PROJECTED AREA of the pixel, mu dA, which the
image-plane sum already supplies. Writing mu0 * mu in the shader double-counts it. With
val = mu0 the two sums come out as

    I = INT mu0+ mu+ dA      the Lambert kernel
    N = INT_lit mu+ dA       the binary kernel, mu+ alone

which is exactly the pair the convex analytic operator uses.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["reduce_frame", "camera_basis", "pixel_rays", "curves_over_psi"]


def reduce_frame(val: torch.Tensor, px: float, tau_i: float = 0.0,
                 tau_b: float = 0.02) -> tuple:
    """The two reductions from a per-pixel value map. `px` is the area of one pixel."""
    inten = (val * (val > tau_i)).sum() * px
    area = (val > tau_b).sum().to(val.dtype) * px
    return inten, area


def camera_basis(view: torch.Tensor) -> tuple:
    """An orthonormal image basis for an orthographic camera looking along -view."""
    up = torch.tensor([0.0, 0.0, 1.0], device=view.device, dtype=view.dtype)
    if abs(float(view @ up)) > 0.99:
        up = torch.tensor([0.0, 1.0, 0.0], device=view.device, dtype=view.dtype)
    ex = torch.cross(up, view, dim=0); ex = ex / ex.norm()
    ey = torch.cross(view, ex, dim=0); ey = ey / ey.norm()
    return ex, ey


def pixel_rays(view: torch.Tensor, extent: float, res: int) -> tuple:
    """Ray origins on a plane in front of the body, all travelling along -view.

    Rays START on the camera side and travel along -view. Starting at -extent*view and
    marching along +view instead renders the FAR surface, which shows up as the near and far
    cameras swapping values -- 0.98 against 0.00 at one camera and 0.05 against 0.73 at its
    opposite. That failure is silent unless you look at both members of an opposed pair.
    """
    ex, ey = camera_basis(view)
    a = torch.linspace(-extent, extent, res, device=view.device, dtype=view.dtype)
    gx, gy = torch.meshgrid(a, a, indexing="ij")
    o = (gx.reshape(-1, 1) * ex + gy.reshape(-1, 1) * ey) + 2.0 * extent * view
    return o, -view, (2.0 * extent / res) ** 2


def curves_over_psi(render_fn, shape, cam_dir, sun_lab, psi, psi0: float = 0.0, **kw):
    """Both curves over a full rotation, for one camera.

    The body spins about z and the source is FIXED in the lab, so both directions are carried
    into the body frame at each phase. Everything else is the model's own.
    """
    from hac26.conventions import to_body
    cam = to_body(np.asarray(cam_dir, dtype=float), np.asarray(psi, dtype=float), psi0)
    sun = to_body(np.asarray(sun_lab, dtype=float), np.asarray(psi, dtype=float), psi0)
    I, N = [], []
    for j in range(len(psi)):
        i_, n_ = render_fn(shape,
                           torch.tensor(cam[j], dtype=torch.float32),
                           torch.tensor(sun[j], dtype=torch.float32), **kw)
        I.append(float(i_)); N.append(float(n_))
    return np.asarray(I), np.asarray(N)

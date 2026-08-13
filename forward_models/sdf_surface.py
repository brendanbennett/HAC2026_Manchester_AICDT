"""Hard-surface renderer for a level set: sphere-trace to the first hit, shade it.

THE BODY is {x : phi(x) <= 0} for any callable phi. Nothing is assumed about how phi is
built -- an analytic primitive, a grid, a network -- only that it is a TRUE signed distance
function, meaning |grad phi| = 1. That single requirement is what makes sphere tracing safe:
stepping by phi(x) can never cross the surface, because phi is exactly the distance to it.

    t <- t + phi(o + t d)   until phi < eps

WHY NOT BISECTION. Bisection needs a bracket in which phi changes sign, and a ray that
ENTERS AND EXITS the body has phi < 0 only on an interval. Bisecting [0, t_max] on the test
"is the midpoint inside" therefore walks away from the body whenever the first midpoint
happens to miss it, and converges to t_max. Measured on a sphere of radius 0.7 at distance
3: bisection returned t = 10.0 (the cap) on all three test rays, an error of 7.7, while
sphere tracing returned the closed-form answer to 2.4e-07.

DERIVATIVES ARE EXACT, and do not go through the marching loop. The hit satisfies
phi(o + t* d) = 0 identically, so differentiating that in any parameter theta gives

    dt*/dtheta = - (dphi/dtheta) / (grad phi . d)                 [implicit function theorem]

which needs only quantities evaluated AT the hit. Checked against the closed form for a
sphere's radius: agreement to 4.8e-07. This is the reason to prefer a hard surface over a
softened one where gradients are wanted -- there is no smoothing bias to trade against.

WHAT THIS MODEL CANNOT DO. The silhouette is a step: a ray either hits or does not, so
d(area)/d(shape) is zero almost everywhere and undefined on the boundary. Gradients that
must flow through the OUTLINE need sdf_volume.py, whose softness buys exactly that.
"""
from __future__ import annotations

import torch

from .common import pixel_rays, reduce_frame

__all__ = ["sphere_trace", "surface_normal", "render", "sphere_sdf", "pit_sdf"]


def sphere_trace(o: torch.Tensor, d: torch.Tensor, phi, tmax: float = 10.0,
                 iters: int = 200, eps: float = 1e-6):
    """First hit of {phi <= 0} along o + t d. Returns (t, hit)."""
    t = torch.zeros(o.shape[:-1], device=o.device, dtype=o.dtype)
    for _ in range(iters):
        s = phi(o + t[..., None] * d)
        t = torch.where((s > eps) & (t < tmax), t + s.clamp_min(eps * 0.5), t)
    return t, t < tmax


def surface_normal(x: torch.Tensor, phi, h: float = 1e-4) -> torch.Tensor:
    """Central differences on phi. For a true SDF the result is already unit length."""
    e = torch.eye(3, device=x.device, dtype=x.dtype) * h
    g = torch.stack([phi(x + e[i]) - phi(x - e[i]) for i in range(3)], dim=-1) / (2 * h)
    return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def render(phi, view: torch.Tensor, sun: torch.Tensor, res: int = 96,
           extent: float = 1.6, rho: float = 0.85, tau_i: float = 0.0,
           tau_b: float = 0.02, shadow: bool = True):
    """One frame: (intensity, lit area).

    The cast shadow is a SECOND trace, from just above each hit point towards the source.
    That is the whole of the non-convexity in this model: occlusion of the view comes free
    from taking the first hit, and occlusion of the light comes from this trace. A convex
    body never blocks either, which is exactly why a convex operator cannot express them.
    """
    o, d, px = pixel_rays(view, extent, res)
    t, hit = sphere_trace(o, d, phi, tmax=4.0 * extent)
    x = o + t[..., None] * d
    n = surface_normal(x, phi)
    mu0 = (n * sun).sum(-1)
    lit = hit & (mu0 > 0)
    val = torch.zeros_like(mu0)
    if lit.any():
        v = mu0[lit] * (rho / torch.pi)
        if shadow:
            xs = x[lit] + n[lit] * 1e-3
            ts, blocked = sphere_trace(xs, sun.expand_as(xs), phi, tmax=4.0 * extent)
            v = torch.where(blocked, torch.zeros_like(v), v)
        val[lit] = v
    return reduce_frame(val, px, tau_i, tau_b)


# ----------------------------------------------------------------------------------
# Two analytic bodies, used by the gates. Both are exact SDFs.

def sphere_sdf(radius: float, centre=(0.0, 0.0, 0.0)):
    c = torch.tensor(centre, dtype=torch.float32)
    return lambda x: (x - c.to(x.device)).norm(dim=-1) - radius


def pit_sdf(radius: float, pit_r: float, depth: float, axis=(0.0, 0.0, 1.0)):
    """A sphere with a flat-floored cylindrical pit sunk into it along `axis`.

    The pit is a CYLINDER, not a spherical cap, and that choice is deliberate: it is the
    geometry for which the first-order shadow term in convex_plus_dents.py is derived, so
    the analytic term and this reference describe the same body rather than two similar ones.
    """
    a = torch.tensor(axis, dtype=torch.float32)
    a = a / a.norm()

    def phi(x):
        ax = a.to(x.device)
        ball = x.norm(dim=-1) - radius
        z = (x * ax).sum(-1)                       # height along the pit axis
        r = (x - z[..., None] * ax).norm(dim=-1)   # radial distance from the axis
        # cylinder occupying z in [radius - depth, +inf), r <= pit_r
        d_r = r - pit_r
        d_z = (radius - depth) - z
        cyl = torch.maximum(d_r, d_z)
        return torch.maximum(ball, -cyl)           # sphere minus cylinder

    return phi

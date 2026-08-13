"""Closed-form, differentiable Dice between convex bodies via their radial functions.

The point of this module is that the challenge metric is *exactly* computable from a
support function -- no voxel grid, no Minkowski solve, no mesh -- and the expression is
differentiable in h. That makes the scoring metric usable directly as a training loss.

Why it is exact.  A convex body given by its support function is the half-space
intersection K = {x : <x,u_n> <= h_n}.  Along the ray t*v (t >= 0) the binding
constraints are those with <v,u_n> > 0, so the boundary is at

    rho(v) = min_{n : <v,u_n> > 0}  h_n / <v,u_n>                                   (1)

which is the radial function of K.  For two convex bodies that both contain the origin
the intersection is convex and star-shaped about it, with radial function
min(rho_A, rho_B).  Volume in polar coordinates is (1/3) * int_{S^2} rho^3 dw, hence

    Dice = 2 |A n B| / (|A| + |B|)
         = 2 int min(rho_A,rho_B)^3 dw / ( int rho_A^3 dw + int rho_B^3 dw )        (2)

Both (1) and (2) are compositions of min, divide and power: differentiable a.e., with
the subgradient flowing to the active (binding) constraint -- which is exactly the one
that moves the surface.  That last property is the whole reason this beats an MSE on h:
MSE spends gradient on inactive directions, where h can be wrong by any amount without
moving the body at all.

The only approximations are the quadrature over the sphere (a Fibonacci lattice, equal
weights) and convexity of the target -- which the pipeline is bounded by regardless.
"""
from __future__ import annotations

import numpy as np


def fibonacci_sphere(n: int) -> np.ndarray:
    """`n` nearly-equal-area directions on S^2 (spherical Fibonacci lattice).

    Equal-area means equal quadrature weights, so every integral below is a plain mean.
    """
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def support_ray_matrix(normals: np.ndarray, rays: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    """M[v,n] = max(<rays_v, normals_n>, 0), the binding-constraint coefficients.

    Non-binding directions are stored as 0 rather than masked out, which is what makes
    the reciprocal form below free of infinities: a zero coefficient simply never wins
    the max, so no sentinel value ever enters the arithmetic.  (Writing (1) directly as
    a min of h/M needs +inf sentinels, and h/inf = 0 would then win the min -- and 1e30
    sentinels overflow fp16 under autocast.  The reciprocal form has neither problem.)
    """
    return np.maximum(rays @ normals.T, 0.0) * (np.abs(rays @ normals.T) > eps)


def radial_from_support(h: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Equation (1) in reciprocal form: rho(v) = 1 / max_n ( M[v,n] / h_n )."""
    return 1.0 / np.max(M * (1.0 / h)[None, :], axis=1)


def dice_from_radial(ra: np.ndarray, rb: np.ndarray) -> float:
    """Equation (2). Exact for convex bodies containing the origin."""
    inter = np.minimum(ra, rb) ** 3
    return float(2.0 * inter.sum() / ((ra ** 3).sum() + (rb ** 3).sum()))


def dice_from_support(ha: np.ndarray, hb: np.ndarray, M: np.ndarray) -> float:
    return dice_from_radial(radial_from_support(ha, M),
                            radial_from_support(hb, M))


# ---------------- torch side: the training loss --------------------------------------
def torch_radial(h, M):
    """Batched (1), reciprocal form: h is (B,N), M is (V,N) -> (B,V).

    The gradient of the max flows to the single binding constraint per ray -- the one
    whose plane the surface actually touches. That is the property an MSE on h lacks.
    """
    return 1.0 / (M[None, :, :] * (1.0 / h)[:, None, :]).max(dim=2).values


def torch_dice(rho_a, rho_b, eps: float = 1e-8):
    """Batched (2) -> (B,). Differentiable in both arguments."""
    inter = torch.minimum(rho_a, rho_b) ** 3
    return 2.0 * inter.sum(1) / (rho_a.pow(3).sum(1) + rho_b.pow(3).sum(1) + eps)


def torch_dice_loss(h_pred, rho_true, M, chunk: int = 0):
    """1 - Dice(body(h_pred), true body), averaged over the batch.

    `rho_true` is precomputed from the ground-truth mesh (it needs no gradient), so the
    target can come from the real mesh rather than its support-grid approximation.
    Set `chunk` to split the ray axis when (B, V, N) will not fit in memory.
    """
    if not chunk:
        return (1.0 - torch_dice(torch_radial(h_pred, M), rho_true)).mean()
    num = 0.0
    den_a = 0.0
    den_b = 0.0
    for i in range(0, M.shape[0], chunk):
        ra = torch_radial(h_pred, M[i:i + chunk])
        rb = rho_true[:, i:i + chunk]
        num = num + torch.minimum(ra, rb).pow(3).sum(1)
        den_a = den_a + ra.pow(3).sum(1)
        den_b = den_b + rb.pow(3).sum(1)
    return (1.0 - 2.0 * num / (den_a + den_b + 1e-8)).mean()


try:  # torch is optional for the numpy-only paths above
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------- exact radial function of a mesh ------------------------------------
def mesh_radial(verts: np.ndarray, faces: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Radial function of a *convex* mesh, from its facet half-spaces.

    Uses the same min-over-half-spaces identity as (1) but with the mesh's own facet
    planes, so the target is the true body rather than its 1152-normal approximation.
    """
    from .shapes import face_normals_areas

    n, a = face_normals_areas(verts, faces)
    keep = a > 1e-14
    n = n[keep]
    d = (n * verts[faces[keep, 0]]).sum(1)
    d = np.maximum(d, 1e-9)              # origin must be strictly inside
    return radial_from_support(d, support_ray_matrix(n, rays))

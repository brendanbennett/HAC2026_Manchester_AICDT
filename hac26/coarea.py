"""Exact derivatives of the thresholded reductions, by the coarea formula.

The two reductions are discontinuous in the image:

    N(tau)   = sum_p 1[u_p > tau]
    I(tau_I) = sum_p u_p 1[u_p > tau_I]

but the quantities they approximate are not. Writing them as area integrals and perturbing
u by delta u, the coarea formula gives the derivative as an integral over the LEVEL SET:

    dN     = int_{u = tau} (delta u / |grad_x u|) dl
    dI     = int_{u > tau_I} delta u dA  +  tau_I int_{u = tau_I} (delta u / |grad_x u|) dl

So the exact derivative needs the contour at level tau, its arclength element, and the
image-space gradient magnitude on it -- nothing else. Marching squares supplies the first
two, and |grad_x u| comes from finite differences on the image grid, which is legitimate
precisely because the PSF has already made u smooth at the pixel scale.

WHY NOT JUST SOFTEN THE THRESHOLD, which is one line instead of this module:

  * Softening the SILHOUETTE costs A_sigma = A + pi m2 sigma^2 chi(S) + O(sigma^4). The
    first-order term vanishes by symmetry and the second-order term is a deformation
    invariant -- the Euler characteristic -- so it contributes no gradient at all. You pay
    a bias and get nothing.
  * Softening the VALUE threshold costs N_eps = N(tau) - (m2 eps^2 / 2) h'(tau), with h the
    frame's intensity histogram density. h'(tau) swings with phase, so that bias is a
    function of psi: it aliases directly into the signal being fitted.
  * Monte-Carlo pixel noise is the same relaxation applied without admitting it, since
    E[1[u + n > tau]] = Phi((u - tau)/varsigma).

Silhouette coverage is handled separately and upstream, by nvdiffrast's antialias(), which
supplies the analytic coverage derivative at geometric edges. No soft rasteriser is layered
on top of it.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["contour_weights", "threshold_count", "threshold_sum"]


def _grad_mag(u: np.ndarray) -> np.ndarray:
    """|grad u| by central differences on the image grid, in units of 1/pixel."""
    gy, gx = np.gradient(u)
    return np.sqrt(gx ** 2 + gy ** 2)


def contour_weights(u: np.ndarray, tau: float, eps: float = 1e-8):
    """Scatter weights for the level-set integral of `u` at level `tau`.

    Returns (rows, cols, w) such that, for any perturbation field d,

        int_{u = tau} (d / |grad u|) dl  ~=  sum_k w_k * d[rows_k, cols_k]

    Each marching-squares segment contributes its length divided by the local gradient
    magnitude, distributed bilinearly onto the four pixels around its midpoint.
    """
    from skimage.measure import find_contours

    g = _grad_mag(u)
    rows, cols, wts = [], [], []
    for c in find_contours(u, level=tau):
        if len(c) < 2:
            continue
        seg = c[1:] - c[:-1]
        length = np.sqrt((seg ** 2).sum(1))
        mid = 0.5 * (c[1:] + c[:-1])
        r0 = np.clip(np.floor(mid[:, 0]).astype(int), 0, u.shape[0] - 2)
        c0 = np.clip(np.floor(mid[:, 1]).astype(int), 0, u.shape[1] - 2)
        fr, fc = mid[:, 0] - r0, mid[:, 1] - c0
        gm = (g[r0, c0] * (1 - fr) * (1 - fc) + g[r0 + 1, c0] * fr * (1 - fc)
              + g[r0, c0 + 1] * (1 - fr) * fc + g[r0 + 1, c0 + 1] * fr * fc)
        w = length / np.maximum(gm, eps)
        for dr, dc, bw in ((0, 0, (1 - fr) * (1 - fc)), (1, 0, fr * (1 - fc)),
                           (0, 1, (1 - fr) * fc), (1, 1, fr * fc)):
            rows.append(r0 + dr); cols.append(c0 + dc); wts.append(w * bw)
    if not rows:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0)
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(wts))


class _ThresholdCount(torch.autograd.Function):
    """N(tau) = sum 1[u > tau], forward exact, backward by coarea."""

    @staticmethod
    def forward(ctx, u: torch.Tensor, tau: float):
        un = u.detach().cpu().numpy()
        r, c, w = contour_weights(un, float(tau))
        ctx.shape = u.shape
        ctx.save_for_backward(
            torch.as_tensor(r, dtype=torch.long, device=u.device),
            torch.as_tensor(c, dtype=torch.long, device=u.device),
            torch.as_tensor(w, dtype=u.dtype, device=u.device))
        return (u > tau).sum().to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        r, c, w = ctx.saved_tensors
        grad = torch.zeros(ctx.shape, dtype=g.dtype, device=g.device)
        if len(r):
            grad.index_put_((r, c), g * w, accumulate=True)
        return grad, None


class _ThresholdSum(torch.autograd.Function):
    """I(tau) = sum u 1[u > tau]; interior term plus tau times the boundary term."""

    @staticmethod
    def forward(ctx, u: torch.Tensor, tau: float):
        un = u.detach().cpu().numpy()
        r, c, w = contour_weights(un, float(tau))
        mask = (u > tau).to(u.dtype)
        ctx.tau = float(tau)
        ctx.save_for_backward(
            mask,
            torch.as_tensor(r, dtype=torch.long, device=u.device),
            torch.as_tensor(c, dtype=torch.long, device=u.device),
            torch.as_tensor(w, dtype=u.dtype, device=u.device))
        return (u * mask).sum()

    @staticmethod
    def backward(ctx, g):
        mask, r, c, w = ctx.saved_tensors
        grad = g * mask                               # interior: d/du of u where u > tau
        if len(r):
            grad = grad.clone()
            grad.index_put_((r, c), g * ctx.tau * w, accumulate=True)
        return grad, None


def threshold_count(u: torch.Tensor, tau: float) -> torch.Tensor:
    """Pixel count above tau, with the exact coarea derivative."""
    return _ThresholdCount.apply(u, tau)


def threshold_sum(u: torch.Tensor, tau: float) -> torch.Tensor:
    """Summed value above tau, with the exact coarea derivative."""
    return _ThresholdSum.apply(u, tau)

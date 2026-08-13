"""Exact derivatives of the thresholded reductions, by the coarea formula.

The two reductions are discontinuous in the image: a pixel enters the sum only once its value
crosses tau. The coarea formula turns the derivative of such a threshold into an integral
over the level set,

    d/dtheta INT_{u > tau} g = INT_{u = tau} (g / |grad u|) du/dtheta dl

so the gradient is carried by the contour at level tau, weighted by 1/|grad u|. The contour
is traced by marching squares and the weight evaluated per segment.

Softening the threshold instead would change the forward value as well as its derivative, and
the softening width is a free parameter that no measurement sets.
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

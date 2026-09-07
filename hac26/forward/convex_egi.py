"""Convex forward model: T = N o A, linear in the facet areas of the extended Gaussian image.

A is the photometric tensor on the normal grid,
    A[c, k, i] = s_c( <R3(psi_k) u_i, omega_c>, <R3(psi_k) u_i, OMEGA0> ),
where s_c is the kernel of curve c (see `kernel`): Lommel-Seeliger plus Lambert for an
intensity curve, projected area for a binary curve, both zero where the normal faces away
from the camera or the light. Per-curve constant factors cancel under normalization and are
omitted. N is the per-curve mean normalization N(y) = y / mean(y), implemented with a guard
N_eps(y) = y / max(mean(y), eps); it equals N whenever mean(y) >= eps.

Closed forms used by the LPD:
    DN(y)v      = v/mbar - y*mean(v)/mbar^2
    DN(y)^T w   = w/mbar - (<y,w>/(m*mbar^2)) * ones
    [d(N o A)(g)]^T = A^T o DN(Ag)^T
"""
from __future__ import annotations

import numpy as np

from hac26.geometry import OMEGA0, NormalGrid, body_frame_dirs, psi_grid


def kernel(mu: np.ndarray, mu0: np.ndarray, curve_type: str, c_lambert: float,
           ls_weight: float = 1.0, tau: float = 0.0) -> np.ndarray:
    """Per-normal weight of one curve type, from the cosines to the camera (mu) and to the
    light (mu0).

        intensity   ls_weight * mu*mu0/(mu+mu0) + c_lambert * mu*mu0   (Lommel-Seeliger + Lambert)
        binary      mu                                                (projected area)

    The binary curve counts lit pixels, so a facet contributes its projected area whatever
    its brightness. A normal contributes only where mu > 0, mu0 > 0 and mu*mu0 > tau; tau is
    a brightness threshold below which a facet is not counted at all.
    """
    rad = np.where((mu > 0.0) & (mu0 > 0.0), mu * mu0, 0.0)
    lit = rad > tau
    if curve_type == "intensity":
        den = np.where(lit, mu + mu0, 1.0)
        return np.where(lit, ls_weight * mu * mu0 / den + c_lambert * mu * mu0, 0.0)
    if curve_type == "binary":
        return np.where(lit, mu, 0.0)
    raise ValueError(curve_type)


def build_A(normals: np.ndarray, cameras: list, m: int, curve_types: list,
            c_lambert: float = 0.1, sigma: float = 1.0, delta: float = 1.0,
            psi0: float = 0.0, ls_weight: float = 1.0,
            tau_i: float = 0.0, tau_b: float = 0.0) -> np.ndarray:
    """Photometric tensor A of shape (n_curves, m, N) for the given normals.

    `cameras` and `curve_types` are parallel lists, one entry per output curve. stack_A()
    builds the full stack of every camera as intensity and then as binary.
    """
    psi = psi_grid(m, sigma=sigma, psi0=psi0)
    v0 = body_frame_dirs(OMEGA0, psi)                    # (m,3)
    mu0 = normals @ v0.T                                 # (N,m)
    rows = []
    for cam, ctype in zip(cameras, curve_types):
        v = body_frame_dirs(cam.omega(delta=delta), psi)  # (m,3)
        mu = normals @ v.T                                # (N,m)
        rows.append(kernel(mu, mu0, ctype, c_lambert, ls_weight=ls_weight,
                       tau=(tau_i if ctype == 'intensity' else tau_b)).T)  # (m,N)
    return np.stack(rows, axis=0)


def stack_A(grid: NormalGrid, cameras: list, m: int, c_lambert: float = 0.1,
            sigma: float = 1.0, delta: float = 1.0, psi0: float = 0.0) -> tuple:
    """Full operator for one model: every camera's intensity curve, then every camera's
    binary curve. Returns (A, curve_types) with A of shape (2 * len(cameras), m, N)."""
    cams2 = list(cameras) + list(cameras)
    types = ["intensity"] * len(cameras) + ["binary"] * len(cameras)
    A = build_A(grid.normals, cams2, m, types, c_lambert=c_lambert,
                sigma=sigma, delta=delta, psi0=psi0)
    return A, types


# ---------- numpy reference implementations -----------------------------------------
def normalize_np(y: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """N_eps(y): divide each curve (last axis) by max(mean, eps)."""
    mbar = np.maximum(y.mean(axis=-1, keepdims=True), eps)
    return y / mbar


def dn_np(y: np.ndarray, v: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Derivative of N_eps at y applied to v, per curve (last axis = frames)."""
    mbar = y.mean(axis=-1, keepdims=True)
    guard = mbar >= eps
    mb = np.maximum(mbar, eps)
    mv = v.mean(axis=-1, keepdims=True)
    return np.where(guard, v / mb - y * mv / mb**2, v / eps)


def dn_adjoint_np(y: np.ndarray, w: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Adjoint of dn_np: DN(y)^T w = w/mbar - (<y,w>/(m*mbar^2)) ones where mbar >= eps,
    and w/eps where the guard is active."""
    m = y.shape[-1]
    mbar = y.mean(axis=-1, keepdims=True)
    guard = mbar >= eps
    mb = np.maximum(mbar, eps)
    yw = (y * w).sum(axis=-1, keepdims=True)
    return np.where(guard, w / mb - yw / (m * mb**2) * np.ones_like(w), w / eps)


def forward_np(A: np.ndarray, g: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """T(g) = N_eps(A g); A (C,m,N), g (N,) -> (C,m)."""
    return normalize_np(np.einsum("cmn,n->cm", A, g), eps=eps)


def deriv_adjoint_np(A: np.ndarray, g: np.ndarray, h: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """[dT(g)]^T h = A^T DN(Ag)^T h; h (C,m) -> (N,)."""
    y = np.einsum("cmn,n->cm", A, g)
    return np.einsum("cmn,cm->n", A, dn_adjoint_np(y, h, eps=eps))


# ---------- torch operator ----------------------------------------------------------
try:
    import torch

    class ConvexPhotometricOperator(torch.nn.Module):
        """T = N_eps o A as a torch module, with the closed-form derivative adjoint.

        One buffer, A (C, m, N) float32. The curve mask is not stored here; the calls that
        need it take it as an argument, shape (B, C), with 0 marking a missing curve.
        """

        def __init__(self, A: np.ndarray, eps: float = 1e-3):
            super().__init__()
            self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
            self.eps = float(eps)

        @property
        def n_curves(self) -> int:
            return self.A.shape[0]

        @property
        def m(self) -> int:
            return self.A.shape[1]

        @property
        def n(self) -> int:
            return self.A.shape[2]

        def raw(self, g: "torch.Tensor") -> "torch.Tensor":
            """A g : (B,N) -> (B,C,m)."""
            return torch.einsum("cmn,bn->bcm", self.A, g)

        def normalize(self, y: "torch.Tensor") -> "torch.Tensor":
            """N_eps along the last axis."""
            mbar = y.mean(dim=-1, keepdim=True).clamp_min(self.eps)
            return y / mbar

        def forward(self, g: "torch.Tensor") -> "torch.Tensor":
            return self.normalize(self.raw(g))

        def adjoint_raw(self, r: "torch.Tensor") -> "torch.Tensor":
            """A^T r : (B,C,m) -> (B,N)."""
            return torch.einsum("cmn,bcm->bn", self.A, r)

        def dn_adjoint(self, y: "torch.Tensor", w: "torch.Tensor") -> "torch.Tensor":
            """DN(y)^T w along the last axis; same branches as dn_adjoint_np."""
            mfr = y.shape[-1]
            mbar = y.mean(dim=-1, keepdim=True)
            guard = mbar >= self.eps
            mb = mbar.clamp_min(self.eps)
            yw = (y * w).sum(dim=-1, keepdim=True)
            full = w / mb - yw / (mfr * mb**2)
            return torch.where(guard, full, w / self.eps)

        def deriv_adjoint(self, g: "torch.Tensor", h: "torch.Tensor",
                          mask: "torch.Tensor | None" = None) -> "torch.Tensor":
            """[dT(g)]^T h with optional per-curve mask (B,C): masked curves contribute 0."""
            y = self.raw(g)
            r = self.dn_adjoint(y, h)
            if mask is not None:
                r = r * mask[..., None]
            return self.adjoint_raw(r)

except ImportError:  # torch is optional for the numpy-side of the package
    pass

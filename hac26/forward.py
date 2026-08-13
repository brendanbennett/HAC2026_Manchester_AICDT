"""Exact convex forward model of HAC 2026: T = N o A (see docs/HAC2026_LPD_forward_model.md).

A is the photometric matrix on the EGI:
    A[(c,k), i] = s_c( <R3(psi_k) u_i, omega_c>, <R3(psi_k) u_i, OMEGA0> )
with kernels
    intensity: s(mu, mu0) = (mu*mu0/(mu+mu0) + c_L * mu*mu0) * 1{mu>0} * 1{mu0>0}
               (Lommel-Seeliger + Lambert; per-curve constants kappa_c f(alpha_c)
                cancel under normalization and are omitted)
    binary:    b(mu, mu0) = mu * 1{mu>0} * 1{mu0>0}
N is per-curve mean-normalization N(y) = y / mean(y) (challenge page), implemented with
an epsilon guard N_eps(y) = y / max(mean(y), eps); exact whenever mean(y) >= eps.

Closed forms used by the LPD (verified in tests/test_core.py):
    DN(y)v      = v/mbar - y*mbar(v)/mbar^2
    DN(y)^T w   = w/mbar - (<y,w>/(m*mbar^2)) * ones
    [d(N o A)(g)]^T = A^T o DN(Ag)^T
"""
from __future__ import annotations

import numpy as np

from .geometry import OMEGA0, NormalGrid, body_frame_dirs, psi_grid


def kernel(mu: np.ndarray, mu0: np.ndarray, curve_type: str, c_lambert: float,
           ls_weight: float = 1.0, tau: float = 0.0) -> np.ndarray:
    """Per-normal weight for one curve type.

    THE TWO CURVE TYPES ARE DIFFERENT FUNCTIONALS OF THE AREA MEASURE:

        intensity   mu+ mu0+ . 1[mu+ mu0+ > tau_I]      summed pixel VALUE
        binary      mu+      . 1[mu+ mu0+ > tau_B]      pixel COUNT, i.e. lit area

    The binary curve counts pixels, so each lit facet contributes its projected area mu+
    regardless of how bright it is -- the brightness only decides whether it is counted at
    all. The thresholds are the organisers' own: Otsu for binary, a fixed lower value for
    intensity, neither published. They are nuisance parameters, not zeros, and a facet
    whose radiance falls below one contributes nothing.

    tau = 0 recovers the previous behaviour exactly, so nothing silently changes.
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
    """Photometric tensor A, shape (n_curves, m, N).

    curve_types: list parallel to the output curve axis; the full 56-curve stack is
    [28 cameras x 'intensity'] + [28 cameras x 'binary'] via stack_A().
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
    """Full operator for one model: 28 intensity curves then 28 binary curves.

    Returns (A, curve_types) with A of shape (56, m, N)."""
    cams2 = list(cameras) + list(cameras)
    types = ["intensity"] * len(cameras) + ["binary"] * len(cameras)
    A = build_A(grid.normals, cams2, m, types, c_lambert=c_lambert,
                sigma=sigma, delta=delta, psi0=psi0)
    return A, types


# ---------- numpy reference implementations (used in tests) ------------------------
def normalize_np(y: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    mbar = np.maximum(y.mean(axis=-1, keepdims=True), eps)
    return y / mbar


def dn_np(y: np.ndarray, v: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Frechet derivative of N_eps at y applied to v (per curve, last axis = frames)."""
    mbar = y.mean(axis=-1, keepdims=True)
    guard = mbar >= eps
    mb = np.maximum(mbar, eps)
    mv = v.mean(axis=-1, keepdims=True)
    return np.where(guard, v / mb - y * mv / mb**2, v / eps)


def dn_adjoint_np(y: np.ndarray, w: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Adjoint of dn_np: DN(y)^T w = w/mbar - (<y,w>/(m*mbar^2)) ones, on the branch mbar>=eps."""
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
        """T = N_eps o A as a torch module; closed-form derivative adjoint.

        Buffers:
            A     (C, m, N) float32
            mask  broadcastable (C,) default ones -- 0 marks curves absent from a file.
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
            mbar = y.mean(dim=-1, keepdim=True).clamp_min(self.eps)
            return y / mbar

        def forward(self, g: "torch.Tensor") -> "torch.Tensor":
            return self.normalize(self.raw(g))

        def adjoint_raw(self, r: "torch.Tensor") -> "torch.Tensor":
            """A^T r : (B,C,m) -> (B,N)."""
            return torch.einsum("cmn,bcm->bn", self.A, r)

        def dn_adjoint(self, y: "torch.Tensor", w: "torch.Tensor") -> "torch.Tensor":
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

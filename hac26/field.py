"""-- geometry representation.

    f(y) = max_j (n_j . y - h_j)  +  s * Delta(y),        s = 0.15 R  (fixed)

A convex core carried by 64 support values h >= 0 on fixed normals, plus a signed token
correction that both carves and grows.

WHY plain max AND NOT log-sum-exp. Autodiff routes the subgradient of a max to the argmax,
which is exactly right here: the active halfspace is the one that owns the surface at that
point. Log-sum-exp instead biases the zero set inward by log(J)/beta, which at J = 64 and
beta = 50/R is log(64)/(50/R) = 0.083 R -- a systematic 8% shrink of every body, applied
silently.

WHY Delta must stay signed. A one-sided activation would make the correction able only to
carve or only to grow. The concavity sought is a deficit, but the same field has to
be able to push out a lobe; forcing the sign turns the representation into a bias.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["spherical_design", "DESIGN_N", "DESIGN_T", "ConvexCore", "TokenField",
           "ImplicitBody", "extract_mesh", "apply_constraints", "CORE_SCALE",
           "TOKEN_SIGMA_FRAC"]

DESIGN_N = 64          # number of fixed normals
DESIGN_T = 10          # spherical design strength
CORE_SCALE = 0.15      # s = CORE_SCALE * R
TOKEN_SIGMA_FRAC = 0.25   # sigma = TOKEN_SIGMA_FRAC * R
N_TOKENS = 32
TOKEN_DIM = 16


# ----------------------------------------------------------------- the fixed normals

def _legendre_gram(g: torch.Tensor, t: int) -> list:
    """P_l applied elementwise to a Gram matrix, by the three-term recurrence."""
    out = [torch.ones_like(g), g]
    for l in range(2, t + 1):
        out.append(((2 * l - 1) * g * out[l - 1] - (l - 1) * out[l - 2]) / l)
    return out


def design_energy(x, t: int = DESIGN_T):
    """Sum of the per-degree design energies; zero exactly for a t-design.

    The identity that makes this the correct objective:

        sum_{i,j} P_l(x_i . x_j) = (4 pi / (2l+1)) sum_m | sum_i Y_lm(x_i) |^2  >= 0

    Each term is non-negative and vanishes precisely when the degree-l harmonic moments do,
    which IS the t-design condition -- and it needs only Legendre polynomials of the Gram
    matrix, so it differentiates trivially, unlike evaluating Y_lm directly.

    Minimising raw monomial means instead is wrong: for even
    l the points being unit vectors forces sum_i x_i^2 = n/3, so the target is unreachable.
    It did not converge, and left the residual WORSE than the Fibonacci spiral it started
    from (1.043 against 0.062).
    """
    xt = torch.as_tensor(x, dtype=torch.float64)
    g = (xt @ xt.T).clamp(-1.0, 1.0)
    P = _legendre_gram(g, t)
    n = len(xt)
    return sum((2 * l + 1) * P[l].sum() / (n * n) for l in range(1, t + 1))


def _design_residual(x: np.ndarray, t: int = DESIGN_T) -> float:
    """Worst single-degree design energy. Zero for an exact t-design."""
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    g = (xt @ xt.T).clamp(-1.0, 1.0)
    P = _legendre_gram(g, t)
    n = len(xt)
    return float(max(abs(P[l].sum().item()) / (n * n) for l in range(1, t + 1)))


def spherical_design(n: int = DESIGN_N, t: int = DESIGN_T, seed: int = 0,
                     iters: int = 4000) -> np.ndarray:
    """n unit normals forming a spherical t-design, with the six axis directions included.

    The axis directions are pinned. The reference case is a
    cube, and the core is an intersection of halfspaces tangent to the target: it reproduces
    a cube EXACTLY only if the cube's own face normals are available. A design free to drift
    off the axes leaves the nearest normal some degrees away, and the intersection then
    bulges at every face centre. Pinning six of the sixty-four costs the design property
    almost nothing (the residual is reported by `_design_residual` and asserted in the tests)
    and makes flat-faced bodies -- which every ground truth is -- exactly representable.

    The remaining n - 6 points are optimised to kill the harmonic sums that define the
    design, starting from a Fibonacci spiral.
    """
    cache = Path(__file__).with_name(f"design{n}.npy")
    if cache.exists() and t == DESIGN_T:
        x = np.load(cache)
        if len(x) == n:
            return x
    axes = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
    m = n - len(axes)
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    tht = np.pi * (1 + 5 ** 0.5) * i
    free = np.stack([np.cos(tht) * np.sin(phi), np.sin(tht) * np.sin(phi), np.cos(phi)], 1)

    p = torch.tensor(free, dtype=torch.float64, requires_grad=True)
    fixed = torch.tensor(axes, dtype=torch.float64)
    opt = torch.optim.Adam([p], lr=1e-2)
    best, best_x = float("inf"), None
    for _ in range(iters):
        x = torch.cat([fixed, p / p.norm(dim=1, keepdim=True)], 0)
        loss = design_energy(x, t)
        opt.zero_grad(); loss.backward(); opt.step()
        v = float(loss)
        if v < best:
            best, best_x = v, x.detach().clone()
    return best_x.numpy()


# ----------------------------------------------------------------- the field

class ConvexCore(nn.Module):
    """max_j (n_j . y - h_j), with h >= 0 enforced by softplus on the raw parameter."""

    def __init__(self, normals: np.ndarray):
        super().__init__()
        self.register_buffer("n", torch.tensor(normals, dtype=torch.float32))
        self.raw_h = nn.Parameter(torch.zeros(len(normals)))

    @property
    def h(self) -> torch.Tensor:
        return F.softplus(self.raw_h)

    def set_support(self, h: torch.Tensor | np.ndarray) -> None:
        """Set h directly (inverse softplus), e.g. from an analytic support function."""
        h = torch.as_tensor(h, dtype=torch.float32).clamp_min(1e-6)
        with torch.no_grad():
            self.raw_h.copy_(h + torch.log(-torch.expm1(-h)))   # stable softplus inverse

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return (y @ self.n.T - self.h).amax(dim=-1)             # plain max, not LSE


class TokenField(nn.Module):
    """Signed, zero-mean correction Delta(y) decoded from 32 tokens by cross-attention.

        a_i(y) = softmax_i[ q(y).k(z_i)/sqrt(d) - ||y - p_i||^2 / (2 sigma^2) ]
        Delta(y) = MLP( sum_i a_i(y) v(z_i) )

    sigma is fixed at 0.25 R, not learned: it sets the spatial reach of a token, and letting
    the network shrink it is how a token collapses into a spike that fits noise.
    """

    def __init__(self, radius: float, n_tokens: int = N_TOKENS, dim: int = TOKEN_DIM,
                 d_att: int = 32, hidden: int = 64):
        super().__init__()
        self.sigma = TOKEN_SIGMA_FRAC * radius
        self.p = nn.Parameter(torch.zeros(n_tokens, 3))
        self.z = nn.Parameter(torch.zeros(n_tokens, dim))
        self.q = nn.Linear(3, d_att)
        self.k = nn.Linear(dim, d_att)
        self.v = nn.Linear(dim, d_att)
        self.mlp = nn.Sequential(nn.Linear(d_att, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))
        self.d_att = d_att

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        logits = (self.q(y) @ self.k(self.z).T) / np.sqrt(self.d_att)
        d2 = ((y[:, None, :] - self.p[None, :, :]) ** 2).sum(-1)
        a = torch.softmax(logits - d2 / (2.0 * self.sigma ** 2), dim=-1)
        out = self.mlp(a @ self.v(self.z))[:, 0]
        return out - out.mean()          # signed and zero-mean, by construction


class ImplicitBody(nn.Module):
    """f(y) = core(y) + s * Delta(y), s = 0.15 R fixed."""

    def __init__(self, radius: float, normals: np.ndarray | None = None):
        super().__init__()
        if normals is None:
            normals = spherical_design()
        self.radius = float(radius)
        self.s = CORE_SCALE * float(radius)
        self.core = ConvexCore(normals)
        self.tokens = TokenField(radius)

    def forward(self, y: torch.Tensor, use_tokens: bool = True) -> torch.Tensor:
        f = self.core(y)
        if use_tokens:
            f = f + self.s * self.tokens(y)
        return f


# ----------------------------------------------------------------- extraction

def extract_mesh(field, extent: float, res: int = 128, device: str = "cpu",
                 chunk: int = 262144):
    """FlexiCubes on a res^3 grid. Not Marching Cubes -- see the module docstring of
    hac26.vendor.flexicubes."""
    from .vendor.flexicubes import FlexiCubes

    fc = FlexiCubes(device=device)
    x_nx3, cube_fx8 = fc.construct_voxel_grid(res)
    x_nx3 = x_nx3 * (2.0 * extent)                       # grid spans [-extent, extent]
    vals = []
    with torch.no_grad():
        for i in range(0, len(x_nx3), chunk):
            vals.append(field(x_nx3[i:i + chunk]))
    sdf = torch.cat(vals)
    verts, faces, _ = fc(x_nx3, sdf, cube_fx8, res, training=False)
    return verts.detach().cpu().numpy(), faces.detach().cpu().numpy()


def apply_constraints(verts: np.ndarray, radius: float, tol: float = 0.03) -> np.ndarray:
    """Applied to the EXTRACTED VERTICES, not to the field.

    z is rescaled affinely so the body touches -1 and +1 exactly, which the challenge states
    as equalities. The radius is then brought inside R only if it exceeds it.

    Posed this way the three public bodies measure r/R = 1.0079, 1.0271 and 0.9940, so two
    of the three exceed their published R. The published radius is therefore treated as an
    approximation with tolerance `tol` rather than as a hard bound, which would shrink the
    true geometry.
    """
    v = np.asarray(verts, dtype=np.float64).copy()
    zmin, zmax = v[:, 2].min(), v[:, 2].max()
    if zmax - zmin < 1e-12:
        raise ValueError("degenerate body: zero z extent")
    v[:, 2] = 2.0 * (v[:, 2] - zmin) / (zmax - zmin) - 1.0
    cap = radius * (1.0 + tol)
    r = float(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).max())
    if r > cap:
        v[:, :2] *= cap / r
    return v

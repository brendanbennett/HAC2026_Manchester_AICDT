"""The LPD, as six Euler steps of a conditional flow.

PRIMAL: the 32 tokens, 32 x (p in R^3 + z in R^16) = 608 dimensions. h is not in the flow.
The convex core is what a cheap linear operator already recovers well; putting it in the
sampled variable would make the network re-derive something already solvable, and would let
sampling noise move the overall size of the body.

DUAL: the psi-Fourier coefficients of the mean-normalised curves, m = 1..40, whitened by
s_{c,m}.

The dual network is shared across m and never mixes m. Expanding a curve as
L_g(psi) = int k_g(R_psi^-1 n) dS(n) and using Y_lm(R_psi^-1 n) = e^{-i m psi} Y_lm(n),

    L-hat_g(m) = sum_l k^g_{lm} S_lm

so the operator is exactly block-diagonal in m. There is no cross-m coupling to learn, and
an architecture able to mix m would be free to model one -- fitting noise with it. So the
set transformer runs independently at each m with weights shared across m, and m enters
only through a Fourier embedding. Attention runs across GEOMETRIES at fixed m, which is
precisely the cross-geometry amplitude-ratio computation that recovers the m = 0 content
that mean normalisation appears to destroy. Permutation invariance handles missing
geometries for free.

PROFILING. The cheap convex operator A_conv is kept alongside. At each iteration the
residual is split by the projector Pi onto its range, and BOTH r and (I - Pi) r are fed to
the primal network. The second is the part of the data no convex body can explain, so the
convex block cannot absorb the concavity signal before the network sees it.

Flow matching rather than L2 or expected-DICE. Either of those computes E[x | g], and the
posterior here provably contains indistinguishable pairs: the spindle r(z) = R(1 - |z|/2)
and the hourglass r(z) = R(1/2 + |z|/2) have equal volume, equal silhouette area from every
equatorial direction, equal R and equal z-extent, and being axisymmetric they give constant
curves -- so all 28 mean-normalised curves are identically 1.0 for both, while they sit
0.263 apart in Dice. Their conditional mean is neither of them and is smoother than both.
A flow samples from the posterior instead of averaging over it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["N_TOKENS", "TOKEN_DIM", "CODE_DIM", "N_MODES", "N_STEPS",
           "DualSetTransformer", "PrimalNet", "LPDFlow", "flow_targets"]

N_TOKENS = 32
TOKEN_DIM = 16
CODE_DIM = N_TOKENS * (3 + TOKEN_DIM)      # 608
N_MODES = 40
N_STEPS = 6


def fourier_embed(m: torch.Tensor, dim: int = 16) -> torch.Tensor:
    """Embedding of the rotation order m. The only way m enters the dual network."""
    k = torch.arange(dim // 2, device=m.device, dtype=torch.float32)
    a = m[..., None].float() / (10.0 ** (2 * k / dim))
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class DualSetTransformer(nn.Module):
    """Attention across GEOMETRIES at fixed m, weights shared across m.

    Input  (B, C, M, F) with C geometries and M rotation orders, plus a geometry mask.
    Output (B, C, M, width). No operation mixes different m.
    """

    def __init__(self, in_feat: int = 6, width: int = 96, heads: int = 4, blocks: int = 3,
                 m_dim: int = 16):
        super().__init__()
        self.inp = nn.Linear(in_feat + m_dim + 4, width)
        self.att = nn.ModuleList([nn.MultiheadAttention(width, heads, batch_first=True)
                                  for _ in range(blocks)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                                nn.SiLU(), nn.Linear(width, width))
                                  for _ in range(blocks)])
        self.m_dim = m_dim

    def forward(self, feats, geom_tag, mask, modes):
        B, C, M, _ = feats.shape
        me = fourier_embed(modes, self.m_dim)[None, None].expand(B, C, M, self.m_dim)
        x = self.inp(torch.cat([feats, me, geom_tag[:, :, None, :].expand(B, C, M, 4)], -1))
        kpm = (mask < 0.5)                                   # True where a geometry is absent
        for att, mlp in zip(self.att, self.mlp):
            y = x.permute(0, 2, 1, 3).reshape(B * M, C, -1)  # attend over C, at fixed m
            km = kpm[:, None, :].expand(B, M, C).reshape(B * M, C)
            km = torch.where(km.all(-1, keepdim=True), torch.zeros_like(km), km)
            y, _ = att(y, y, y, key_padding_mask=km)
            x = x + y.reshape(B, M, C, -1).permute(0, 2, 1, 3)
            x = x + mlp(x)
        return x


class PrimalNet(nn.Module):
    """Maps the dual summary plus the current code to a flow velocity in code space."""

    def __init__(self, width: int = 96, hidden: int = 1024):
        # hidden must exceed CODE_DIM: the velocity is mostly x_t rescaled, so a layer
        # narrower than the code cannot pass it through even as an identity.
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CODE_DIM + 2 * width + 16, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, CODE_DIM))
        # Skip path with a learned, time-dependent gain. The velocity target is
        #
        #     u = x1 - x0 = (x1 - x_t) / (1 - t)
        #
        # identically, so the dominant term is the network's own input scaled by a function
        # of t alone. An MLP fed the concatenation of t and x_t must synthesise that
        # multiplicative interaction from additive layers across all CODE_DIM channels.
        # The gain is zero-initialised, so training starts from the plain MLP.
        self.gain = nn.Sequential(nn.Linear(16, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.gain[-1].weight)
        nn.init.zeros_(self.gain[-1].bias)

    def forward(self, code, dual_summary, perp_summary, t_embed):
        return (self.gain(t_embed) * code
                + self.net(torch.cat([code, dual_summary, perp_summary, t_embed], -1)))


class LPDFlow(nn.Module):
    """Six unrolled iterations, each one Euler step of a conditional flow."""

    def __init__(self, width: int = 96, n_modes: int = N_MODES, n_steps: int = N_STEPS):
        super().__init__()
        self.dual = DualSetTransformer(width=width)
        self.primal = PrimalNet(width=width)
        self.n_modes, self.n_steps = n_modes, n_steps
        self.register_buffer("modes", torch.arange(1, n_modes + 1))

    def _summaries(self, resid, perp, geom_tag, mask):
        """Dual network applied to the residual and, separately, to its convex complement."""
        d = self.dual(resid, geom_tag, mask, self.modes)
        p = self.dual(perp, geom_tag, mask, self.modes)
        w = mask[:, :, None, None]
        return ((d * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1e-6),
                (p * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1e-6))

    def velocity(self, code, resid, perp, geom_tag, mask, t):
        ds, ps = self._summaries(resid, perp, geom_tag, mask)
        te = fourier_embed(t, 16)
        return self.primal(code, ds, ps, te)

    @torch.no_grad()
    def sample(self, resid_fn, geom_tag, mask, batch: int = 1, device="cpu"):
        """x0 ~ N(0, I) then six Euler steps of size 1/6.

        `resid_fn(code)` returns (residual, convex-complement) for the current code, so the
        operator is re-applied at every step rather than linearised once.
        """
        x = torch.randn(batch, CODE_DIM, device=device)
        for k in range(self.n_steps):
            t = torch.full((batch,), k / self.n_steps, device=device)
            r, p = resid_fn(x)
            x = x + self.velocity(x, r, p, geom_tag, mask, t) / self.n_steps
        return x


def flow_targets(x1: torch.Tensor, generator=None):
    """One training draw: x0 ~ N(0, I), the six interpolation times, and the target velocity.

    The target is x1 - x0 at every t, which is what makes this conditional flow matching
    rather than a denoiser: the velocity field is constant along each straight path, so the
    network is never asked to predict a posterior mean.
    """
    x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
    ts = torch.arange(N_STEPS, device=x1.device, dtype=x1.dtype) / N_STEPS
    xt = (1 - ts[:, None, None]) * x0[None] + ts[:, None, None] * x1[None]
    return x0, ts, xt, (x1 - x0)

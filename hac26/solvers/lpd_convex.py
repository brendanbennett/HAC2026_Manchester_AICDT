"""Learned Primal-Dual network for HAC 2026 (Adler & Oektem, arXiv:1707.06474, Alg. 3).

Unrolled scheme (I iterations, per-iteration parameters):
    h_i = h_{i-1} + Gamma_i( h_{i-1}, T(softplus(f_{i-1}^{(2)})), d, tags, mask )
    f_i = f_{i-1} + Lambda_i( f_{i-1}, [dT(softplus(f_{i-1}^{(1)}))]^T h_i^{(1)}, coords )
    return p = softplus(f_I^{(1)}) / sum(...)
where T = N_eps o A is the exact convex photometric operator (forward.py) and the
derivative adjoint is the closed form A^T o DN^T, chained with softplus' = sigmoid.

Design choices tied to exact structure of the problem:
- Dual nets use 1D convolutions along the frame axis with *circular* padding
  (curves are exactly one revolution; azimuthal equivariance lemma).
- Primal nets are 2D CNNs on the (theta, phi) EGI grid, circular in phi.
- Per-curve conditioning channels ("tags"): azimuth/360, elevation/90, phase/180,
  is_binary; plus the availability mask (missing files at higher difficulty levels).
- The network predicts the *scale-free* EGI direction p = g/sum(g): absolute scale is
  provably unidentifiable from mean-normalized curves; final size comes from the
  z in [-1,1] prior downstream.
Deviation from the paper: the last conv of each block is zero-initialized (stabilizes
deep unrolls; the paper used Xavier everywhere).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hac26.forward.convex_egi import ConvexPhotometricOperator


def make_tags(cameras: list, curve_types: list) -> torch.Tensor:
    rows = []
    for cam, ctype in zip(cameras, curve_types):
        rows.append([cam.azimuth_deg / 360.0,
                     cam.elevation_deg / 90.0,
                     cam.phase_angle_deg / 180.0,
                     1.0 if ctype == "binary" else 0.0])
    return torch.tensor(rows, dtype=torch.float32)  # (C, 4)


class DualBlock(nn.Module):
    """Residual CNN on (B, ch, C, m); convolutions along frames only, circular."""

    def __init__(self, n_dual: int, n_in: int, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(n_in, ch, (1, 3))
        self.c2 = nn.Conv2d(ch, ch, (1, 3))
        self.c3 = nn.Conv2d(ch, n_dual, (1, 3))
        self.a1, self.a2 = nn.PReLU(ch), nn.PReLU(ch)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (1, 1, 0, 0), mode="circular")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.c1(self._pad(x)))
        x = self.a2(self.c2(self._pad(x)))
        return self.c3(self._pad(x))


class PrimalBlock(nn.Module):
    """Residual CNN on (B, ch, n_theta, n_phi); circular in phi, replicate in theta."""

    def __init__(self, n_primal: int, n_in: int, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(n_in, ch, 3)
        self.c2 = nn.Conv2d(ch, ch, 3)
        self.c3 = nn.Conv2d(ch, n_primal, 3)
        self.a1, self.a2 = nn.PReLU(ch), nn.PReLU(ch)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 0, 0), mode="circular")      # phi
        return F.pad(x, (0, 0, 1, 1), mode="replicate")  # theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.c1(self._pad(x)))
        x = self.a2(self.c2(self._pad(x)))
        return self.c3(self._pad(x))


class LPDNet(nn.Module):
    def __init__(self, op: ConvexPhotometricOperator, n_theta: int, n_phi: int,
                 cameras: list, curve_types: list, n_iter: int = 15,
                 n_primal: int = 7, n_dual: int = 7, ch: int = 48,
                 support_head: bool = False, r_cond: bool = False,
                 gate_rank: int = 0, gate_bias: float = 3.0):
        super().__init__()
        assert op.n == n_theta * n_phi
        self.op = op
        self.n_theta, self.n_phi = n_theta, n_phi
        self.n_iter, self.n_primal, self.n_dual = n_iter, n_primal, n_dual
        self.support_head = support_head
        # Conditioning on the a-priori bounding radius R. Per-curve mean normalization
        # destroys the cross-camera amplitudes that encode the body's latitude profile,
        # so the aspect ratio is close to unidentifiable from the curves alone. R
        # supplies exactly that missing number, and supplying it as an INPUT lets the
        # network use it while reading the curves -- not merely when writing the answer.
        self.r_cond = r_cond
        self.register_buffer("tags", make_tags(cameras, curve_types))  # (C,4)
        th = (np.arange(n_theta) + 0.5) * np.pi / n_theta
        coords = np.stack([np.repeat(np.cos(th)[:, None], n_phi, 1),
                           np.repeat(np.sin(th)[:, None], n_phi, 1)])
        self.register_buffer("coords", torch.as_tensor(coords, dtype=torch.float32))
        n_tag = 4
        self.duals = nn.ModuleList(
            [DualBlock(n_dual, n_dual + 2 + n_tag + 1, ch) for _ in range(n_iter)])
        n_r = 1 if r_cond else 0
        n_back = max(1, gate_rank)
        self.primals = nn.ModuleList(
            [PrimalBlock(n_primal, n_primal + n_back + 2 + n_r, ch) for _ in range(n_iter)])
        # Support-function head. The unroll still reasons in EGI space (that is where
        # the photometric operator lives), but the *prediction* is h(u), which is the
        # better-conditioned description of a convex body: any h yields a valid body
        # by half-space intersection, and support functions form a convex cone so an
        # L2-optimal (posterior-mean) h is itself a support function. The EGI of a
        # polytope is a sum of deltas, whose posterior mean is a smeared non-shape.
        self.head_h = (PrimalBlock(1, n_primal + 2 + n_r, ch) if support_head else None)

        # --- occlusion gate ---------------------------------------------------------
        # Self-occlusion and cast shadow act as an entrywise gate on the operator,
        # L = sum_k A g_k v_k. Writing v as a rank-R sum v = sum_r d1^(r) (x) d2^(r) puts the
        # shape-space factor into R primal channels (w_r = d2^(r) * g) and the data-space
        # factor into the dual, so K = A stays fixed with its exact adjoint and every
        # iteration is still one Chambolle-Pock step:
        #
        #     y = N( sum_r d1^(r) * (A w_r) )
        #
        # R is bounded by identifiability rather than compute: R*N unknowns against C*m
        # measurements.
        #
        # d1 is a visibility factor and must stay in [0, 1], so it is a sigmoid with no
        # clamp and no branch. Unconstrained, y can go negative, normalize() hits its
        # clamp_min(eps) floor and the adjoint's fallback branch multiplies the backward
        # pass by 1/eps. The gate CNN's output bias starts at +gate_bias at rank 0 and small
        # and positive above it, so every rank starts the same distance from open. A
        # multiplicative scalar in front of the sigmoid is not used: its gradient carries
        # its own value, so one initialised near zero cannot open.
        self.gate_rank = gate_rank
        self.gate_bias = gate_bias
        if gate_rank:
            assert n_primal >= 1 + gate_rank, "need a primal channel per gate rank"
            self.gate_d1 = nn.ModuleList(
                [DualBlock(gate_rank, n_dual + 2 + n_tag + 1, ch) for _ in range(n_iter)])
            budget = 1.0 / (1.0 + math.exp(gate_bias))        # sigmoid(-gate_bias)
            t = budget / max(1, gate_rank - 1)                # per-rank share
            corr = math.log(t / (1.0 - t))
            for blk in self.gate_d1:
                with torch.no_grad():
                    blk.c3.bias.fill_(corr)
                    blk.c3.bias[0] = gate_bias

    def _gated_raw(self, w, d1):
        """sum_r d1^(r) * (A w_r):  w (B,R,N), d1 (B,R,C,m) -> (B,C,m)."""
        y = torch.einsum("cmn,brn->brcm", self.op.A, w)
        return (d1 * y).sum(1)

    def _gated_back(self, g1, h0, mask, d1):
        """[d(N o A~)(g1)]^T h0 per rank -> (B,R,N). Uses the fixed A both ways, so the
        adjoint identity holds exactly; only the gate weights differ per rank."""
        R = d1.shape[1]
        raw = self._gated_raw(g1[:, None].expand(-1, R, -1), d1)
        r = self.op.dn_adjoint(raw, h0)
        if mask is not None:
            r = r * mask[..., None]
        return torch.einsum("cmn,brcm->brn", self.op.A, d1 * r[:, None])

    def forward(self, d: torch.Tensor, mask: torch.Tensor,
                log_r: torch.Tensor | None = None) -> tuple:
        """d: (B, C, m) normalized curves (zeros where missing); mask: (B, C) in {0,1}.
        log_r: (B,) log of the a-priori bounding radius, required when r_cond=True."""
        B, C, m = d.shape
        nt, nph = self.n_theta, self.n_phi
        f = d.new_zeros(B, self.n_primal, nt, nph)
        h = d.new_zeros(B, self.n_dual, C, m)
        tags = self.tags.T[None, :, :, None].expand(B, 4, C, m)
        mch = mask[:, None, :, None].expand(B, 1, C, m)
        coords = self.coords[None].expand(B, 2, nt, nph)
        if self.r_cond:
            if log_r is None:
                raise ValueError("r_cond=True requires log_r")
            rch = log_r.reshape(B, 1, 1, 1).expand(B, 1, nt, nph).to(d.dtype)
            coords = torch.cat([coords, rch], dim=1)
        R = self.gate_rank
        for i in range(self.n_iter):
            if R:
                # d1 in (0,1) identically -- it is a sigmoid, nothing else. That is
                # what makes the forward, and its adjoint, defined everywhere.
                d1 = torch.sigmoid(
                    self.gate_d1[i](torch.cat([h, d[:, None], mch, tags, mch], dim=1)))
                w = F.softplus(f[:, 1:1 + R].reshape(B, R, -1))
                y2 = self.op.normalize(self._gated_raw(w, d1)) * mask[:, :, None]
            else:
                g2 = F.softplus(f[:, 1].reshape(B, -1))
                y2 = self.op(g2) * mask[:, :, None]
            dual_in = torch.cat([h, y2[:, None], d[:, None], tags, mch], dim=1)
            h = h + self.duals[i](dual_in)
            x1 = f[:, 0].reshape(B, -1)
            g1 = F.softplus(x1)
            if R:
                back = self._gated_back(g1, h[:, 0], mask, d1) * torch.sigmoid(x1)[:, None]
                back = back.reshape(B, R, nt, nph)
            else:
                back = self.op.deriv_adjoint(g1, h[:, 0], mask).reshape(B, 1, nt, nph) \
                    * torch.sigmoid(x1).reshape(B, 1, nt, nph)
            primal_in = torch.cat([f, back, coords], dim=1)
            f = f + self.primals[i](primal_in)
        g_out = F.softplus(f[:, 0].reshape(B, -1))
        p = g_out / g_out.sum(dim=1, keepdim=True).clamp_min(1e-12)
        if self.head_h is None:
            return p, g_out
        # softplus keeps h > 0 so the origin stays interior to every half-space set;
        # +1 centres the initial prediction near a unit sphere (zero-init last conv).
        h = F.softplus(self.head_h(torch.cat([f, coords], dim=1))[:, 0] + 1.0)
        return p, g_out, h.reshape(B, -1)

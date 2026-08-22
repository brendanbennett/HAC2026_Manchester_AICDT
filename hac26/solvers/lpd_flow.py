"""The LPD, as six Euler steps of a conditional flow.

PRIMAL: two blocks, CODE_DIM = N_DIR + N_SITES = 128 + 1728 = 1856.

  g  -- signed amplitudes on a fixed 12^3 lattice, which is what carries non-convexity.
  dh -- a band-limited correction to the support function, on N_DIR design directions.

dh is in the flow and h is not. The convex core is what a cheap linear operator already
recovers well, so sampling the whole of h would make the network re-derive something already
solvable and would let noise move the overall size of the body. But that operator ASSUMES
convexity, and a non-convex body is darker than its own hull from self-shadowing -- so it
explains darkness with shape and its h is wrong in a direction that always favours a convex
answer. dh corrects that, band-limited to degree <= 5 because the curves constrain only about
35-40 support directions and an out-of-band dh kills facets outright.

Each block gets its own network, because they are different objects on different domains: a
3-D CNN over the lattice for g, a sphere-convolution over the design directions for dh. The
predecessor flattened the whole code into an MLP, which is why it needed `hidden > CODE_DIM`
("the velocity is mostly x_t rescaled"); at 1856 that rule would have cost 13.1 M parameters
for 600 bodies. The structured branches cost about a tenth of that and the rule disappears
with the flattening, because both branches carry the identity path for free -- from S_0 = I in
the sphere bank and from the centre tap of the 3x3x3 kernel.

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

PROFILING, REMOVED. An earlier version fed the primal a second dual pass over `r_perp`,
described here as "the projector Pi onto the operator's range". No projector was ever
computed: it was `r_perp[..., :4] = 0`, a hard cut of rotation orders 1-4, and if --phases was
9 or fewer it silently became identically zero. It cost a full extra dual forward pass, two of
its six input channels were never assigned, and the honest argument against keeping it is that
A_conv is provably blind to concavity -- a notched cube's unshadowed curves are reproduced
exactly, as a fatter box -- so a projector onto its range could not isolate concavity either.
What A_conv is good for is the aligned adjoint channel J^T A^T DN^T r, which conditions the dh
branch and is never added to the velocity.

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

from hac26.field import CODE_DIM, LATTICE_SHAPE, N_DIR, N_SITES, dir_design   # noqa: E402

__all__ = ["CODE_DIM", "N_DIR", "N_SITES", "N_MODES", "N_STEPS", "T_DIM",
           "DualSetTransformer", "SphereBranch", "VolBranch", "PrimalNet", "CodeCodec",
           "LPDFlow", "fourier_embed", "time_embed"]

N_MODES = 40
N_STEPS = 6
T_DIM = 32          # width of the TIME embedding, which is not the mode embedding
G_LIMIT = 4.0       # the decode saturates at this multiple of the corpus's largest fitted
                    # amplitude. Measured: a body at 2.7x the corpus maximum still extracts
                    # and decimates normally, so 4x is comfortably outside anything the flow
                    # should ever emit and far inside where the arithmetic breaks.


def fourier_embed(m: torch.Tensor, dim: int = 16) -> torch.Tensor:
    """Embedding of the rotation order m. The only way m enters the dual network."""
    k = torch.arange(dim // 2, device=m.device, dtype=torch.float32)
    a = m[..., None].float() / (10.0 ** (2 * k / dim))
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


def time_embed(t: torch.Tensor, dim: int = T_DIM) -> torch.Tensor:
    """Embedding of t in [0, 1]. Separate from fourier_embed, which is scaled for m = 1..40.

    Reusing the mode embedding for time was a real defect: its slowest band has period about
    47 in its argument, so over t in [0,1] every one of its bands sits in the small-angle
    regime. Measured on the six step times it produced singular values
    [6.85, 1.03, 7.7e-2, 3.2e-3, 7.2e-5, 1.0e-6] -- condition number 6.7e6, effectively rank
    four. The gain path, whose entire job is to learn a function of t alone, was reading that.
    Here the frequencies span 1 to 2^(dim/2 - 1) cycles across the unit interval instead.
    """
    k = torch.arange(dim // 2, device=t.device, dtype=torch.float32)
    a = t[..., None].float() * (2.0 * np.pi) * (2.0 ** k)
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


def _sphere_operators(dirs: np.ndarray, k: int = 8) -> np.ndarray:
    """Five fixed (N, N) operators for a convolution on the sphere: I, a k-NN mean, two
    tangential directional means, and a k-NN second moment.

    A sphere has no global grid, so a convolution has to be built from operators that are
    equivariant under the sampling rather than from a fixed stencil. These five span the same
    information a 3x3 stencil gives on a plane: the value, the local mean, the two tangential
    first moments and one second moment. S_0 = I is what gives the branch its identity path
    for free, which is why the `hidden > CODE_DIM` rule the flat MLP needed does not apply.
    """
    n = len(dirs)
    g = dirs @ dirs.T
    np.fill_diagonal(g, -2.0)
    nb = np.argsort(-g, axis=1)[:, :k]                      # k nearest by cosine
    ops = np.zeros((5, n, n), dtype=np.float32)
    ops[0] = np.eye(n, dtype=np.float32)
    rows = np.repeat(np.arange(n), k)
    cols = nb.reshape(-1)
    ops[1][rows, cols] = 1.0 / k
    # a right-handed tangent frame at each direction, seeded from the least-aligned axis
    seed = np.zeros_like(dirs)
    seed[np.arange(n), np.argmin(np.abs(dirs), axis=1)] = 1.0
    e1 = seed - (seed * dirs).sum(1, keepdims=True) * dirs
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(dirs, e1)
    off = dirs[nb] - dirs[:, None, :]                       # (n, k, 3)
    d = np.linalg.norm(off, axis=2) + 1e-12
    ops[2][rows, cols] = ((off * e1[:, None, :]).sum(2) / d).reshape(-1) / k
    ops[3][rows, cols] = ((off * e2[:, None, :]).sum(2) / d).reshape(-1) / k
    ops[4][rows, cols] = (d ** 2 / (d ** 2).mean()).reshape(-1) / k
    return ops


class SphereConv(nn.Module):
    """y_c' = sum_k W_k[c, c'] (S_k x)_c + b. Weights shared across directions."""

    def __init__(self, ops: torch.Tensor, c_in: int, c_out: int):
        super().__init__()
        self.register_buffer("S", ops, persistent=False)
        self.w = nn.Parameter(torch.randn(len(ops), c_in, c_out) / np.sqrt(len(ops) * c_in))
        self.b = nn.Parameter(torch.zeros(c_out))

    def forward(self, x):                                   # x: (B, N, C_in)
        y = torch.einsum("kij,bjc->bkic", self.S, x)        # (B, K, N, C_in)
        return torch.einsum("bkic,kco->bio", y, self.w) + self.b


class _FiLM(nn.Module):
    """Per-channel scale and shift from the conditioning vector. Zero-init, so a fresh block
    starts as the identity and the branch starts as its own skip path."""

    def __init__(self, cond_dim: int, width: int):
        super().__init__()
        self.f = nn.Linear(cond_dim, 2 * width)
        nn.init.zeros_(self.f.weight); nn.init.zeros_(self.f.bias)
        self.width = width

    def forward(self, h, cond, spatial_dims: int):
        a, b = self.f(cond).chunk(2, -1)
        shape = (h.shape[0], self.width) + (1,) * spatial_dims
        if spatial_dims == 0:                               # (B, N, C) layout
            return h * (1 + a[:, None, :]) + b[:, None, :]
        return h * (1 + a.reshape(shape)) + b.reshape(shape)


class SphereBranch(nn.Module):
    """Velocity for the dh block: a sphere convolution over the N_DIR design directions.

    Input channels: dh_t, the aligned adjoint channel J^T A^T DN^T r (whitened per sample --
    its raw magnitude is order 1e2), h_base, and the three components of the direction itself.
    The adjoint channel is CONDITIONING and is never added to the velocity: A_conv is blind to
    concavity and under-signals a deep pit by 3x, so it can say which face is wrong but never
    what shape the dent is.
    """

    def __init__(self, cond_dim: int, width: int = 128, blocks: int = 4, in_ch: int = 6):
        super().__init__()
        ops = torch.from_numpy(_sphere_operators(dir_design(N_DIR)))
        self.inp = SphereConv(ops, in_ch, width)
        self.conv = nn.ModuleList([nn.ModuleList([SphereConv(ops, width, width),
                                                  SphereConv(ops, width, width)])
                                   for _ in range(blocks)])
        self.film = nn.ModuleList([_FiLM(cond_dim, width) for _ in range(blocks)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(blocks)])
        self.head = SphereConv(ops, width, 1)
        nn.init.zeros_(self.head.w); nn.init.zeros_(self.head.b)
        self.act = nn.SiLU()

    def forward(self, x, cond):                             # x: (B, N_DIR, in_ch)
        h = self.inp(x)
        for (c1, c2), film, norm in zip(self.conv, self.film, self.norm):
            y = c2(self.act(c1(norm(h))))
            h = h + film(y, cond, spatial_dims=0)
        return self.head(h)[..., 0]                         # (B, N_DIR)


class VolBranch(nn.Module):
    """Velocity for the g block: a 3-D CNN over the fixed lattice.

    Zero padding, not circular: the box is not periodic and a wrapped kernel would couple the
    two sides of the body. No culling either -- it would break the fixed index set that makes
    the code portable. The `core_sdf` channel is what replaces culling: it tells the network
    where the convex core's surface is, so a site deep inside or far outside is identifiable
    without removing it from the code.

    Input channels: g_t, core_sdf at the sites, the inside indicator, and normalised x, y, z.
    """

    def __init__(self, cond_dim: int, width: int = 64, blocks: int = 4, in_ch: int = 6):
        super().__init__()
        self.shape = LATTICE_SHAPE
        self.inp = nn.Conv3d(in_ch, width, 3, padding=1)
        self.conv = nn.ModuleList([nn.ModuleList([nn.Conv3d(width, width, 3, padding=1),
                                                  nn.Conv3d(width, width, 3, padding=1)])
                                   for _ in range(blocks)])
        self.film = nn.ModuleList([_FiLM(cond_dim, width) for _ in range(blocks)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(blocks)])
        self.head = nn.Conv3d(width, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.act = nn.SiLU()

    def forward(self, x, cond):                             # x: (B, in_ch, nx, ny, nz)
        h = self.inp(x)
        for (c1, c2), film, norm in zip(self.conv, self.film, self.norm):
            y = c2(self.act(c1(norm(h))))
            h = h + film(y, cond, spatial_dims=3)
        return self.head(h).reshape(x.shape[0], -1)         # (B, N_SITES)


class CodeCodec(nn.Module):
    """Between raw code units and the space the flow actually works in.

    Two jobs, both required and neither previously present anywhere in the repo.

    WHITENING. x0 is drawn from N(0, I) and the target is x1 - x0, so the two endpoints have
    to live in the same space. Measured on fitted bodies, dh has std 0.0199 and g has std
    0.0483 in units of R -- about fifty times narrower than N(0, I). Untransformed, the flow
    spends its whole trajectory travelling from a unit Gaussian to a spike.

    ASINH on g. Measured kurtosis 14.5 and max|z| 12.98; after asinh, 3.17 and 4.53. The heavy
    tail IS the deep carves, so the alternative -- clipping, or letting a Gaussian flow model
    it badly -- is exactly the alternative that gives back convex answers.

    Per-BLOCK SCALARS, not per-coordinate. A per-site mean and variance is body-dependent and
    would break the weight sharing that lets a few hundred bodies teach 1728 amplitudes -- the
    volume branch is a convolution precisely so that every site is read by the same kernel, and
    giving each site its own affine pre-transform undoes that. It is also unestimable: a
    600-body corpus gives 599 degrees of freedom per coordinate, and at 60 bodies the
    per-coordinate std is mostly noise, which then multiplies a 4-sigma tail draw into a body
    the corpus never contained.

    The scale is a MEDIAN ABSOLUTE DEVIATION, not a standard deviation, for the same reason:
    it is the amplitude distribution's own heavy tail that would otherwise set the scale.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("g_s", torch.ones(1))
        self.register_buffer("mu", torch.zeros(2))     # [dh, g], one scalar each
        self.register_buffer("sd", torch.ones(2))
        self.register_buffer("u_lim", torch.full((1,), 80.0))

    @torch.no_grad()
    def fit(self, codes: torch.Tensor, eps_std: float) -> None:
        """`codes` are raw fitted codes; `eps_std` is the scale of the dh perturbation the
        flow is trained against, because the corpus dh block is identically zero by
        construction (the corpus h is exact, so there is nothing to correct)."""
        g = codes[:, N_DIR:]
        self.g_s.fill_(float(g.abs().median().clamp_min(1e-12)))
        u = torch.asinh(g / self.g_s)
        med = u.median()
        mad = (u - med).abs().median().clamp_min(1e-8) * 1.4826    # -> sigma for a Gaussian
        self.mu[0] = 0.0                      # dh is centred by construction
        self.sd[0] = max(float(eps_std), 1e-12)
        self.mu[1] = float(med)
        self.sd[1] = float(mad)
        # The saturation point of the decode, expressed as a bound on the AMPLITUDE and
        # converted back through asinh. See decode() for why this exists.
        g_lim = float(g.abs().max()) * G_LIMIT
        self.u_lim.fill_(float(np.arcsinh(g_lim / float(self.g_s))))

    def _split(self, x):
        return x[..., :N_DIR], x[..., N_DIR:]

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        dh, g = self._split(raw)
        u = torch.asinh(g / self.g_s)
        return torch.cat([(dh - self.mu[0]) / self.sd[0],
                          (u - self.mu[1]) / self.sd[1]], -1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z -> raw code. SATURATING, and that is not optional.

        sinh is unbounded and its relative error grows with its argument, so an ordinary
        regression error in z becomes an astronomical error in the field: measured, doubling
        z takes |g| from 2.7x the corpus maximum to 545x, and tripling it to 1e5x. That alone
        is survivable -- the level set is nonsense but a mesh still comes out. What is not
        survivable is that float32 sinh OVERFLOWS at |u| ~ 89, i.e. |z| ~ 69: g becomes
        +-inf, the field is non-finite, and the extracted mesh is garbage that fails inside
        facet indexing several call frames later, long after the cause.

        A single gradient spike is enough to get there, and because the training checkpoint
        restores weights, optimiser moments and RNG together, a resume replays it exactly.

        Clamping `u` fixes the whole chain at its one source: no non-finite amplitude can be
        produced by any caller -- training probe, sampler or ablation. The limit is set from
        the corpus's own maximum amplitude times G_LIMIT, so it cannot bind on anything the
        flow has been taught to produce; it only saturates excursions that were already
        meaningless. Per coordinate rather than a vector rescale, so an in-range site is
        never touched by an out-of-range neighbour.
        """
        zd, zg = self._split(z)
        u = (zg * self.sd[1] + self.mu[1]).clamp(-self.u_lim, self.u_lim)
        return torch.cat([zd * self.sd[0] + self.mu[0], self.g_s * torch.sinh(u)], -1)


class PrimalNet(nn.Module):
    """The two branches plus the shared conditioning, and the time-dependent skip gain.

    The skip path survives the rewrite because the velocity target is
    u = x1 - x0 = (x1 - x_t)/(1 - t) identically, so the dominant term is the input scaled by
    a function of t alone. It is now per-block: dh and g have different scales and a single
    scalar gain would have to compromise between them. Zero-initialised, so training starts
    from the branches alone.
    """

    def __init__(self, summary_dim: int, cond_width: int = 256):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(summary_dim + T_DIM, cond_width), nn.SiLU(),
                                  nn.Linear(cond_width, cond_width))
        self.sphere = SphereBranch(cond_width)
        self.vol = VolBranch(cond_width)
        self.gain = nn.Sequential(nn.Linear(T_DIM, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.gain[-1].weight); nn.init.zeros_(self.gain[-1].bias)

    def forward(self, code, summary, t_embed, sphere_ch, vol_ch):
        c = self.cond(torch.cat([summary, t_embed], -1))
        v_dh = self.sphere(torch.cat([code[:, None, :N_DIR].transpose(1, 2), sphere_ch], -1), c)
        g = code[:, N_DIR:].reshape(-1, 1, *LATTICE_SHAPE)
        v_g = self.vol(torch.cat([g, vol_ch], 1), c)
        gain = self.gain(t_embed)
        skip = torch.cat([gain[:, :1] * code[:, :N_DIR], gain[:, 1:] * code[:, N_DIR:]], -1)
        return skip + torch.cat([v_dh, v_g], -1)


class LPDFlow(nn.Module):
    """Six unrolled iterations, each one Euler step of a conditional flow."""

    def __init__(self, width: int = 96, n_modes: int = N_MODES, n_steps: int = N_STEPS,
                 mode_feat: int = 16):
        super().__init__()
        self.dual = DualSetTransformer(width=width)
        # Pool over GEOMETRIES only, then project each mode to `mode_feat` with ONE shared
        # Linear. The predecessor summed over geometries AND modes while dividing by the
        # geometry count alone, so the summary was exactly N_MODES = 40 times a true mean --
        # measured 40.000004, independent of the mask -- and which mode carried the signal was
        # discarded. A dense Linear(n_modes*width, ...) would instead be precisely the cross-m
        # mixing this module's docstring spends fifteen lines arguing against.
        self.mode_proj = nn.Linear(width, mode_feat)
        self.summary_dim = n_modes * mode_feat
        self.primal = PrimalNet(self.summary_dim)
        self.codec = CodeCodec()
        self.n_modes, self.n_steps = n_modes, n_steps
        self.register_buffer("modes", torch.arange(1, n_modes + 1))

    def _summary(self, resid, geom_tag, mask):
        d = self.dual(resid, geom_tag, mask, self.modes)         # (B, C, M, width)
        w = mask[:, :, None, None]
        pooled = (d * w).sum(1) / w.sum(1).clamp_min(1e-6)       # (B, M, width) -- true mean
        # Slots beyond M = min(N_MODES, phases//2) are zero-filled by the caller and carry no
        # data, but the dual still emits bias + mode-embedding for them. Measured at phases=16
        # they contributed 81.7% MORE norm than the real modes. Detect them from the input
        # rather than from a constructor argument, so a checkpoint cannot disagree with a run.
        live = (resid.abs().sum((1, 3)) > 0).float()[..., None]  # (B, M, 1)
        return (self.mode_proj(pooled) * live).reshape(resid.shape[0], -1)

    def velocity(self, code, resid, geom_tag, mask, t, sphere_ch, vol_ch):
        te = time_embed(t)
        return self.primal(code, self._summary(resid, geom_tag, mask), te, sphere_ch, vol_ch)

    @torch.no_grad()
    def sample(self, resid_fn, geom_tag, mask, batch: int = 1, device="cpu"):
        """x0 ~ N(0, I) then n_steps Euler steps, in the codec's whitened space throughout.

        `resid_fn(code)` returns (residual, sphere_channels, vol_channels) for the current
        code and re-applies the operator at every step rather than linearising once. It is
        given the WHITENED code and decodes internally, so nothing outside this loop has to
        know which space it is holding.
        """
        x = torch.randn(batch, CODE_DIM, device=device)
        for k in range(self.n_steps):
            t = torch.full((batch,), k / self.n_steps, device=device)
            r, sph, vol = resid_fn(x, t)
            x = x + self.velocity(x, r, geom_tag, mask, t, sph, vol) / self.n_steps
        return x

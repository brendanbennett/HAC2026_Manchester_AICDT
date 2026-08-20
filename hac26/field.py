"""Implicit shape representation: a convex core plus a signed volumetric correction.

    f(y) = max_j (n_j . y - h_j) + Delta(y),   Delta(y) = sum_k g_k exp(-||(y-p_k)/sigma||^2/2)

The core is an intersection of half-spaces on fixed design normals, so its zero level set is a
convex polytope whose faces are exactly the planes n_j . y = h_j. Delta is a signed correction
on a fixed volumetric lattice, which is what carries non-convexity, and h itself carries a
band-limited correction dh because the convex stage that supplies h cannot see concavity.

Plain max, not log-sum-exp. Autodiff routes the subgradient of a max to the argmax, which is
the half-space that owns the surface at that point; log-sum-exp instead biases the zero set
inward by log(J)/beta, shrinking every body.

Delta stays signed. A one-sided activation could only carve or only grow. Concavity is a
deficit, but the same field must also be able to push out a lobe, so forcing the sign would
turn the representation into a bias.

The kernel width is a fixed fraction of the lattice spacing, so the finest feature Delta can
express is set by the kernel and the coarsest by the box; the coarsest feature the core can
express is set by the design density. Nothing here scales with `radius`: every caller now
decodes in the canonical frame (xy r_max = 1) and restores the published width afterwards with
fit_to_cylinder, which is the convention hac26/shapes.py::canonicalize_r states.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["spherical_design", "design_sha", "DESIGN_N", "DESIGN_T", "DESIGN_ITERS",
           "ConvexCore", "GaussianLattice", "ImplicitBody", "extract_mesh",
           "apply_constraints", "LATTICE_SHAPE", "LATTICE_EXTENT", "LATTICE_ALPHA",
           "N_SITES", "N_DIR", "CODE_DIM", "SH_DEGREE", "dir_design", "sh_expand"]

DESIGN_N = 4096        # a design of N normals gives facets ~4/sqrt(N) across and leaves a
                       # bulge ~2/N of the support distance at each face centre. This size
                       # needs a committed hac26/design4096.npy: above 512 the build is ~15 h
                       # on a CPU core, so spherical_design() refuses it rather than hanging.
DESIGN_T = 10          # spherical design strength
DESIGN_ITERS = 4000    # the only iteration count whose result is allowed into the cache
CORE_CHUNK_ELEMS = 6e7 # cap on the (points x normals) intermediate, ~240 MB in float32

# ------------------------------------------------------------------- the correction
#
# Delta(y) = sum_k g_k exp(-||(y - p_k)/sigma||^2 / 2)   on a FIXED lattice of sites p_k.
#
# The sites are not learned and are not part of the code; the code IS g. The predecessor was
# 32 cross-attention tokens whose softmax weights sum to one, so co-located tokens AVERAGED
# rather than added -- you could not carve deeper by clustering, and `sigma` was not a reach
# but the softness of a Voronoi partition of all space. Measured, a surface arrangement
# saturates at 0.36 R of carve depth (identical to four decimals from N=513 to N=4095; the
# shell's numerical rank saturates at 838) while a volumetric lattice reaches 0.64 R. At the
# body centre a shell's reachability is 0.47 against a lattice's 1690.
LATTICE_SHAPE = (12, 12, 12)      # 8^3 leaves 9 disconnected components at extreme aspect,
N_SITES = 12 * 12 * 12            # 10^3 leaves 1, and 12^3 -> 16^3 buys under 0.003 Dice
LATTICE_EXTENT = 1.1              # half-width of the site box. The challenge pose puts every
                                  # body inside [-1,1]^3 (z exactly, xy by canonicalize_r), so
                                  # cell centres at +-1.008 straddle the surface rather than
                                  # sitting on it. The lattice is FRAME-FIXED: it is not
                                  # scaled by `radius`, because every call site now decodes at
                                  # radius 1.0 and restores the width with fit_to_cylinder.
LATTICE_ALPHA = 0.9               # sigma = LATTICE_ALPHA * spacing, per axis. Monotone in
                                  # alpha; at alpha <= 0.6 and N >= 1728 the fit punches a
                                  # hole clean through a bilobe waist (Dice 0.95-0.97).
LATTICE_CHUNK_ELEMS = 6e6         # cap on the (points x sites) intermediate. NOT the same as
                                  # CORE_CHUNK_ELEMS: measured at a 64^3 grid, 6e6 beats 6e7
                                  # by 3.3x on CPU (cache-bound) while a GPU wants the
                                  # opposite, so this is raised on CUDA in GaussianLattice.

# ------------------------------------------------------------------- the support correction
SH_DEGREE = 5                     # dh is band-limited to this spherical-harmonic degree.
                                  # White dh at 2% R kills 3% of facets at N=64, 7% at 128,
                                  # 27% at 256 and 53% at 512 -- and a dead facet has an
                                  # exactly zero row in J, i.e. no gradient at all. Degree <= 4
                                  # kills 0% at every N up to 512, and (5+1)^2 = 36 is about
                                  # the 35-40 dh directions the curves actually constrain.
N_DIR = 128                       # dh is carried as samples on this many design directions,
                                  # which is what SphereConv needs; the band limit is enforced
                                  # structurally by sh_expand(), not by a penalty.
CODE_DIM = N_DIR + N_SITES        # 1856


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
                     iters: int = DESIGN_ITERS) -> np.ndarray:
    """n unit normals forming a spherical t-design, with the six axis directions included.

    The axis directions are pinned. The reference case is a
    cube, and the core is an intersection of halfspaces tangent to the target: it reproduces
    a cube EXACTLY only if the cube's own face normals are available. A design free to drift
    off the axes leaves the nearest normal some degrees away, and the intersection then
    bulges at every face centre. Pinning the six coordinate axes costs the design property
    almost nothing (the residual is reported by `_design_residual` and asserted in the tests)
    and makes flat-faced bodies -- which every ground truth is -- exactly representable.

    The remaining n - 6 points are optimised to kill the harmonic sums that define the
    design, starting from a Fibonacci spiral.

    Whatever is generated here is cached beside this file as design{n}.npy, so the two-minute
    build at n=512 is paid once per checkout rather than once per ImplicitBody().

    Above n=512 it is refused rather than built. The objective is a pair of n x n Gram
    matrices per Legendre order: measured on one CPU core at n=4096 that is 13.8 s and 3.6 GB
    per iteration, so the default 4000 iterations take about 15 hours. Building that silently
    inside a constructor is indistinguishable from a hang, and it would run inside every
    worker of a process pool. The error names the one command that fixes it.
    """
    cache = Path(__file__).with_name(f"design{n}.npy")
    if cache.exists() and t == DESIGN_T:
        x = np.load(cache)
        if len(x) == n:
            return x
    if n > 512:
        raise FileNotFoundError(
            f"{cache} is missing. Generate it once with "
            f"`python scripts/make_design.py --n {n}` (add --device cuda if you have a GPU: "
            f"on a CPU core this is ~15 h at n=4096, minutes on a GPU). It is cached "
            f"afterwards, and hac26/design64.npy and design512.npy are the sizes this "
            f"function will build for you on the spot.")
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
        v = abs(float(loss))          # the sum cancels to ~1e-15 and can go slightly
        if v < best:                  # negative; selecting on the signed value latches
            best, best_x = v, x.detach().clone()   # onto numerical noise, not the minimum
    out = best_x.numpy()
    if t == DESIGN_T and iters == DESIGN_ITERS:
        _write_design_cache(cache, out)
    return out


def _write_design_cache(cache: Path, x: np.ndarray) -> None:
    """Publish the design atomically, so a concurrent process cannot read a half-written file.

    Only ever called for the default `iters`: the filename keys on `n` alone, so caching a
    short debug or test build would silently become what every later caller gets. A 20-iter
    design has residual 7.7e-04 against 4.8e-07 for a full one -- 1600x worse, below what the
    tests demand, and with an identical `design_sha`, so nothing downstream could detect it.

    Generation is deterministic (fixed Fibonacci start, Adam on float64), so two processes that
    race produce the same array and whichever `os.replace` lands last is still correct. The
    rename is atomic on POSIX; writing straight to the destination is not.
    """
    import os
    import tempfile
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(cache.parent), suffix=".npy")
        os.close(fd)
        np.save(tmp, x)
        os.replace(tmp, cache)
    except OSError:
        if tmp is not None and Path(tmp).exists():
            try:
                os.unlink(tmp)        # do not leave a partial file behind
            except OSError:
                pass                  # a read-only install is not a reason to fail the run


def design_sha(x: np.ndarray) -> str:
    """Short digest of a design, so a corpus can refuse a design it was not fitted against.

    `design_n` alone does not identify the normals: the optimiser is deterministic on one
    build but two machines that generate the cache independently -- one on CPU, one on CUDA --
    can land on different points. The support vector h is indexed by normal, so mixing them
    silently reindexes every body in the corpus.
    """
    import hashlib
    # float32, because that is the precision ConvexCore stores the normals at: two designs
    # that agree there are interchangeable downstream, and digesting float64 would make the
    # cached file and the model's own buffer hash differently for the same design.
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


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

    def forward(self, y: torch.Tensor, chunk: int | None = None,
                h: torch.Tensor | None = None) -> torch.Tensor:
        """max_j (n_j . y - h_j), evaluated in chunks over query points.

        `h` may be supplied to override the stored support -- ImplicitBody passes the
        dh-corrected support that way, so the correction stays differentiable without
        ConvexCore having to know dh exists.

        The intermediate is (points x normals). At a 64^3 extraction grid that is 275k
        points, so it is 0.07 GB at 64 normals and 4.50 GB at 4096 -- the term that decides
        whether a large design is usable at all. Chunking bounds it without changing the
        result: the max is taken per point, so points never interact.
        """
        n_norm = self.n.shape[0]
        hh = self.h if h is None else h
        if chunk is None:
            chunk = max(4096, int(CORE_CHUNK_ELEMS // max(n_norm, 1)))
        if y.shape[0] <= chunk:
            return (y @ self.n.T - hh).amax(dim=-1)             # plain max, not LSE
        return torch.cat([(y[i:i + chunk] @ self.n.T - hh).amax(dim=-1)
                          for i in range(0, y.shape[0], chunk)], dim=0)


def _grid_lattice(shape=LATTICE_SHAPE, extent: float = LATTICE_EXTENT):
    """Cell centres of an axis-aligned grid over [-extent, extent]^3, and the spacing.

    Centres, not corners: with 12 cells over +-1.1 the outermost centre is at +-1.0083, so
    the lattice straddles the surface of a posed body (which reaches exactly 1.0) instead of
    placing a site exactly on it, where its gradient with respect to the surface is weakest.
    """
    axes, spacing = [], []
    for n_ax in shape:
        step = 2.0 * extent / n_ax
        axes.append(-extent + (np.arange(n_ax) + 0.5) * step)
        spacing.append(step)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return grid.astype(np.float32), np.asarray(spacing, dtype=np.float32)


# ----------------------------------------------------------- the band-limited dh basis

def dir_design(n: int = N_DIR) -> np.ndarray:
    """The directions dh is sampled on. A spherical design, so the quadrature is unbiased."""
    return spherical_design(n)


def _real_sh(x: np.ndarray, degree: int = SH_DEGREE) -> np.ndarray:
    """Un-normalised real spherical harmonics up to `degree`, evaluated at unit vectors x.

    Normalisation is irrelevant here: this basis is only ever used through a pseudo-inverse,
    which is invariant to a rescaling of the columns. What matters is that the columns span
    exactly the degree <= `degree` subspace, which the associated Legendre construction gives.
    """
    from scipy.special import lpmv
    x = np.asarray(x, dtype=np.float64)
    ct = np.clip(x[:, 2], -1.0, 1.0)
    ph = np.arctan2(x[:, 1], x[:, 0])
    cols = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            p_lm = lpmv(abs(m), l, ct)
            if m > 0:
                cols.append(p_lm * np.cos(m * ph))
            elif m < 0:
                cols.append(p_lm * np.sin(-m * ph))
            else:
                cols.append(p_lm)
    return np.stack(cols, 1)                       # (len(x), (degree+1)^2)


def sh_expand(src: np.ndarray, dst: np.ndarray, degree: int = SH_DEGREE) -> np.ndarray:
    """Matrix taking dh sampled on `src` directions to dh sampled on `dst` directions.

    `Y_dst @ pinv(Y_src)`. Because Y has only (degree+1)^2 columns, this matrix is also the
    band-limit projector: whatever the flow emits on `src`, what reaches h is the degree
    <= SH_DEGREE part of it. The band limit is therefore structural rather than a penalty the
    sampler is free to ignore, which matters because an out-of-band dh kills facets, and a
    dead facet has an exactly zero row in J -- no gradient at all, not merely a bad one.
    """
    y_src, y_dst = _real_sh(src, degree), _real_sh(dst, degree)
    return (y_dst @ np.linalg.pinv(y_src)).astype(np.float32)


# ----------------------------------------------------------------- the correction field

class GaussianLattice(nn.Module):
    """Delta(y) = sum_k g_k exp(-||(y - p_k)/sigma||^2 / 2), sites fixed, g learned.

    `g` is the only parameter and IS the code: no latent, no learned positions, no shared
    decoder, and nothing to load from disk. Amplitudes ADD where the kernels overlap, which
    is the property the cross-attention predecessor lacked -- its softmax weights summed to
    one, so clustering averaged instead of deepening.

    Delta is signed and is NOT made zero-mean. The predecessor subtracted the mean over the
    call's own points, which made Delta a function of the batch: `extract_mesh` evaluates the
    grid in chunks, so each chunk had a different constant subtracted and the extracted
    surface stepped at the chunk boundaries. A fixed-site sum is batch-independent by
    construction, and the overall offset it might otherwise absorb belongs to h.
    """

    def __init__(self, shape=LATTICE_SHAPE, extent: float = LATTICE_EXTENT,
                 alpha: float = LATTICE_ALPHA):
        super().__init__()
        sites, spacing = _grid_lattice(shape, extent)
        # persistent=False: derived from the constants above, identical for every body. A
        # checkpoint that carried them would silently redefine another body's lattice.
        self.register_buffer("p", torch.from_numpy(sites), persistent=False)
        self.register_buffer("inv2", torch.from_numpy(
            (1.0 / (alpha * spacing) ** 2).astype(np.float32)), persistent=False)
        self.register_buffer("pb", (self.p ** 2 * self.inv2).sum(1), persistent=False)
        self.g = nn.Parameter(torch.zeros(len(sites)))

    def forward(self, y: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """Evaluated as ONE matmul per chunk, not as a broadcast difference.

            ||(y - p)/s||^2 = sum_d y_d^2/s_d^2 + sum_d p_d^2/s_d^2 - 2 (y/s^2).p

        The obvious form materialises a (points x sites x 3) intermediate. Measured on two
        CPU cores at a 64^3 extraction grid: 14,573 ms for the broadcast against 3,773 ms
        here, agreeing to 1.7e-05 (float32 noise). The cross term is a (n,3) @ (3,S) matmul,
        which is what BLAS and cuBLAS exist for.
        """
        if chunk is None:
            elems = LATTICE_CHUNK_ELEMS * (16.0 if y.is_cuda else 1.0)
            chunk = max(1024, int(elems // max(len(self.p), 1)))
        out = []
        for i in range(0, y.shape[0], chunk):
            q = y[i:i + chunk]
            d2 = (q ** 2 * self.inv2).sum(1, keepdim=True) \
                + self.pb[None] - 2.0 * ((q * self.inv2) @ self.p.T)
            out.append(torch.exp(-0.5 * d2.clamp_min(0.0)) @ self.g)
        return torch.cat(out) if len(out) > 1 else out[0]


class ImplicitBody(nn.Module):
    """f(y) = max_j (n_j . y - h_j) + Delta(y).

    There is no scale factor in front of Delta any more. `s = CORE_SCALE * radius` was
    algebraically redundant -- s * sum_k g_k E_k is identically sum_k (s g_k) E_k, and fits at
    s = 0.15 R and s = 1 agreed to 8.7e-11 -- but it silently set the units g had to learn in,
    and therefore the effective learning rate and the scale the flow had to whiten away.

    h carries an optional band-limited correction dh, sampled on `dir_design(N_DIR)` and
    expanded onto the full design by sh_expand. It is applied INSIDE the softplus, so
    positivity is automatic and no clamp is needed:

        h = softplus( inv_softplus(h_base) + expand(dh) )

    dh exists because h_base comes from the convex stage, which assumes convexity. A
    non-convex body is darker than its own hull from self-shadowing, so that inversion
    explains darkness with shape and h_base is wrong in a direction that always favours a
    convex answer. Freezing h bakes that in.

    The band limit is EXACT on the argument of the softplus and approximate on h itself:
    d/dx softplus = sigmoid is not constant, and h varies across normals, so the induced
    change in h carries about 5% out-of-band content. That is first-order -- it does not
    shrink with the perturbation -- and it is small against the fully-white dh that kills 7%
    of facets at N=128, but it is not zero.
    """

    def __init__(self, radius: float = 1.0, normals: np.ndarray | None = None,
                 n_normals: int = DESIGN_N, with_dh: bool = True):
        super().__init__()
        if normals is None:
            normals = spherical_design(n_normals)
        self.radius = float(radius)
        self.core = ConvexCore(normals)
        self.delta = GaussianLattice()
        self.with_dh = bool(with_dh)
        if with_dh:
            self.dh = nn.Parameter(torch.zeros(N_DIR))
            self.register_buffer("dh_expand",
                                 torch.from_numpy(sh_expand(dir_design(N_DIR), normals)),
                                 persistent=False)

    def set_support(self, h) -> None:
        """Set the BASE support, i.e. the origin dh is measured from."""
        self.core.set_support(h)

    def support(self) -> torch.Tensor:
        """The support actually used: softplus(raw_h + expand(dh)).

        The base is `core.raw_h` itself, NOT a frozen copy of it. An earlier draft kept a
        separate non-persistent `raw_h_base` buffer, which made `core.raw_h` unreachable from
        the loss -- and `scripts/fit_shapes.py` optimises exactly that tensor, so the corpus
        fit would have silently stopped learning h while still reporting a falling loss. The
        two roles are separable by call, not by storage: the corpus fit leaves dh at zero and
        learns raw_h, and reconstruction pins raw_h to the convex stage's answer and learns dh.
        """
        if not self.with_dh:
            return self.core.h
        return F.softplus(self.core.raw_h + self.dh_expand @ self.dh)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.core(y, h=self.support()) + self.delta(y)


# ----------------------------------------------------------------- extraction

def extract_mesh(field, extent: float, res: int = 128, device: str = "cpu",
                 chunk: int = 262144):
    """FlexiCubes on a res^3 grid. Not Marching Cubes -- see the module docstring of
    hac26.vendor.flexicubes.

    `extent` must cover the correction lattice and its kernels, or sites near the edge of the
    box are evaluated nowhere and the surface they shape is silently absent from the mesh.
    """
    reach = LATTICE_EXTENT + 3.0 / float(np.sqrt(GaussianLattice().inv2.min()))
    if extent < reach - 1e-6:
        raise ValueError(
            f"extract_mesh extent {extent:.3f} does not cover the correction lattice, which "
            f"reaches {reach:.3f} (LATTICE_EXTENT {LATTICE_EXTENT} plus 3 sigma). Sites "
            f"outside the grid shape a surface the extraction cannot see.")
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

    The published radius is treated as an approximation with tolerance `tol` rather than a
    hard bound: two of the three public bodies exceed their own published R when posed this
    way, so clamping would shrink true geometry.
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

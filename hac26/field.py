"""The shape representation: a convex core plus a signed correction on a fixed lattice.

    f(y) = max_j (n_j . y - h_j) + q(y/|y|) + Delta(y),
    Delta(y) = sum_k g_k exp(-||(y-p_k)/sigma||^2/2),   q(u) = sum_lm c_lm Ybar_lm(u)

The body is where f < 0. The core is the intersection of half-spaces on fixed normals n_j
with support values h_j, so on its own it is a convex polytope whose faces lie in the planes
n_j . y = h_j. Delta is a sum of Gaussian bumps with signed amplitudes g_k on fixed sites p_k;
it is the only part that can make the body non-convex. The convex stage that supplies h
cannot see concavities and so gets the hull slightly wrong, and two corrections to it are
available: dh, a band-limited correction to the support values themselves, and q, a
degree-two field added to f, whose coefficients are displacements in body units. A caller
uses one or the other.

The max is a plain max, not a smooth one. Autodiff sends the gradient to the half-space that
owns the surface at that point; a smooth maximum would pull every face inward.

Delta is signed because a concavity removes material and a lobe adds it, and the same field
must be able to do both.

Everything here lives in the canonical frame (the body's z extent is [-1, 1] and its xy
radius is 1). Callers restore the published width afterwards with fit_to_cylinder; see
hac26/shapes.py::canonicalize_r.
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
           "EXTRACT_EXTENT", "EXTRACT_RES", "RADIAL_DEGREE", "N_RADIAL", "radial_basis",
           "radial_field",
           "lattice_kernel",
           "N_SITES", "N_DIR", "CODE_DIM", "SH_DEGREE", "dir_design", "sh_expand",
           "support_resample", "support_resample_weights"]

DESIGN_N = 4096        # number of core normals. More normals give smaller facets. This size
                       # needs the committed hac26/design4096.npy: building a design this
                       # large takes hours on a CPU, so spherical_design() refuses to do it
                       # inside a constructor.
DESIGN_T = 10          # spherical design strength
DESIGN_ITERS = 4000    # the only iteration count whose result is allowed into the cache
CORE_CHUNK_ELEMS = 6e7 # cap on the (points x normals) intermediate, in float32 elements

# ------------------------------------------------------------------- the correction
#
# Delta(y) = sum_k g_k exp(-||(y - p_k)/sigma||^2 / 2)   on a FIXED lattice of sites p_k.
#
# The sites are not learned and are not part of the code; the code is g. Amplitudes add where
# kernels overlap, so several sites together can carve deeper than one.
LATTICE_SHAPE = (24, 24, 24)      # sites per axis. What decides this is how sharp a surface
                                  # the lattice can make, because the curves are far more
                                  # sensitive to the sharpness of a carve than the voxel
                                  # overlap is: a carve the lattice can only hold blurred
                                  # fits the curves worse than no carve at all, however well
                                  # it overlaps the body. notes/representation.md measures
                                  # what each size can hold.
N_SITES = 24 * 24 * 24
LATTICE_EXTENT = 1.1              # half-width of the site box. The canonical pose puts every
                                  # body inside [-1, 1]^3; the box is a little wider so that
                                  # cell centres straddle the surface instead of sitting on it.
LATTICE_ALPHA = 0.75              # sigma = LATTICE_ALPHA * spacing, per axis. Too small and
                                  # the kernels stop overlapping, and a fit can punch a hole
                                  # through a thin waist; too large and no combination of
                                  # amplitudes makes a surface sharp enough to fit the curves.
EXTRACT_EXTENT = 1.6              # half-width of the grid extract_mesh runs on, in the canonical
                                  # frame. It has to cover the lattice plus its kernels;
                                  # extract_mesh checks that.
EXTRACT_RES = 96                  # side of that grid. It has to resolve the kernels, whose
                                  # width is LATTICE_ALPHA times the site spacing, or the
                                  # extraction is a coarser body than the one the amplitudes
                                  # describe; extract_mesh reports the ratio.
LATTICE_CHUNK_ELEMS = 1.2e7       # cap on the (points x sites) intermediate. A small chunk is
                                  # faster on a CPU and slower on a GPU, so GaussianLattice
                                  # raises it on CUDA; the raised value is what has to fit
                                  # beside the rest of a render on an eight-gigabyte card.

# ------------------------------------------------------------------- the support correction
SH_DEGREE = 5                     # dh is band-limited to this spherical-harmonic degree. A
                                  # rough dh removes facets from the polytope, and a removed
                                  # facet has no area and therefore no gradient.
N_DIR = 128                       # dh is carried as samples on this many design directions,
                                  # which is what SphereConv needs; the band limit is applied
                                  # by sh_expand() when dh is expanded onto the core normals.
CODE_DIM = N_DIR + N_SITES


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

    Minimising raw monomial means instead does not work: for even l the points being unit
    vectors forces sum_i x_i^2 = n/3, so that target is unreachable.
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

    The axis directions are pinned so that an axis-aligned box is exactly representable: the
    core reproduces a face exactly only if that face's normal is one of its normals. Pinning
    six points costs the design property almost nothing (`_design_residual` reports it and
    the tests check it).

    The remaining n - 6 points are optimised to zero the harmonic sums that define the
    design, starting from a Fibonacci spiral.

    The result is cached beside this file as design{n}.npy. Large n is refused rather than
    built: the objective holds n x n matrices per Legendre order, so a large build takes
    hours, and running that silently inside a constructor looks like a hang. The error names
    the command that builds and caches the file.
    """
    cache = Path(__file__).with_name(f"design{n}.npy")
    if cache.exists() and t == DESIGN_T:
        x = np.load(cache)
        if len(x) == n:
            return x
    if n > 512:
        raise FileNotFoundError(
            f"{cache} is missing. Generate it once with "
            f"`python scripts/make_design.py --n {n}` (add --device cuda if you have a GPU; "
            f"on a CPU this takes hours). It is cached afterwards.")
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
    """Write the design file atomically, so a concurrent process cannot read a half-written
    file.

    Only called for the default `iters`: the filename keys on `n` alone, so caching a short
    debug build would silently become what every later caller gets.

    Generation is deterministic, so two processes that race produce the same array and
    whichever rename lands last is still correct.
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

    The number of normals alone does not identify them: two machines that build the cache
    independently can land on different points. h is indexed by normal, so mixing two
    designs silently changes what every entry of h means.
    """
    import hashlib
    # float32, the precision ConvexCore stores the normals at, so the cached file and the
    # model's own buffer hash the same for the same design.
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
        """max_j (n_j . y - h_j) at the query points y, in chunks.

        `h` may be supplied to override the stored support. ImplicitBody passes the
        dh-corrected support that way, so the correction stays differentiable.

        The (points x normals) intermediate is the memory cost of the whole field. Chunking
        bounds it without changing the result, since the max is taken per point.
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
    """Cell centres of an axis-aligned grid over [-extent, extent]^3, and the spacing per
    axis."""
    axes, spacing = [], []
    for n_ax in shape:
        step = 2.0 * extent / n_ax
        axes.append(-extent + (np.arange(n_ax) + 0.5) * step)
        spacing.append(step)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return grid.astype(np.float32), np.asarray(spacing, dtype=np.float32)


# ----------------------------------------------------------- the band-limited dh basis

def dir_design(n: int = N_DIR) -> np.ndarray:
    """The directions dh is sampled on: a spherical design of n points."""
    return spherical_design(n)


def _real_sh(x: np.ndarray, degree: int = SH_DEGREE) -> np.ndarray:
    """Real spherical harmonics up to `degree` at the unit vectors x, without normalisation.

    The basis is only used through a pseudo-inverse, which does not care how the columns are
    scaled. What matters is that the columns span exactly the harmonics of degree <= `degree`.
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


RADIAL_DEGREE = 2                          # band limit of the radial reshaping term
N_RADIAL = (RADIAL_DEGREE + 1) ** 2        # its coefficients


def radial_basis(u: torch.Tensor) -> torch.Tensor:
    """The real spherical harmonics of degree at most RADIAL_DEGREE at the unit vectors u
    (n, 3), as (n, N_RADIAL), each scaled to unit root mean square over the sphere.

    Unit root mean square rather than unit integral so that a coefficient is a length. The
    degree-zero column is then identically one, and a coefficient vector of size s displaces
    the surface of a convex core by about s in body units, because on a facet of a polytope
    the core's gradient has unit norm and adding a constant to the field moves the level set
    by that constant.
    """
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    r3, r15 = np.sqrt(3.0), np.sqrt(15.0)
    one = torch.ones_like(x)
    return torch.stack([
        one,
        r3 * y, r3 * z, r3 * x,
        r15 * x * y, r15 * y * z,
        0.5 * np.sqrt(5.0) * (3.0 * z ** 2 - 1.0),
        r15 * x * z, 0.5 * r15 * (x ** 2 - y ** 2)], -1)


def radial_field(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """sum_lm c_lm Ybar_lm(y / |y|) at the points y (n, 3).

    The direction of a point is undefined at the origin, which is inside every body the
    challenge poses and never on a level set, so it is guarded and not special-cased.
    """
    return radial_basis(y) @ c


def support_resample_weights(src: np.ndarray, dst: np.ndarray, k: int = 6):
    """The resampling of support_resample as (indices, weights), both (len(dst), k): the k
    source directions each destination direction reads, and their weights. The form to use
    when the dense matrix would be large."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    idx = np.argsort(-(dst @ src.T), axis=1)[:, :k]
    # One batched pseudo-inverse rather than a least-squares solve per direction. pinv gives
    # the least-norm solution of the underdetermined system, which is what lstsq returns.
    A = np.concatenate([np.ones((len(dst), 1, k)),
                        src[idx].transpose(0, 2, 1)], axis=1)         # (D, 4, k)
    b = np.concatenate([np.ones((len(dst), 1)), dst], axis=1)         # (D, 4)
    w = np.einsum("dkj,dj->dk", np.linalg.pinv(A), b)                 # (D, k)
    return idx, w.astype(np.float32)


def support_resample(src: np.ndarray, dst: np.ndarray, k: int = 6) -> np.ndarray:
    """Matrix that resamples a support function from `src` directions onto `dst` directions.

    This is not sh_expand. sh_expand keeps only low harmonic degrees, which is right for dh
    and wrong for h: the support function of a flat-faced body is not band-limited, and the
    public models include flat-faced bodies.

    Within one flat face a support function is an affine function of the direction, so
    weights over the k nearest source directions that reproduce affine functions are exact
    there and interpolate smoothly elsewhere. They are the least-norm solution of

        sum_i w_i = 1,      sum_i w_i n_i = u

    k is larger than the four points that would pin the constraints exactly, because a
    near-degenerate quadruple makes the weights blow up.
    """
    idx, w = support_resample_weights(src, dst, k)
    W = np.zeros((len(dst), len(np.asarray(src))), dtype=np.float32)
    np.put_along_axis(W, idx, w, axis=1)
    return W


def sh_expand(src: np.ndarray, dst: np.ndarray, degree: int = SH_DEGREE) -> np.ndarray:
    """Matrix taking dh sampled on `src` directions to dh sampled on `dst` directions.

    `Y_dst @ pinv(Y_src)`. Because Y has only (degree+1)^2 columns, this is also the band
    limit: whatever is emitted on `src`, only its part of degree <= `degree` reaches h. So
    the band limit is built in rather than encouraged by a penalty.
    """
    y_src, y_dst = _real_sh(src, degree), _real_sh(dst, degree)
    return (y_dst @ np.linalg.pinv(y_src)).astype(np.float32)


# ----------------------------------------------------------------- the correction field

class GaussianLattice(nn.Module):
    """Delta(y) = sum_k g_k exp(-||(y - p_k)/sigma||^2 / 2), sites fixed, g learned.

    `g` is the only parameter and is the code: no latent vector, no learned positions, no
    shared decoder. Amplitudes add where the kernels overlap.

    Delta is signed and is not made zero-mean. Subtracting a mean over the points of one call
    would make the field depend on how the points are batched; any overall offset belongs
    to h.
    """

    def __init__(self, shape=LATTICE_SHAPE, extent: float = LATTICE_EXTENT,
                 alpha: float = LATTICE_ALPHA):
        super().__init__()
        sites, spacing = _grid_lattice(shape, extent)
        # persistent=False: these follow from the constants above and are the same for every
        # body, so a checkpoint must not carry its own copy.
        self.register_buffer("p", torch.from_numpy(sites), persistent=False)
        self.register_buffer("inv2", torch.from_numpy(
            (1.0 / (alpha * spacing) ** 2).astype(np.float32)), persistent=False)
        self.register_buffer("pb", (self.p ** 2 * self.inv2).sum(1), persistent=False)
        self.g = nn.Parameter(torch.zeros(len(sites)))

    def forward(self, y: torch.Tensor, chunk: int | None = None,
                g: torch.Tensor | None = None) -> torch.Tensor:
        """Delta at the points y. `g` may be supplied to override the stored amplitudes, so a
        caller can differentiate through amplitudes it holds itself.

        The squared distances are expanded so the cross term is one matrix product:

            ||(y - p)/s||^2 = sum_d y_d^2/s_d^2 + sum_d p_d^2/s_d^2 - 2 (y/s^2).p

        which avoids a (points x sites x 3) intermediate.
        """
        gg = self.g if g is None else g
        if chunk is None:
            elems = LATTICE_CHUNK_ELEMS * (8.0 if y.is_cuda else 1.0)
            chunk = max(1024, int(elems // max(len(self.p), 1)))
        out = []
        for i in range(0, y.shape[0], chunk):
            q = y[i:i + chunk]
            d2 = (q ** 2 * self.inv2).sum(1, keepdim=True) \
                + self.pb[None] - 2.0 * ((q * self.inv2) @ self.p.T)
            out.append(torch.exp(-0.5 * d2.clamp_min(0.0)) @ gg)
        return torch.cat(out) if len(out) > 1 else out[0]


def lattice_kernel(shape=LATTICE_SHAPE, extent: float = LATTICE_EXTENT,
                   alpha: float = LATTICE_ALPHA, cut: float = 4.0):
    """The kernels of one lattice evaluated at the lattice's own sites, sparse.

    K[j, k] = phi_k(p_j), so K g is the field the amplitudes g make at the sites. It is what
    turns a requested carve depth into amplitudes and back, and a solver uses it to say what
    a coordinate is worth before spending a render on it.

    A kernel is dropped beyond `cut` standard deviations, where it is 3e-4 of its peak. The
    sites are a regular grid, so the neighbours within that radius are an index box and are
    found by arithmetic rather than by a search, which is what keeps this affordable when the
    lattice has more than ten thousand sites.
    """
    from scipy import sparse

    sites, spacing = _grid_lattice(shape, extent)
    sig = alpha * spacing                                       # (3,)
    n = np.asarray(shape, dtype=np.int64)
    half = np.ceil(cut * sig / spacing).astype(np.int64)
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    idx = idx.reshape(-1, 3)
    rows, cols, vals = [], [], []
    box = np.stack(np.meshgrid(*[np.arange(-h, h + 1) for h in half], indexing="ij"), -1)
    for d in box.reshape(-1, 3):
        j = idx + d
        ok = np.all((j >= 0) & (j < n), axis=1)
        if not ok.any():
            continue
        v = np.exp(-0.5 * (((d * spacing) / sig) ** 2).sum())
        if v < np.exp(-0.5 * cut ** 2):
            continue
        flat = j[ok, 0] * shape[1] * shape[2] + j[ok, 1] * shape[2] + j[ok, 2]
        rows.append(np.nonzero(ok)[0])
        cols.append(flat)
        vals.append(np.full(int(ok.sum()), v))
    m = int(np.prod(shape))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                     np.concatenate(cols))), shape=(m, m))


class ImplicitBody(nn.Module):
    """f(y) = max_j (n_j . y - h_j) + q(y/|y|) + Delta(y), with
    h = softplus(inv_softplus(h_base) + expand(dh)) and q the radial reshaping term.

    h_base is the support the body starts from: the hull support of a training body, or the
    convex stage's answer at reconstruction. dh is a band-limited correction to it, sampled on
    `dir_design(N_DIR)` and expanded onto the core normals by sh_expand. It is added inside the
    softplus, so h stays positive without a clamp.

    dh exists because the convex stage assumes convexity. A non-convex body is darker than
    its own hull because it shadows itself, and a convex inversion explains that darkness with
    shape, so h_base is wrong in a direction that favours a convex answer.

    The band limit is exact on the argument of the softplus and only approximate on h itself,
    because the slope of softplus varies across normals.

    q is the same correction written differently, and a caller uses one or the other. It is
    the degree-two part alone, added to the field rather than inside the softplus, so a
    coefficient is a displacement in body units and the reshaping enters linearly. What the
    convex stage gets wrong about the hull of a non-convex body is almost all of degree two,
    and a fit that must move the hull and carve it in the same step needs the reshaping in as
    few coordinates as it can be written in.
    """

    def __init__(self, normals: np.ndarray | None = None, lattice_shape=LATTICE_SHAPE):
        super().__init__()
        if normals is None:
            normals = spherical_design(DESIGN_N)
        self.core = ConvexCore(normals)
        self.delta = GaussianLattice(shape=lattice_shape)
        self.dh = nn.Parameter(torch.zeros(N_DIR))
        self.register_buffer("dh_expand",
                             torch.from_numpy(sh_expand(dir_design(N_DIR), normals)),
                             persistent=False)

    def set_support(self, h) -> None:
        """Set the base support h_base, the origin dh is measured from."""
        self.core.set_support(h)

    def support(self, dh: torch.Tensor | None = None) -> torch.Tensor:
        """The support actually used: softplus(raw_h + expand(dh)), with `dh` overriding the
        stored correction when given. raw_h is the parameter itself, not a copy, so h stays
        learnable through this call if a caller wants that."""
        d = self.dh if dh is None else dh
        return F.softplus(self.core.raw_h + self.dh_expand @ d)

    def forward(self, y: torch.Tensor, dh: torch.Tensor | None = None,
                g: torch.Tensor | None = None, c: torch.Tensor | None = None) -> torch.Tensor:
        """f at the points y. `dh`, `g` and `c` override the stored code when given, so a
        caller can differentiate f through parameters it holds itself. `c` is the radial
        reshaping term and is absent unless it is passed."""
        f = self.core(y, h=self.support(dh)) + self.delta(y, g=g)
        return f if c is None else f + radial_field(y, c)


# ----------------------------------------------------------------- extraction

_GRID_CACHE: dict = {}
_REACH = None


def _voxel_grid(res: int, device):
    """FlexiCubes plus its voxel grid, cached per (res, device). Building the grid is slow and
    depends on nothing else."""
    from .vendor.flexicubes import FlexiCubes

    key = (int(res), str(device))
    if key not in _GRID_CACHE:
        fc = FlexiCubes(device=device)
        _GRID_CACHE[key] = (fc,) + tuple(fc.construct_voxel_grid(res))
    return _GRID_CACHE[key]


def kernel_pitch_ratio(res: int, extent: float = EXTRACT_EXTENT) -> float:
    """Kernel width over grid pitch of an extraction. Below one the grid does not resolve the
    correction's own kernels and the extracted body is coarser than the amplitudes describe;
    the operator's default resolution keeps it at about two."""
    sigma = LATTICE_ALPHA * 2.0 * LATTICE_EXTENT / max(LATTICE_SHAPE)
    return sigma / (2.0 * extent / res)


def extract_mesh(field, extent: float, res: int = EXTRACT_RES, device: str = "cpu",
                 chunk: int = 262144, grad: bool = False):
    """The surface f = 0 as a triangle mesh, by FlexiCubes on a res^3 grid over
    [-extent, extent]^3.

    With `grad=False` the result is a pair of numpy arrays. With `grad=True` the vertices are
    a torch tensor that is differentiable through the field values on the grid, so a caller
    that computes `field` from tensors it holds can differentiate the mesh with respect to
    them; the faces are then a torch long tensor.

    `extent` must cover the lattice and its kernels; otherwise sites near the edge of the box
    shape a surface the grid never sees.
    """
    global _REACH
    if _REACH is None:
        _REACH = LATTICE_EXTENT + 3.0 / float(GaussianLattice().inv2.min().sqrt())
    reach = _REACH
    if extent < reach - 1e-6:
        raise ValueError(
            f"extract_mesh extent {extent:.3f} does not cover the correction lattice, which "
            f"reaches {reach:.3f} (LATTICE_EXTENT {LATTICE_EXTENT} plus 3 sigma). Sites "
            f"outside the grid shape a surface the extraction cannot see.")
    fc, x_nx3, cube_fx8 = _voxel_grid(res, device)
    x_nx3 = x_nx3 * (2.0 * extent)                       # grid spans [-extent, extent]
    with torch.set_grad_enabled(grad):
        sdf = torch.cat([field(x_nx3[i:i + chunk]) for i in range(0, len(x_nx3), chunk)])
        verts, faces, _ = fc(x_nx3, sdf, cube_fx8, res, training=False)
    if grad:
        return verts, faces.long()
    return verts.detach().cpu().numpy(), faces.detach().cpu().numpy()


def apply_constraints(verts: np.ndarray, radius: float, tol: float = 0.03) -> np.ndarray:
    """Apply the challenge's pose constraints to extracted vertices.

    z is rescaled so the body touches -1 and +1 exactly, which the challenge states as
    equalities. The xy radius is then brought down to `radius` only if it exceeds it by more
    than `tol`: the published radius is approximate, and a posed public body can sit slightly
    past it, so clamping hard would shrink true geometry.
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

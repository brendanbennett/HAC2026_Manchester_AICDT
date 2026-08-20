"""Implicit training corpus, stratified by hull-deficit statistics.

WHY THIS EXISTS
---------------
`lpd_flow` learns a correction on top of the convex hull, so the only thing that matters
about a training body is the field it induces:

    D(u) = ( r_hull(u) - r_body(u) ) / mean(r_hull)

Two summary numbers describe it: `D_rms`, the size of the correction the flow must output,
and `D_lo`, the fraction of D's power at spherical-harmonic degrees l <= 4. Measured on the
public models (`hac26/scoring` frame, challenge pose):

    model 1   D_rms 0.003   D_lo 0.17     (near-sphere, effectively convex)
    model 2   D_rms 0.000   D_lo 0.00     (cube of side 6, exactly convex)
    model 3   D_rms 0.201   D_lo 0.91     (deep waist; power almost entirely low-order)

The previous mesh corpora both failed here, in the same way and for different reasons.
Boring holes into a convex body gave D_rms 0.060 with D_lo 0.36 -- a third of the amplitude
and most of it above l = 4, which is the part of the spectrum model 3 does not have. The
seven-archetype mesh corpus that replaced it was worse than it looked: five of its seven
classes measured D_rms <= 0.021 with silhouette convexity deficit at the rasterisation
noise floor, i.e. they were convex training examples, and the flow spent most of its
gradient learning to emit zero. Craters and small basins are essentially invisible to both
challenge measures -- the side-view measure compares silhouette boundary curves, and a
crater does not change a silhouette from any direction.

WHAT THIS DOES DIFFERENTLY
--------------------------
1. Implicit (SDF + marching cubes) rather than mesh booleans. Smooth-min unions give
   rounded necks instead of the sharp bite marks of a boolean subtract, and the output is
   a single watertight component by construction. That last point is not cosmetic: the old
   `rubble_pile` concatenated overlapping icosphere SHELLS, so it was not a solid at all --
   `m.volume` was meaningless for it and `fit_shapes.samples()` calls
   `trimesh.nearest.signed_distance`, which is undefined on a non-watertight mesh with
   interior surfaces. Those bodies were feeding the token library garbage occupancy.

2. Stratified acceptance, not a target. D_rms is binned and each bin gets a quota, with
   model 3's 0.20 as an anchor in the middle rather than a bullseye -- the secret models
   are ordered by increasing difficulty, so several are likely to be MORE concave than
   model 3, and a corpus collapsed onto model 3's exact statistics would be both a bad bet
   and hard to defend against the challenge's general-purpose-algorithm rule. `D_lo` is
   filtered with a LOWER BOUND only: that is what rejects the hole-punching failure mode
   without prescribing a shape.

3. A deliberate near-convex fraction (default 40%). Models 1 and 2 currently reconstruct
   correctly *because* the flow outputs approximately zero on them. The voxel measure is a
   symmetric difference, so a hallucinated concavity on a convex body is scored exactly as
   harshly as a missed one; a corpus that is all deep concavities would trade models 1, 2
   and 4 for model 3.

4. R is sampled first and imposed exactly. The published bounding-cylinder radii are the
   half-width against a fixed z half-height of 1, so R IS the aspect ratio. Nine of the ten
   sit in [0.67, 1.475] and model 10 is a lone outlier at 3.95; the previous corpus put only
   3.6% of samples within 10% of model 9's R and 0.7% within 10% of model 10's.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------
# grid / implicit helpers
# --------------------------------------------------------------------------------------
_GRID: dict = {}


def _grid(n: int = 64, lim: float = 1.7):
    key = (n, lim)
    if key not in _GRID:
        g = np.linspace(-lim, lim, n)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        _GRID[key] = (np.stack([X, Y, Z], -1), 2 * lim / (n - 1))
    return _GRID[key]


def _rand_rot(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.diag(r))


def _smin(a, b, k):
    """Polynomial smooth minimum; `k` is the blend radius, and it is the parameter that
    turns a pair of touching spheres into a body with a neck rather than two spheres."""
    h = np.clip(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _ellipsoid_sdf(P, c, ax, rot):
    q = (P - c) @ rot
    s = float(np.min(ax))
    return (np.linalg.norm(q / ax, axis=-1) - 1.0) * s


def _mesh_from(F: np.ndarray, sp: float, lim: float):
    """Marching cubes -> single watertight component, or None.

    The border is forced positive before extraction: a level set that reaches the edge of
    the grid produces an open surface, and an open surface has no interior, which silently
    breaks both `signed_distance` (fit_shapes) and any volume statistic.
    """
    import trimesh
    from skimage import measure
    F = F.copy()
    F[0, :, :] = F[-1, :, :] = 1.0
    F[:, 0, :] = F[:, -1, :] = 1.0
    F[:, :, 0] = F[:, :, -1] = 1.0
    if F.min() >= 0:
        return None
    try:
        v, f, *_ = measure.marching_cubes(F, 0.0, spacing=(sp, sp, sp))
    except (ValueError, RuntimeError):
        return None
    m = trimesh.Trimesh(v - lim, f, process=True)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda c: len(c.faces))
    m.remove_unreferenced_vertices()
    m.fix_normals()
    if len(m.faces) < 200 or not m.is_watertight:
        return None
    return m


# --------------------------------------------------------------------------------------
# directions and the spherical-harmonic basis (cached: the basis is the expensive part)
# --------------------------------------------------------------------------------------
def fib_dirs(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


_BASIS: dict = {}


def _sh_basis(dirs: np.ndarray, lmax: int):
    key = (len(dirs), lmax, float(dirs[0, 0]))
    if key in _BASIS:
        return _BASIS[key]
    from scipy.special import sph_harm_y
    th = np.arccos(np.clip(dirs[:, 2], -1, 1))
    ph = np.mod(np.arctan2(dirs[:, 1], dirs[:, 0]), 2 * np.pi)
    cols, deg = [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            y = sph_harm_y(l, abs(m), th, ph)
            y = y.real * np.sqrt(2) if m > 0 else (y.imag * np.sqrt(2) if m < 0 else y.real)
            cols.append(y)
            deg.append(l)
    _BASIS[key] = (np.stack(cols, 1), np.asarray(deg))
    return _BASIS[key]


def sh_field(dirs: np.ndarray, rng: np.random.Generator, sigma: float,
             corr_deg: float, lmax: int = 8) -> np.ndarray:
    """Zero-mean Gaussian random field on the sphere with a prescribed correlation angle.

    This is the Muinonen Gaussian-random-sphere construction: shorter correlation angle
    moves power to higher degrees. It replaces per-vertex i.i.d. jitter, which has power at
    the mesh resolution and reads as static rather than as lumpiness.
    """
    B, deg = _sh_basis(dirs, lmax)
    ell = np.maximum(deg, 1)
    w = np.exp(-0.5 * (ell * np.radians(corr_deg)) ** 2)
    w[deg == 0] = 0.0
    s = B @ (rng.normal(size=B.shape[1]) * w)
    return s / max(float(s.std()), 1e-9) * sigma


# --------------------------------------------------------------------------------------
# challenge pose and the bounding-cylinder radius
# --------------------------------------------------------------------------------------
def to_challenge_pose(m):
    """Rotation axis = z, body touching z = +-1, xy-centroid on the axis."""
    import trimesh
    v = np.asarray(m.vertices, float).copy()
    zmin, zmax = v[:, 2].min(), v[:, 2].max()
    if zmax - zmin < 1e-9:
        return None
    v[:, 2] -= 0.5 * (zmin + zmax)
    v *= 2.0 / (zmax - zmin)
    c = trimesh.Trimesh(v, m.faces, process=False)
    cen = np.asarray(c.centroid, float) if c.is_volume else v.mean(0)
    v[:, 0] -= cen[0]
    v[:, 1] -= cen[1]
    return trimesh.Trimesh(v, m.faces, process=False)


def cylinder_R(m) -> float:
    """Minimal bounding-cylinder radius: the minimum ENCLOSING CIRCLE of the xy projection.

    Taken about the vertex mean instead, the cube reads 1.46 against the published 1.42,
    because uneven tessellation pulls the mean off axis. The enclosing circle reproduces
    all three public values to 0.02 (1.11 / 1.41 / 0.86 vs 1.12 / 1.42 / 0.88).
    """
    from scipy.optimize import minimize
    from scipy.spatial import ConvexHull
    p = np.asarray(m.vertices, float)[:, :2]
    if len(p) > 3:
        try:
            p = p[ConvexHull(p).vertices]
        except Exception:
            pass
    r = minimize(lambda c: np.sqrt(((p - c) ** 2).sum(1)).max(), p.mean(0),
                 method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9,
                                                "maxiter": 4000})
    return float(r.fun)


def set_R(m, R: float):
    """Scale x,y so the bounding-cylinder radius equals R. z is untouched, so the body
    stays in the challenge pose. Exactly `hac26.recon.fit_to_cylinder`, applied at TRAIN
    time so the network never sees an aspect ratio it will not see at test time."""
    import trimesh
    cur = cylinder_R(m)
    if cur < 1e-9:
        return None
    v = np.asarray(m.vertices, float).copy()
    v[:, :2] *= R / cur
    return trimesh.Trimesh(v, m.faces, process=False)


def sample_R(rng: np.random.Generator, t: float | None = None) -> float:
    """Aspect-ratio prior, shaped by the published table (1.12, 1.42, 0.88, 1.475, 1.22,
    0.925, 1.205, 1.24, 0.67, 3.95). Tight body plus two deliberate tails: model 9 sits at
    0.67 and model 10 at 3.95, and a corpus without those tails cannot represent either.
    """
    u = rng.random()
    if t is not None and t > 0.30:
        w = float(np.clip((t - 0.30) / 0.70, 0.0, 1.0))     # 0 at t=0.3, 1 at t=1
        if u < 0.25 + 0.35 * w:
            return float(rng.uniform(0.60, 0.95))   # narrow, waisted -- the model 3 regime
        if u < 0.90 + 0.05 * w:
            return float(rng.uniform(0.95, 1.55))
        if u < 0.97:
            return float(rng.uniform(1.60, 2.60))
        return float(rng.uniform(3.00, 4.60))
    if u < 0.12:
        return float(rng.uniform(0.60, 0.84))       # model 9 regime
    if u < 0.82:
        return float(rng.uniform(0.85, 1.55))       # models 1-8
    if u < 0.92:
        return float(rng.uniform(1.60, 2.60))       # bridge, unoccupied but plausible
    return float(rng.uniform(3.00, 4.60))           # model 10 regime


# --------------------------------------------------------------------------------------
# the acceptance statistic
# --------------------------------------------------------------------------------------
_RAY_DIRS = None


def hull_deficit(m, n_ray: int = 642, lmax: int = 12, lo: int = 4) -> tuple:
    """(D_rms, D_lo) for the radial hull-deficit field. This is the acceptance test.

    Rays are cast from the hull centroid; a direction on which the ray misses the body
    entirely returns the full hull radius, which is the correct deficit for that direction.
    """
    global _RAY_DIRS
    if _RAY_DIRS is None or len(_RAY_DIRS) != n_ray:
        _RAY_DIRS = fib_dirs(n_ray)
    dirs = _RAY_DIRS
    hull = m.convex_hull
    origin = np.asarray(hull.centroid, float)

    def outer(mesh):
        org = np.repeat(origin[None], len(dirs), 0)
        loc, idx, _ = mesh.ray.intersects_location(org, dirs, multiple_hits=True)
        r = np.zeros(len(dirs))
        if len(loc):
            np.maximum.at(r, idx, np.linalg.norm(loc - origin, axis=1))
        return r

    r_h = outer(hull)
    r_b = outer(m)
    ok = r_h > 1e-9
    if ok.sum() < 0.5 * len(dirs):
        return np.nan, np.nan
    D = np.zeros(len(dirs))
    D[ok] = (r_h[ok] - r_b[ok]) / r_h[ok].mean()
    D_rms = float(np.sqrt((D ** 2).mean()))
    if D.std() < 1e-6:
        return D_rms, 0.0
    B, deg = _sh_basis(dirs, lmax)
    coef, *_ = np.linalg.lstsq(B, D, rcond=None)
    p = coef ** 2
    tot = p[deg >= 1].sum()
    D_lo = float(p[(deg >= 1) & (deg <= lo)].sum() / tot) if tot > 0 else 0.0
    return D_rms, D_lo


# --------------------------------------------------------------------------------------
# families.  `t` in [0, 1] is a depth hint: 0 = barely concave, 1 = deep.
# --------------------------------------------------------------------------------------
def _f_lobes_field(rng, t: float):
    """Smooth union of 2-3 lobes along the rotation axis, modulated by a random field.

    The lobe union supplies the low-order deficit (model 3's waist); the field breaks the
    snowman look and adds the irregularity real targets have. Measured at t ~ 0.8 this
    family lands at D_rms 0.19, D_lo 0.90 against model 3's 0.20 / 0.91.
    """
    P, sp = _grid()
    k = 2 if rng.random() < 0.65 else 3
    axis = np.array([0.0, 0.0, 1.0])
    if rng.random() < 0.25:                      # occasionally off-axis, for generality
        axis = axis + rng.normal(scale=0.35, size=3)
        axis /= np.linalg.norm(axis)
    blend = float(np.interp(t, [0, 1], [0.30, 0.12]))
    sep = float(np.interp(t, [0, 1], [0.50, 1.15]))
    F = None
    c = -axis * 0.5 * sep * (k - 1)
    for i in range(k):
        ax = rng.uniform(0.42, 0.78, size=3) * (1.0 if i == 0 else rng.uniform(0.7, 1.0))
        d = _ellipsoid_sdf(P, c, ax, _rand_rot(rng))
        F = d if F is None else _smin(F, d, blend)
        c = c + axis * sep + rng.normal(scale=0.09, size=3)
    m = _mesh_from(F, sp, 1.7)
    if m is None:
        return None
    return _modulate(m, rng, sigma=float(np.interp(t, [0, 1], [0.06, 0.26])),
                     corr_deg=float(np.interp(t, [0, 1], [45.0, 25.0])))


def _f_scalloped(rng, t: float):
    """Broad scallops carved by subtractor spheres whose CENTRES LIE OUTSIDE the body.

    This is the distinction that makes it work: a subtractor centred on the surface with a
    small radius is a hole (high-order deficit, invisible in silhouette); one centred
    outside with a radius comparable to the body removes a broad shell of material and
    changes the outline. Same operator, opposite spectrum.
    """
    P, sp = _grid()
    ax = rng.uniform(0.60, 0.95, size=3)
    F = _ellipsoid_sdf(P, np.zeros(3), ax, _rand_rot(rng))
    n_cut = int(np.interp(t, [0, 1], [2, 9]))
    for _ in range(max(1, n_cut)):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        r = float(np.interp(t, [0, 1], [0.35, 1.05])) * rng.uniform(0.85, 1.2)
        c = u * (float(ax.mean()) + r * rng.uniform(0.45, 0.85))
        F = np.maximum(F, -(np.linalg.norm(P - c, axis=-1) - r))
    return _mesh_from(F, sp, 1.7)


def _f_gauss_sphere(rng, t: float):
    """Gaussian random sphere: star-shaped, so it is always printable and connected, and
    at large sigma it is genuinely non-convex all around rather than in a few places."""
    import trimesh
    sub = 3
    base = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
    u = np.asarray(base.vertices, float)
    sigma = float(np.interp(t, [0, 1], [0.10, 0.48]))
    corr = float(np.interp(t, [0, 1], [50.0, 22.0]))
    s = sh_field(u, rng, sigma, corr)
    r = np.exp(s - 0.5 * sigma ** 2) * 0.8
    return trimesh.Trimesh(u * r[:, None], base.faces, process=True)


def _f_lobes_scallop(rng, t: float):
    """Hybrid: a lobed body then scalloped. Two independent concavity mechanisms on one
    body, which is what stops the flow from keying on a single generator signature."""
    m = _f_lobes_field(rng, t * 0.7)
    if m is None:
        return None
    P, sp = _grid()
    from scipy.spatial import cKDTree
    # rebuild an SDF from the mesh is expensive; carve on the implicit side instead by
    # re-running the lobe field is not possible here, so carve the mesh radially: pull
    # vertices inside a subtractor sphere back to its surface. Cheap, and equivalent for
    # the broad-scallop case because the subtractor is large relative to the facets.
    v = np.asarray(m.vertices, float)
    ext = float(np.abs(v).max())
    for _ in range(int(rng.integers(1, 4))):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        r = float(np.interp(t, [0, 1], [0.4, 0.95])) * ext
        c = u * (ext + r * rng.uniform(0.45, 0.8))
        d = np.linalg.norm(v - c, axis=1)
        sel = d < r
        if sel.any():
            v[sel] = c + (v[sel] - c) / d[sel, None] * r
    import trimesh
    return trimesh.Trimesh(v, m.faces, process=True)


def _f_near_convex(rng, t: float):
    """The convex / near-convex half of the corpus: models 1, 2 and 4 live here.

    Kept as meshes rather than implicit because the point of these is sharp facets and
    exact convexity, which marching cubes on a 64^3 grid would round off.
    """
    import trimesh
    from hac26.shapes import platonic
    pick = rng.random()
    if pick < 0.30:                                     # platonic, cube-weighted
        name = str(rng.choice(["cube", "tetra", "octa", "dodeca", "icosa"],
                              p=[0.44, 0.14, 0.14, 0.14, 0.14]))
        v = platonic(name) * rng.uniform(0.5, 1.0, size=3)
        return trimesh.convex.convex_hull(v)
    if pick < 0.60:                                     # faceted rock: hull of few points
        n = int(rng.integers(9, 18))
        p = rng.normal(size=(n, 3))
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        p *= rng.uniform(0.55, 1.0, size=(n, 1))
        p *= rng.uniform(0.8, 1.25, size=3)
        return trimesh.convex.convex_hull(p)
    if pick < 0.85:                                     # smooth lumpy, model-1-like
        return _f_gauss_sphere(rng, rng.uniform(0.0, 0.25))
    m = _f_gauss_sphere(rng, 0.15)                      # smooth body with a shallow basin
    return _modulate(m, rng, sigma=0.05, corr_deg=40.0)


def _modulate(m, rng, sigma: float, corr_deg: float):
    """Radial random-field modulation of an existing mesh."""
    import trimesh
    v = np.asarray(m.vertices, float)
    c = v.mean(0)
    u = v - c
    r = np.linalg.norm(u, axis=1)
    d = u / np.maximum(r, 1e-9)[:, None]
    r2 = r * np.exp(sh_field(d, rng, sigma, corr_deg))
    return trimesh.Trimesh(c + d * r2[:, None], m.faces, process=True)


_NONCONVEX = [_f_lobes_field, _f_lobes_field, _f_scalloped, _f_gauss_sphere,
              _f_lobes_scallop]


# --------------------------------------------------------------------------------------
# stratified sampler
# --------------------------------------------------------------------------------------
# Bin edges on D_rms. Model 3 (0.201) sits in the middle bin, as an ANCHOR: the secret
# models are ordered by increasing difficulty, so the bins above it are the ones most
# likely to matter and the corpus must reach them.
D_BINS = [(0.05, 0.11), (0.11, 0.17), (0.17, 0.24), (0.24, 0.32), (0.32, 0.45)]
D_LO_MIN = 0.70          # lower bound only -- this is what rejects punched holes
NEAR_CONVEX_MAX = 0.03   # models 1 and 2 measure 0.003 and 0.000


def _draw_one(rng, want: str, t: float):
    """Generate, pose, impose R. Returns (mesh, R) or None."""
    fam = _f_near_convex if want == "convex" else _NONCONVEX[int(rng.integers(len(_NONCONVEX)))]
    try:
        m = fam(rng, t)
    except Exception:
        return None
    # Floor of 4, not 50: the marching-cubes path already enforces >= 200 faces inside
    # `_mesh_from`, and a 50-face floor here silently rejected every low-poly convex hull --
    # including the cube, which is model 2. `hac26.calibrate.decimate` subdivides coarse
    # meshes up to the token budget later, so a 12-face body is fine downstream.
    if m is None or len(m.faces) < 4:
        return None
    m = to_challenge_pose(m)
    if m is None:
        return None
    R = sample_R(rng, None if want == "convex" else t)
    m = set_R(m, R)
    return (m, R) if m is not None else None


def sample_corpus(n: int, seed: int = 0, p_near_convex: float = 0.40,
                  max_attempts_per_body: int = 14, verbose: bool = False) -> list:
    """Return `n` bodies as (verts, faces, meta), stratified by hull deficit.

    Acceptance:
      * near-convex quota (p_near_convex of n): D_rms < NEAR_CONVEX_MAX.
      * the rest: D_lo >= D_LO_MIN and D_rms inside a bin whose quota is not yet full.
    A body that is non-convex but lands in a full bin is discarded rather than kept, which
    is the whole point -- without the quota the sampler piles up in the shallow bins where
    the families are easiest to hit.
    """
    rng = np.random.default_rng(seed)
    n_conv = int(round(p_near_convex * n))
    n_non = n - n_conv
    quota = [n_non // len(D_BINS)] * len(D_BINS)
    for i in range(n_non - sum(quota)):
        quota[i] += 1
    filled = [0] * len(D_BINS)
    got_conv = 0
    out, attempts, budget = [], 0, max_attempts_per_body * n

    while (got_conv < n_conv or sum(filled) < n_non) and attempts < budget:
        attempts += 1
        need_conv = got_conv < n_conv
        want = "convex" if (need_conv and (sum(filled) >= n_non or rng.random() < 0.4)) \
            else "nonconvex"
        # aim at the emptiest bin, so late attempts target the deep tail rather than
        # re-rolling the easy shallow one
        if want == "nonconvex":
            short = [q - f for q, f in zip(quota, filled)]
            b = int(np.argmax(short))
            t = float(np.clip((b + rng.random()) / len(D_BINS), 0.02, 1.0))
        else:
            t = float(rng.random() * 0.3)
        drawn = _draw_one(rng, want, t)
        if drawn is None:
            continue
        m, R = drawn
        try:
            d_rms, d_lo = hull_deficit(m)
        except Exception:
            continue
        if not np.isfinite(d_rms):
            continue
        if want == "convex":
            if d_rms >= NEAR_CONVEX_MAX or got_conv >= n_conv:
                continue
            got_conv += 1
            kind = "near_convex"
        else:
            if d_lo < D_LO_MIN:
                continue
            b = next((i for i, (lo, hi) in enumerate(D_BINS) if lo <= d_rms < hi), None)
            if b is None or filled[b] >= quota[b]:
                continue
            filled[b] += 1
            kind = f"nonconvex_bin{b}"
        out.append((np.asarray(m.vertices, float), np.asarray(m.faces, np.int64),
                    {"kind": kind, "R": R, "D_rms": d_rms, "D_lo": d_lo}))
        if verbose:
            print(f"  [{len(out):>3}/{n}] {kind:<16} R {R:5.2f}  "
                  f"D_rms {d_rms:.3f}  D_lo {d_lo:.2f}", flush=True)

    if len(out) < n and verbose:
        print(f"  WARNING: {len(out)}/{n} after {attempts} attempts "
              f"(convex {got_conv}/{n_conv}, bins {filled} of {quota})", flush=True)
    rng.shuffle(out)
    return out


def report(bodies: list) -> str:
    """One-line-per-stratum summary, for checking a corpus before spending a training run."""
    from collections import defaultdict
    g = defaultdict(list)
    for _, _, meta in bodies:
        g[meta["kind"]].append(meta)
    lines = [f"{'stratum':<18}{'n':>4}{'D_rms':>18}{'D_lo':>14}{'R':>16}"]
    for k in sorted(g):
        d = np.array([m["D_rms"] for m in g[k]])
        l = np.array([m["D_lo"] for m in g[k]])
        r = np.array([m["R"] for m in g[k]])
        lines.append(f"{k:<18}{len(d):>4}{d.min():>8.3f}-{d.max():<9.3f}"
                     f"{l.mean():>10.2f}    {r.min():>6.2f}-{r.max():<8.2f}")
    lines.append("  reference: model 1 D_rms 0.003 / model 2 0.000 / model 3 0.201 (D_lo 0.91)")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="sample and report a corpus")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-convex", type=float, default=0.40)
    a = ap.parse_args()
    bodies = sample_corpus(a.n, seed=a.seed, p_near_convex=a.p_convex, verbose=True)
    print()
    print(report(bodies))

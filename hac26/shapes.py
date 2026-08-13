"""Synthetic shape generation, EGI extraction, and a brute-force convex curve renderer.

The brute-force renderer evaluates the facet-sum model directly on a mesh (valid for
convex bodies, where visible-and-lit <=> mu>0 and mu0>0)
and is used to cross-validate the matrix route A @ g in tests.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.special import gammaln, lpmv

from .geometry import OMEGA0, NormalGrid, body_frame_dirs, cell_index, project_closure, psi_grid
from forward_models.convex_egi import kernel


# ---------- icosphere ---------------------------------------------------------------
def icosphere(subdiv: int = 3) -> tuple:
    """Unit icosphere (verts, faces). subdiv=3 -> 642 verts, 1280 faces."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                      [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                      [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(subdiv):
        edge_mid: dict = {}
        vlist = list(verts)

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in edge_mid:
                p = vlist[i] + vlist[j]
                p = p / np.linalg.norm(p)
                edge_mid[key] = len(vlist)
                vlist.append(p)
            return edge_mid[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.array(vlist)
        faces = np.array(new_faces)
    return verts, faces


# ---------- real spherical harmonics -------------------------------------------------
def real_sh_basis(L: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Rows = real orthonormal SH Y for l=1..L (l=0 excluded; it is a pure log-scale
    shift and scale is unidentifiable from normalized curves). Shape (n_coef, npts)."""
    x = np.cos(theta)
    rows = []
    for l in range(1, L + 1):
        nl0 = np.sqrt((2 * l + 1) / (4 * np.pi))
        rows.append(nl0 * lpmv(0, l, x))
        for m in range(1, l + 1):
            nlm = np.sqrt((2 * l + 1) / (4 * np.pi)) * np.exp(
                0.5 * (gammaln(l - m + 1) - gammaln(l + m + 1)))
            plm = lpmv(m, l, x)
            rows.append(np.sqrt(2.0) * nlm * plm * np.cos(m * phi))
            rows.append(np.sqrt(2.0) * nlm * plm * np.sin(m * phi))
    return np.stack(rows, axis=0)


def sh_lognormal_mesh(rng: np.random.Generator, L: int = 6, amp: float = 0.35,
                      decay: float = 1.5, subdiv: int = 3) -> tuple:
    """Star-shaped body r(u) = exp(sum a_lm Y_lm(u)), a_lm ~ N(0, (amp/(1+l)^decay)^2)."""
    u, faces = icosphere(subdiv)
    theta = np.arccos(np.clip(u[:, 2], -1, 1))
    phi = np.mod(np.arctan2(u[:, 1], u[:, 0]), 2 * np.pi)
    B = real_sh_basis(L, theta, phi)
    ls = np.concatenate([[l] * (2 * l + 1) for l in range(1, L + 1)])
    a = rng.normal(0.0, amp / (1.0 + ls) ** decay)
    r = np.exp(a @ B)
    return u * r[:, None], faces, (a, L)


def random_convex_polytope(rng: np.random.Generator, n_pts: int = 40) -> tuple:
    pts = rng.normal(size=(n_pts, 3)) * rng.uniform(0.5, 1.5, size=3)
    hull = ConvexHull(pts)
    return pts[hull.vertices], None


def ellipsoid_mesh(rng: np.random.Generator, subdiv: int = 3) -> tuple:
    u, faces = icosphere(subdiv)
    axes = rng.uniform(0.5, 1.5, size=3)
    return u * axes, faces, axes


# ---------- convex hull with outward-oriented faces ----------------------------------
def hull_mesh(points: np.ndarray) -> tuple:
    hull = ConvexHull(points)
    verts = points
    faces = hull.simplices.copy()
    eqs = hull.equations[:, :3]
    for i, f in enumerate(faces):
        n = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
        if n @ eqs[i] < 0:
            faces[i] = f[::-1]
    return verts, faces


def face_normals_areas(verts: np.ndarray, faces: np.ndarray) -> tuple:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    a2 = np.linalg.norm(cr, axis=1)
    keep = a2 > 1e-14
    n = np.zeros_like(cr)
    n[keep] = cr[keep] / a2[keep, None]
    return n, 0.5 * a2


def mesh_to_egi(verts: np.ndarray, faces: np.ndarray, grid: NormalGrid,
                close: bool = True) -> np.ndarray:
    """Bin facet areas by facet normal into the grid cells (discretized S_K).
    For a closed oriented mesh sum_f area_f n_f = 0 exactly; binning to cell-center
    normals perturbs this, so optionally re-project onto the closure cone."""
    n, a = face_normals_areas(verts, faces)
    idx = cell_index(grid, n)
    g = np.bincount(idx, weights=a, minlength=grid.n).astype(float)
    return project_closure(g, grid.normals) if close else g


def rescale_touch_z(verts: np.ndarray) -> np.ndarray:
    """Uniform scale + translation so that min z = -1, max z = +1 (challenge pose),
    xy-centroid at the rotation axis. Photometrically this only changes overall scale,
    which the normalization cancels (scale-invariance lemma)."""
    v = verts.copy()
    v[:, 0] -= v[:, 0].mean()
    v[:, 1] -= v[:, 1].mean()
    zmin, zmax = v[:, 2].min(), v[:, 2].max()
    v[:, 2] -= 0.5 * (zmin + zmax)
    return v * (2.0 / (zmax - zmin))


def canonicalize_r(verts: np.ndarray) -> np.ndarray:
    """Scale xy so the max axis distance is 1, leaving z (already in [-1,1]) alone.

    The challenge publishes a bounding radius R per model, and per-curve mean
    normalization provably destroys the cross-camera amplitudes that encode the
    body's latitude profile -- i.e. its aspect ratio. Rather than ask the network to
    predict a coordinate the data cannot determine and then anisotropically rescale
    its answer at test time (a train/test mismatch), train on the CANONICAL shape
    with r_max = 1 and restore the width from R at reconstruction:

        train target : canonicalize_r(hull)              (r_max = 1)
        test  output : fit_to_cylinder(prediction, R)    (r_max = R)

    The two are exact inverses, so the network never spends capacity on the
    unidentifiable degree of freedom and nothing is applied at test time that was
    absent at train time."""
    v = verts.copy()
    r = float(np.sqrt((v[:, :2] ** 2).sum(1)).max())
    if r > 1e-12:
        v[:, :2] /= r
    return v


def mesh_support(verts: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Support function h(u) = max_{x in conv(verts)} <x, u>, sampled on `normals`.

    Exact for the convex hull of `verts` (the max of a linear functional over a
    polytope is attained at a vertex). Unlike the EGI this needs no closure
    condition and no positivity repair: h determines the body directly as
    {x : <x,u> <= h(u) for all u}."""
    return (verts @ normals.T).max(axis=0)


def star_inside(points: np.ndarray, a_L: tuple) -> np.ndarray:
    """Exact inside test for the SH star-shaped body: |x| <= r(x/|x|)."""
    a, L = a_L
    r = np.linalg.norm(points, axis=-1)
    safe = np.where(r > 1e-12, r, 1.0)
    u = points / safe[..., None]
    theta = np.arccos(np.clip(u[..., 2], -1, 1))
    phi = np.mod(np.arctan2(u[..., 1], u[..., 0]), 2 * np.pi)
    B = real_sh_basis(L, theta.ravel(), phi.ravel())
    rad = np.exp(a @ B).reshape(r.shape)
    return r <= rad


# ---------- brute-force convex renderer ----------------------------------------------
def mesh_curves_convex(verts: np.ndarray, faces: np.ndarray, cameras: list, m: int,
                       curve_types: list, c_lambert: float = 0.1, sigma: float = 1.0,
                       delta: float = 1.0, psi0: float = 0.0,
                       ls_weight: float = 1.0) -> np.ndarray:
    """Raw (unnormalized) curves of a CONVEX mesh, shape (n_curves, m)."""
    n, a = face_normals_areas(verts, faces)
    psi = psi_grid(m, sigma=sigma, psi0=psi0)
    v0 = body_frame_dirs(OMEGA0, psi)
    mu0 = n @ v0.T
    out = []
    for cam, ctype in zip(cameras, curve_types):
        v = body_frame_dirs(cam.omega(delta=delta), psi)
        mu = n @ v.T
        out.append(a @ kernel(mu, mu0, ctype, c_lambert, ls_weight=ls_weight))
    return np.stack(out, axis=0)


# ---------- training-shape sampler ----------------------------------------------------
def sample_training_shape(rng: np.random.Generator, grid: NormalGrid,
                          p_flat: float = 0.0) -> dict:
    """Random body, challenge-posed; returns EGI of its convex hull + metadata.

    `p_flat` mixes in flat-faced / few-face bodies (see sample_flat_shape). The default
    of 0 reproduces the original distribution exactly, so old checkpoints stay
    comparable; the trained presets set it explicitly.
    """
    meta = None
    if p_flat and rng.random() < p_flat:
        v, kind = sample_flat_shape(rng)
        v = rescale_touch_z(v)
        hv, hf = hull_mesh(v)
        g = mesh_to_egi(hv, hf, grid)
        return {"kind": kind, "verts": hv, "faces": hf, "g": g,
                "p": g / max(g.sum(), 1e-12), "meta": None}
    kind = rng.choice(["sh", "poly", "ellipsoid"], p=[0.6, 0.3, 0.1])
    if kind == "sh":
        v, f, meta = sh_lognormal_mesh(rng, L=int(rng.integers(4, 9)),
                                       amp=rng.uniform(0.2, 0.5))
    elif kind == "poly":
        v, _ = random_convex_polytope(rng, n_pts=int(rng.integers(12, 60)))
    else:
        v, f, meta = ellipsoid_mesh(rng)
    v = rescale_touch_z(v)
    hv, hf = hull_mesh(v)
    g = mesh_to_egi(hv, hf, grid)
    return {"kind": kind, "verts": hv, "faces": hf, "g": g,
            "p": g / max(g.sum(), 1e-12), "meta": meta}


def sample_damit_shape(rng: np.random.Generator, grid: NormalGrid,
                       pool: list) -> dict:
    """Draw a random real DAMIT mesh, convex-hull it, challenge-pose it, return its EGI.

    Same output contract as sample_training_shape. A random 3D rotation is applied before
    posing so the (arbitrary) DAMIT body frame is not memorized as the pole axis."""
    verts, faces = pool[int(rng.integers(len(pool)))]
    v = verts @ _random_rotation(rng).T
    v = rescale_touch_z(v)
    hv, hf = hull_mesh(v)
    g = mesh_to_egi(hv, hf, grid)
    return {"kind": "damit", "verts": hv, "faces": hf, "g": g,
            "p": g / max(g.sum(), 1e-12), "meta": None}


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix (QR of a Gaussian, sign-fixed)."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.diag(r))


# ---------- flat-faced and few-face bodies -------------------------------------------
# The original training family (sh / random-hull / ellipsoid) contains essentially no
# bodies with large flat facets: a hull of 12-60 Gaussian points is round, and the SH and
# ellipsoid families are smooth by construction, while challenge model 2 is a cube, whose
# EGI mass concentrates in 6 cells. These generators add flat-faced and few-faced bodies.

_PLATONIC = {}


def platonic(kind: str) -> np.ndarray:
    """Vertices of a platonic solid, unit-ish scale (cached)."""
    if kind in _PLATONIC:
        return _PLATONIC[kind]
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    if kind == "tetra":
        v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    elif kind == "cube":
        v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    elif kind == "octa":
        v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    elif kind == "dodeca":
        v = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        v += [[0, s * 1 / phi, t * phi] for s in (-1, 1) for t in (-1, 1)]
        v += [[s * 1 / phi, t * phi, 0] for s in (-1, 1) for t in (-1, 1)]
        v += [[s * phi, 0, t * 1 / phi] for s in (-1, 1) for t in (-1, 1)]
        v = np.array(v, float)
    elif kind == "icosa":
        v = [[0, s, t * phi] for s in (-1, 1) for t in (-1, 1)]
        v += [[s, t * phi, 0] for s in (-1, 1) for t in (-1, 1)]
        v += [[s * phi, 0, t] for s in (-1, 1) for t in (-1, 1)]
        v = np.array(v, float)
    else:
        raise ValueError(kind)
    _PLATONIC[kind] = v / np.linalg.norm(v, axis=1).max()
    return _PLATONIC[kind]


def prism_mesh(rng: np.random.Generator) -> np.ndarray:
    """Right prism on a regular or jittered n-gon: the archetypal few-face body."""
    n = int(rng.integers(3, 9))
    a = np.arange(n) * 2 * np.pi / n + rng.uniform(0, 2 * np.pi)
    rad = 1.0 + rng.normal(0, 0.08, size=n) if rng.random() < 0.5 else np.ones(n)
    ring = np.stack([rad * np.cos(a), rad * np.sin(a)], axis=1)
    ring = ring * rng.uniform(0.6, 1.4, size=2)          # elliptical cross-section
    hz = rng.uniform(0.4, 1.8)
    top = np.hstack([ring * rng.uniform(0.55, 1.0), np.full((n, 1), hz)])   # allow taper
    bot = np.hstack([ring, np.full((n, 1), -hz)])
    return np.vstack([top, bot])


def faceted_mesh(rng: np.random.Generator) -> np.ndarray:
    """A smooth body sliced by a few random half-spaces -> genuinely flat facets.

    This is the physically motivated one: real small bodies acquire flat faces from
    large impacts and from fracture along planes, so a cut ellipsoid is a much better
    model of a faceted asteroid than either a platonic solid or a Gaussian hull.
    """
    v, _, _ = sh_lognormal_mesh(rng, L=int(rng.integers(3, 7)), amp=rng.uniform(0.1, 0.35))
    v = v * rng.uniform(0.6, 1.4, size=3)
    for _ in range(int(rng.integers(1, 6))):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        off = np.quantile(v @ u, rng.uniform(0.55, 0.95))
        proj = v @ u
        cut = proj > off
        if cut.any():
            v[cut] -= np.outer(proj[cut] - off, u)       # project onto the cutting plane
    return v


def bilobe_mesh(rng: np.random.Generator) -> np.ndarray:
    """Hull of two overlapping ellipsoids -- a contact-binary silhouette."""
    u, _ = icosphere(2)
    out = []
    sep = rng.uniform(0.4, 1.1)
    for k in (-1, 1):
        ax = rng.uniform(0.45, 1.0, size=3)
        c = np.zeros(3)
        c[0] = k * sep
        out.append(u * ax + c)
    return np.vstack(out)


def sample_flat_shape(rng: np.random.Generator) -> tuple:
    """Draw one flat-faced / few-face body. Returns (points, kind)."""
    kind = rng.choice(["platonic", "prism", "faceted", "bilobe"], p=[0.2, 0.3, 0.35, 0.15])
    if kind == "platonic":
        name = rng.choice(["tetra", "cube", "octa", "dodeca", "icosa"])
        v = platonic(name) * rng.uniform(0.6, 1.5, size=3)     # anisotropic: not just the solid
        kind = f"platonic_{name}"
    elif kind == "prism":
        v = prism_mesh(rng)
    elif kind == "faceted":
        v = faceted_mesh(rng)
    else:
        v = bilobe_mesh(rng)
    return v @ _random_rotation(rng).T, kind

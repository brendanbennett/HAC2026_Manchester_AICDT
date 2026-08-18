"""How varied a shape library is, and whether each body is admissible.

Two questions, kept apart.

VALIDITY is per body and binary: closed, one component, strictly non-convex, correctly posed.
`check_body` returns the individual answers rather than a single bool so a failure says which
constraint broke.

DIVERSITY is a property of the set. The headline number is the participation ratio

    PR = (sum_i lambda_i)^2 / sum_i lambda_i^2

of the eigenvalues of the library's covariance -- the number of directions that actually carry
variance, equal to D for an isotropic cloud and to 1 for a cloud on a line. PR is the right
statistic here because the shape prior Gamma in `map_gauss_newton` IS that covariance: a
library with PR = 4 gives a prior that is nearly a four-parameter family, and every direction
outside it is pinned to the library mean no matter what the data says.

A caution that decides how the numbers below are read: PR is not invariant to what the code
measures. A library whose bodies differ mostly in overall SIZE scores its top eigenvalue on a
degree of freedom the challenge normalisation removes anyway, so PR on unposed bodies flatters
a library that is not actually varied in shape. Everything here is therefore measured on
posed bodies, and `descriptor_support` is normalised by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .shape_library import (_parity_occupancy, convexity_ratio, decimate_mesh,
                            is_edge_manifold, n_components)

__all__ = ["participation_ratio", "spectrum", "descriptor_support", "descriptor_concavity",
           "library_descriptors", "occupancy", "principal_frame", "dice", "pairwise_dice",
           "check_body", "check_library", "design_normals"]


def design_normals(n: int = 64) -> np.ndarray:
    """The spherical design the solver's convex core is defined on."""
    p = Path(__file__).with_name(f"design{n}.npy")
    if p.exists():
        return np.load(p)
    from .field import spherical_design            # needs torch; only if the cache is absent
    return spherical_design(n)


# --------------------------------------------------------------------------- diversity

def spectrum(X: np.ndarray) -> np.ndarray:
    """Eigenvalues of the sample covariance of rows of X, descending, clipped at 0."""
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    lam = np.linalg.eigvalsh(np.cov(Xc, rowvar=False))
    return np.clip(lam, 0.0, None)[::-1]


def participation_ratio(X: np.ndarray) -> float:
    """Effective number of dimensions spanned by the rows of X."""
    lam = spectrum(X)
    s2 = float((lam ** 2).sum())
    return float(lam.sum() ** 2 / s2) if s2 > 0 else 0.0


def descriptor_support(verts: np.ndarray, normals: np.ndarray | None = None) -> np.ndarray:
    """h(n) = max_v <v, n> on the design normals.

    This is not a proxy: `scripts/fit_shapes.py` computes exactly this vector as `h0s` and
    loads it into `ConvexCore.set_support`, so it IS the convex half of the fitted code,
    obtainable without running the autodecoder.
    """
    n = design_normals() if normals is None else np.asarray(normals, float)
    return (np.asarray(verts, float) @ n.T).max(axis=0)


def descriptor_concavity(verts: np.ndarray, faces: np.ndarray,
                         probes: np.ndarray, normals: np.ndarray | None = None,
                         res: int = 64, extent: float = 1.35) -> np.ndarray:
    """f_body(y) - f_core(y) at fixed probe points: what the token field has to carry.

    `TokenField` is fitted to exactly this difference -- the convex core explains f_core and
    Delta makes up the rest -- so the spread of this vector across a library bounds how many
    dimensions the token code can possibly need. Computed from the body's own occupancy by a
    distance transform, so it needs neither torch nor trimesh.
    """
    n = design_normals() if normals is None else np.asarray(normals, float)
    h = descriptor_support(verts, n)
    occ = occupancy(verts, faces, res, extent)
    sp = 2.0 * extent / (res - 1)
    sdf = (ndimage.distance_transform_edt(~occ, sampling=sp)
           - ndimage.distance_transform_edt(occ, sampling=sp))
    idx = ((np.asarray(probes, float) + extent) / sp).T
    f_body = ndimage.map_coordinates(sdf, idx, order=1, mode="nearest")
    f_core = (np.asarray(probes, float) @ n.T - h).max(axis=1)
    return f_body - f_core


def _probe_points(n: int = 512, seed: int = 0, radius: float = 1.15) -> np.ndarray:
    """A fixed cloud filling the posed body's bounding cylinder. Fixed across the library:
    the descriptor is only comparable between bodies if the probes are the same points."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([r * np.cos(t), r * np.sin(t), rng.uniform(-1.0, 1.0, n)], axis=1)


def library_descriptors(bodies: list, n_probes: int = 512, res: int = 64) -> dict:
    """Support, concavity and combined descriptors for a library of `Body`."""
    nrm = design_normals()
    probes = _probe_points(n_probes)
    H = np.stack([descriptor_support(b.verts, nrm) for b in bodies])
    C = np.stack([descriptor_concavity(b.verts, b.faces, probes, nrm, res=res)
                  for b in bodies])
    # scale the two blocks to equal mean variance before concatenating, so the combined PR
    # is not simply whichever block happens to carry larger numbers
    hs = np.sqrt(max(H.var(0).mean(), 1e-18))
    cs = np.sqrt(max(C.var(0).mean(), 1e-18))
    return {"support": H, "concavity": C, "combined": np.hstack([H / hs, C / cs])}


# --------------------------------------------------------------------------- overlap

def occupancy(verts: np.ndarray, faces: np.ndarray, res: int = 64,
              extent: float = 1.35, decimate: bool = True) -> np.ndarray:
    v, f = np.asarray(verts, float), np.asarray(faces, np.int64)
    if decimate and len(f) > 3000:
        v, f = decimate_mesh(v, f, extent, res)
    return _parity_occupancy(v, f, extent, res)


def principal_frame(verts: np.ndarray, faces: np.ndarray, res: int = 64,
                    extent: float = 1.35) -> np.ndarray:
    """Rotation taking the body's principal axes onto the coordinate axes.

    The inertia tensor is computed from the OCCUPANCY, not from the vertices: vertex moments
    weight a densely tessellated region more heavily than a sparsely tessellated one of the
    same volume, which makes the frame depend on the meshing rather than on the body.

    Axes are ordered by eigenvalue. The four proper sign flips are left unresolved here and
    maximised over in `dice`, because a body with two near-equal moments has no stable sign
    and picking one by a skewness rule silently reports a low Dice for two bodies that are
    the same shape.
    """
    occ = occupancy(verts, faces, res, extent)
    idx = np.argwhere(occ).astype(float)
    if len(idx) < 4:
        return np.eye(3)
    sp = 2.0 * extent / (res - 1)
    p = idx * sp - extent
    p -= p.mean(0)
    _, vecs = np.linalg.eigh(np.cov(p, rowvar=False))
    R = vecs[:, ::-1].T                       # rows = principal axes, largest first
    if np.linalg.det(R) < 0:
        R[2] *= -1.0
    return R


_SIGN_FLIPS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)


def occupancies_aligned(bodies: list, res: int = 48, extent: float = 1.35) -> list:
    """Occupancy grids for all four proper sign flips of each body's principal frame.

    Computed once per body rather than once per PAIR: `pairwise_dice` compares O(n^2) pairs
    but each body's own alignment and rasterisation is reusable across all of them, so this
    turns an O(pairs) cost into an O(n_bodies) one.
    """
    out = []
    for b in bodies:
        R = principal_frame(b.verts, b.faces, res, extent)
        va = b.verts @ R.T
        out.append([occupancy(va * s, b.faces, res, extent) for s in _SIGN_FLIPS])
    return out


def _dice_from_occ(occs_a: list, occs_b: list) -> float:
    A = occs_a[0]
    na = A.sum()
    best = 0.0
    for B in occs_b:
        nb = B.sum()
        if na + nb == 0:
            continue
        best = max(best, 2.0 * float((A & B).sum()) / float(na + nb))
    return best


def dice(a, b, res: int = 64, extent: float = 1.35, align: bool = True) -> float:
    """Voxel Dice between two bodies, after aligning both on their principal axes.

    Intersection is not rotation-invariant, so comparing two library bodies in their stored
    poses measures how they happen to be oriented as much as how they are shaped. Both are
    carried into their own principal frame first; the residual four-fold sign ambiguity is
    resolved by taking the best of the four proper flips.

    For many pairs from the same library, `pairwise_dice` is the right entry point: it
    shares each body's alignment and rasterisation across every pair instead of repeating
    them, which is what this function does independently for one pair.
    """
    va, fa = (a.verts, a.faces) if hasattr(a, "verts") else a
    vb, fb = (b.verts, b.faces) if hasattr(b, "verts") else b
    if not align:
        A = occupancy(va, fa, res, extent)
        B = occupancy(vb, fb, res, extent)
        na, nb = A.sum(), B.sum()
        return 2.0 * float((A & B).sum()) / float(na + nb) if na + nb else 0.0
    Ra = principal_frame(va, fa, res, extent)
    Rb = principal_frame(vb, fb, res, extent)
    occs_a = [occupancy((va @ Ra.T) * _SIGN_FLIPS[0], fa, res, extent)]
    occs_b = [occupancy((vb @ Rb.T) * s, fb, res, extent) for s in _SIGN_FLIPS]
    return _dice_from_occ(occs_a, occs_b)


def pairwise_dice(bodies: list, res: int = 48, extent: float = 1.35,
                  max_pairs: int | None = 300, seed: int = 0) -> np.ndarray:
    """Dice over a random sample of distinct pairs. Aligned, so this measures shape spread.

    A library is varied when this distribution is broad and centred well below 1; a library
    of near-copies concentrates near 1 whatever its codes look like.
    """
    n = len(bodies)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
    occs = occupancies_aligned(bodies, res, extent)
    return np.array([_dice_from_occ(occs[i], occs[j]) for i, j in pairs])


# --------------------------------------------------------------------------- validity

def check_body(body, radius: float = 1.0, convexity_max: float = 0.95,
               pose_tol: float = 1e-6, radius_tol: float = 1e-6) -> dict:
    """Every per-body constraint, reported separately."""
    v, f = body.verts, body.faces
    z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
    rmax = float(np.hypot(v[:, 0], v[:, 1]).max())
    cx, cy = float(v[:, 0].mean()), float(v[:, 1].mean())
    c = convexity_ratio(v, f)
    return {
        "non_convex": bool(c < convexity_max), "convexity": float(c),
        "closed": bool(is_edge_manifold(f)),
        "single_component": bool(n_components(v, f) == 1),
        "z_span": bool(abs(z0 + 1.0) <= pose_tol and abs(z1 - 1.0) <= pose_tol),
        "inside_cylinder": bool(rmax <= radius * (1.0 + radius_tol)),
        "on_axis": bool(max(abs(cx), abs(cy)) <= 0.25 * radius),
        "z_range": (z0, z1), "r_max": rmax, "xy_centroid": (cx, cy),
    }


def check_library(bodies: list, **kw) -> dict:
    """`check_body` over a library, plus the indices that failed each constraint."""
    rows = [check_body(b, **kw) for b in bodies]
    keys = ["non_convex", "closed", "single_component", "z_span", "inside_cylinder", "on_axis"]
    return {"n": len(bodies),
            "pass": {k: int(sum(r[k] for r in rows)) for k in keys},
            "failed": {k: [i for i, r in enumerate(rows) if not r[k]] for k in keys},
            "convexity": np.array([r["convexity"] for r in rows]),
            "rows": rows}

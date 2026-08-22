"""Non-convex training bodies, generated as level sets.

Why level sets rather than mesh booleans. Every constraint the library has to satisfy is a
statement about a SET, not about a triangulation: one connected component, no interior void,
closed surface, a chosen non-convexity. On an occupancy grid all four are decidable and
repairable with `scipy.ndimage` before a single triangle exists, and marching cubes then
returns a closed oriented manifold by construction. The alternative -- concatenating or
booleaning meshes -- is what the previous corpus did, and it produced bodies that are two
interpenetrating closed surfaces (see `docs/shape_library.md`); a signed distance sampled
against such a mesh is not a signed distance, so the codes fitted to it were fitted to noise.

Bodies are built compositionally as fields f(x) with f < 0 inside. Unions are min, cuts are
max(f, -g). These are not metric distances away from the zero set, but they are exactly
signed, and only the sign and the location of the zero crossing matter here.

Non-convexity is a GATE, not a hope. `sample_body` measures volume / hull volume on the
finished mesh and redraws with intensified modifiers until it passes, so "strictly
non-convex" is a postcondition of the sampler rather than a property the recipes are
believed to have.

The two external sources enter through `body_from_mesh` (Thingi10K: voxelise, repair,
re-pose) and `body_from_convex_points` (DAMIT: a convex hull used only as a starting field,
with modifiers applied and the same gate enforced afterwards, so a DAMIT body can never
reach the library still convex).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as _dcfield
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

from .shapes import icosphere, real_sh_basis

__all__ = [
    "Body", "LibrarySpec", "sample_body", "build_library",
    "body_from_mesh", "body_from_convex_points",
    "mesh_volume", "hull_volume", "convexity_ratio", "is_edge_manifold",
    "n_components", "pose", "extract", "voxelise",
    "sd_sphere", "sd_ellipsoid", "sd_box", "sd_cylinder", "sd_torus",
    "sd_convex", "sd_star_sh", "op_union", "op_subtract", "op_intersect",
    "op_smooth_union", "op_displace", "decimate_mesh", "min_feature_radius",
]

Field = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- primitives
# Each returns f(p) for p of shape (..., 3), negative inside.

def _rot(axis_z: np.ndarray) -> np.ndarray:
    """Orthonormal frame whose third row is `axis_z`; rows map world -> local."""
    w = np.asarray(axis_z, float)
    w = w / max(np.linalg.norm(w), 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, w); u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return np.stack([u, v, w], axis=0)


def sd_sphere(centre=(0, 0, 0), radius: float = 1.0) -> Field:
    c = np.asarray(centre, float)
    return lambda p: np.linalg.norm(p - c, axis=-1) - radius


def sd_ellipsoid(centre=(0, 0, 0), axes=(1, 1, 1), rot: np.ndarray | None = None) -> Field:
    """Inigo Quilez's ellipsoid bound: exactly signed, and first-order correct in distance."""
    c = np.asarray(centre, float)
    a = np.asarray(axes, float)
    R = np.eye(3) if rot is None else np.asarray(rot, float)

    def f(p):
        q = (p - c) @ R.T
        k0 = np.linalg.norm(q / a, axis=-1)
        k1 = np.linalg.norm(q / (a * a), axis=-1)
        return np.where(k1 > 1e-12, k0 * (k0 - 1.0) / np.maximum(k1, 1e-12), k0 - 1.0)
    return f


def sd_box(centre=(0, 0, 0), half=(1, 1, 1), rot: np.ndarray | None = None,
           round_r: float = 0.0) -> Field:
    c = np.asarray(centre, float)
    h = np.asarray(half, float) - round_r
    R = np.eye(3) if rot is None else np.asarray(rot, float)

    def f(p):
        q = np.abs((p - c) @ R.T) - h
        out = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        return out + np.minimum(q.max(axis=-1), 0.0) - round_r
    return f


def sd_cylinder(centre=(0, 0, 0), axis=(0, 0, 1), radius: float = 1.0,
                half_height: float = 1.0, round_r: float = 0.0) -> Field:
    c = np.asarray(centre, float)
    R = _rot(axis)

    def f(p):
        q = (p - c) @ R.T
        d_r = np.linalg.norm(q[..., :2], axis=-1) - (radius - round_r)
        d_z = np.abs(q[..., 2]) - (half_height - round_r)
        d = np.stack([d_r, d_z], axis=-1)
        return (np.linalg.norm(np.maximum(d, 0.0), axis=-1)
                + np.minimum(d.max(axis=-1), 0.0) - round_r)
    return f


def sd_torus(centre=(0, 0, 0), axis=(0, 0, 1), major: float = 1.0,
             minor: float = 0.3) -> Field:
    c = np.asarray(centre, float)
    R = _rot(axis)

    def f(p):
        q = (p - c) @ R.T
        return np.hypot(np.linalg.norm(q[..., :2], axis=-1) - major, q[..., 2]) - minor
    return f


def sd_convex(normals: np.ndarray, offsets: np.ndarray) -> Field:
    """Intersection of half-spaces n_j . x <= d_j, as max_j (n_j . x - d_j).

    The same representation `ConvexCore` uses, so a convex basis (DAMIT, or a hull of
    sampled points) enters the generator in the solver's own parameterisation.
    """
    n = np.asarray(normals, float)
    d = np.asarray(offsets, float)
    return lambda p: (p @ n.T - d).max(axis=-1)


def sd_star_sh(coeffs: np.ndarray, l_max: int, centre=(0, 0, 0),
               scale: float = 1.0) -> Field:
    """|x - c| - scale * exp(sum a_lm Y_lm(u)): the star-shaped body of `hac26.shapes`."""
    c = np.asarray(centre, float)
    a = np.asarray(coeffs, float)

    def f(p):
        q = p - c
        r = np.linalg.norm(q, axis=-1)
        safe = np.where(r > 1e-12, r, 1.0)
        u = q / safe[..., None]
        theta = np.arccos(np.clip(u[..., 2], -1.0, 1.0))
        phi = np.mod(np.arctan2(u[..., 1], u[..., 0]), 2.0 * np.pi)
        B = real_sh_basis(l_max, theta.ravel(), phi.ravel())
        return r - scale * np.exp(a @ B).reshape(r.shape)
    return f


# --------------------------------------------------------------------------- operators

def op_union(*fs: Field) -> Field:
    return lambda p: np.min(np.stack([f(p) for f in fs], axis=0), axis=0)


def op_intersect(*fs: Field) -> Field:
    return lambda p: np.max(np.stack([f(p) for f in fs], axis=0), axis=0)


def op_subtract(a: Field, *bs: Field) -> Field:
    def f(p):
        out = a(p)
        for b in bs:
            out = np.maximum(out, -b(p))
        return out
    return f


def op_smooth_union(a: Field, b: Field, k: float = 0.1) -> Field:
    """Exponential smooth min: fills the crease at a neck instead of leaving a cusp.

    A contact binary made with a hard min has a tangent discontinuity at the joint; real
    necks are filleted, and a cusp there is a feature no physical body has.
    """
    def f(p):
        x, y = a(p), b(p)
        m = np.minimum(x, y)
        return m - k * np.log1p(np.exp(-np.abs(x - y) / max(k, 1e-9)))
    return f


def op_displace(a: Field, d: Callable[[np.ndarray], np.ndarray]) -> Field:
    return lambda p: a(p) + d(p)


def _sh_displacement(rng: np.random.Generator, l_max: int, amp: float) -> Callable:
    """Radial roughness: a band-limited random field on the sphere, added to f."""
    ls = np.concatenate([[l] * (2 * l + 1) for l in range(1, l_max + 1)])
    a = rng.normal(0.0, amp / (1.0 + ls) ** 1.1)

    def d(p):
        r = np.linalg.norm(p, axis=-1)
        safe = np.where(r > 1e-12, r, 1.0)
        u = p / safe[..., None]
        theta = np.arccos(np.clip(u[..., 2], -1.0, 1.0))
        phi = np.mod(np.arctan2(u[..., 1], u[..., 0]), 2.0 * np.pi)
        B = real_sh_basis(l_max, theta.ravel(), phi.ravel())
        return (a @ B).reshape(r.shape)
    return d


def _fibonacci_sphere(n: int) -> np.ndarray:
    """n directions spread quasi-uniformly over the unit sphere (the Fibonacci-lattice
    construction). Used as `pitted`'s candidate crater-centre lattice: n independent random
    directions cluster and leave visible gaps once n is large (the birthday-paradox effect
    on the sphere), which is exactly wrong for a modifier whose point is even coverage. A
    RANDOM PERMUTATION of a uniform lattice, instead of n independent draws, keeps the
    even spacing while still varying which lattice points get used from body to body.
    """
    i = np.arange(n)
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = 2.0 * np.pi * i / golden
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


# --------------------------------------------------------------------------- mesh measures

def mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume by the divergence theorem. Positive for outward-oriented faces."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def hull_volume(verts: np.ndarray) -> float:
    return float(ConvexHull(verts).volume)


def convexity_ratio(verts: np.ndarray, faces: np.ndarray) -> float:
    """volume / hull volume. 1 for a convex body; the gate (LibrarySpec.convexity_max)
    defaults to 0.98."""
    hv = hull_volume(verts)
    return abs(mesh_volume(verts, faces)) / hv if hv > 0 else 1.0


def is_edge_manifold(faces: np.ndarray) -> bool:
    """Every edge used exactly twice, with opposite orientation.

    This is the combinatorial half of `trimesh.is_watertight` and needs no trimesh, so the
    tests can run in an environment where trimesh is absent. `is_watertight` is checked
    against trimesh as well wherever it is importable.
    """
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    key = np.sort(e, axis=1)
    _, counts = np.unique(key, axis=0, return_counts=True)
    if not np.all(counts == 2):
        return False
    # orientation: each undirected edge must appear once in each direction
    fwd = {(int(a), int(b)) for a, b in e}
    return all((b, a) in fwd for a, b in fwd)


def n_components(verts: np.ndarray, faces: np.ndarray) -> int:
    """Connected components of the surface, by vertex adjacency across triangle edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    n = len(verts)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    used = np.unique(faces)
    lab = connected_components(g, directed=False)[1]
    return int(len(np.unique(lab[used])))


def decimate_mesh(verts: np.ndarray, faces: np.ndarray, extent: float, res: int) -> tuple:
    """Vertex-clustering decimation to roughly one triangle per grid cell of size `res`.

    Marching cubes at generation resolution (res=96 by default) routinely produces tens of
    thousands of faces; both the Dice/inertia metrics and the ray-cast curve renderer only
    need geometry resolved to their own, much coarser, grid, so snapping vertices to a cell
    a few times finer than that grid and dropping the triangles it degenerates cuts face
    count by 1-2 orders of magnitude with no visible effect on either. One representative
    vertex per cell (the first one seen) rather than a centroid: cheap, and the sub-cell
    displacement it introduces is below the resolution the caller reads the result at.

    Not manifoldness-preserving: collapsing vertices can merge two originally-distinct
    edges into one non-manifold edge. That is acceptable for both of this function's
    callers -- parity-test occupancy and the ray-cast curve renderer -- neither of which
    requires a manifold input, only a closed one. Do not decimate a mesh that has to stay
    edge-manifold afterwards (e.g. anything headed for `is_edge_manifold`/`n_components`
    validity checks); decimate a COPY for metrics and keep the original for those.
    """
    cell = 2.0 * extent / max(res, 1)
    key = np.round(verts / cell).astype(np.int64)
    _, first_idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    v2 = verts[first_idx]
    f2 = inv[faces]
    keep = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    if not keep.any():
        return verts, faces
    return v2, f2[keep]


# --------------------------------------------------------------------------- grid -> mesh

def voxelise(f: Field, extent: float = 1.6, res: int = 96,
             chunk: int = 400_000) -> tuple:
    """Sample f on a padded res^3 grid. Returns (values, spacing, origin).

    The grid is padded by one voxel of guaranteed-positive field on every side, so the
    zero set cannot touch the boundary and marching cubes cannot produce an open surface.
    """
    a = np.linspace(-extent, extent, res)
    g = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3)
    out = np.empty(len(g))
    for i in range(0, len(g), chunk):
        out[i:i + chunk] = f(g[i:i + chunk])
    vol = out.reshape(res, res, res)
    vol = np.pad(vol, 1, mode="constant", constant_values=float(np.abs(vol).max() + 1.0))
    spacing = 2.0 * extent / (res - 1)
    return vol, spacing, -extent - spacing


def _repair(vol: np.ndarray, eps: float) -> tuple:
    """Force one solid component and no interior void, by editing the field.

    Solid uses 6-connectivity, so a solid touching only at a corner counts as two pieces and
    is discarded rather than being welded into a pinch point that marching cubes would have to
    resolve. Background uses 6-connectivity TOO. That is deliberately not the complementary
    pair: the complementary pair is correct digital topology but marching cubes does not
    implement it, and the mismatch cost 71% of `pitted` attempts -- see the comment on the
    background label call below.

    Discarded pieces are removed by raising f above zero on exactly the voxels that belong to
    them. Editing everything outside a dilation of the kept piece instead is wrong and was
    the first version's bug: a discarded blob lying within the dilation is spared and comes
    back as a second surface component. Voids are removed by lowering f below zero inside
    them, which deletes the internal surface without creating a new crossing, since the
    void's neighbours are solid on every side.
    """
    solid = vol < 0.0
    if not solid.any():
        return vol, 0, 0
    lab, k = ndimage.label(solid, structure=ndimage.generate_binary_structure(3, 1))
    n_solid = k
    if k > 1:
        sizes = ndimage.sum(solid, lab, index=np.arange(1, k + 1))
        keep = lab == (int(np.argmax(sizes)) + 1)
    else:
        keep = solid
    vol = np.where(solid & ~keep, np.maximum(vol, eps), vol)

    bg = vol >= 0.0
    # 6-connectivity for the background, NOT 26. The complementary pair (6-solid,
    # 26-background) is the right one for digital topology, but marching cubes does not
    # implement that pairing: a background pocket joined to the outside only corner-wise is
    # judged "not a void" here and is then SEALED by the trilinear interpolant, which returns
    # a second, closed, interior surface component -- and `sample_body` rejects the body on
    # n_components. Measured on `pitted`, whose many small subtractions produce exactly this
    # pocket: 71% of attempts were discarded, every one of them for multi-component, against
    # 3.8% for craters and scallops. Five reproductions, including one with a single solid
    # component and no voids reported, all go to a single surface under 6-connectivity, at a
    # cost of 56-3015 extra voxels filled out of 96^3.
    lab_b, kb = ndimage.label(bg, structure=ndimage.generate_binary_structure(3, 1))
    outer = lab_b[0, 0, 0]
    void = bg & (lab_b != outer)
    n_void = int(kb - 1)
    if void.any():
        vol = np.where(void, np.minimum(vol, -eps), vol)
    return vol, n_solid, n_void


def extract(f: Field, extent: float = 1.6, res: int = 96) -> tuple:
    """Level set of f as a closed, single-component, void-free mesh.

    Returns (verts, faces, info).
    """
    from skimage import measure

    vol, spacing, origin = voxelise(f, extent, res)
    eps = 1e-3 * max(float(np.abs(vol).max()), 1e-9)
    vol, n_solid, n_void = _repair(vol, eps)
    if not (vol < 0).any():
        raise ValueError("empty body: the field is positive everywhere on the grid")
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0, spacing=(spacing,) * 3)
    verts = verts + origin
    # marching_cubes orients faces for `level` being the INSIDE-high convention; here the
    # inside is where f < 0, so the winding comes out inward. Flip once, then verify.
    faces = faces[:, ::-1].copy()
    if mesh_volume(verts, faces) < 0:
        faces = faces[:, ::-1].copy()
    return verts, faces, {"n_solid_components": n_solid, "n_voids_filled": n_void}


# --------------------------------------------------------------------------- posing

def pose(verts: np.ndarray, radius: float | None = 1.0,
         centre: str = "volume", faces: np.ndarray | None = None) -> np.ndarray:
    """Challenge pose: z spans exactly [-1, 1], the body is centred on the rotation axis,
    and its xy extent is brought to `radius`.

    `radius=None` leaves the xy scale alone and only centres, which is what the Thingi10K
    ingestion wants when the source aspect ratio is worth keeping.

    Scaling xy separately from z is `canonicalize_r`'s convention, not an accident: per-curve
    mean normalisation destroys the cross-camera amplitudes that carry the aspect ratio, so
    the library is built at the canonical r_max and the width is restored from the published
    R at reconstruction time.
    """
    v = np.asarray(verts, float).copy()
    if centre == "volume" and faces is not None:
        v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
        # centroid of the solid, by the divergence theorem on each coordinate
        cr = np.cross(v1 - v0, v2 - v0)
        tet = (v0 + v1 + v2) / 4.0
        w = np.einsum("ij,ij->i", v0 + v1 + v2, cr) / 18.0
        tot = w.sum()
        c = (tet * w[:, None]).sum(0) / tot if abs(tot) > 1e-12 else v.mean(0)
    else:
        c = v.mean(0)
    v[:, 0] -= c[0]
    v[:, 1] -= c[1]
    zmin, zmax = v[:, 2].min(), v[:, 2].max()
    if zmax - zmin < 1e-12:
        raise ValueError("degenerate body: zero z extent")
    v[:, 2] = 2.0 * (v[:, 2] - zmin) / (zmax - zmin) - 1.0
    if radius is not None:
        r = float(np.hypot(v[:, 0], v[:, 1]).max())
        if r > 1e-12:
            v[:, :2] *= radius / r
    return v


# --------------------------------------------------------------------------- the recipes

@dataclass
class LibrarySpec:
    """Everything the sampler is allowed to vary. Exposed so a run is reproducible from it."""
    res: int = 96
    extent: float = 1.6
    radius: float = 1.0
    convexity_max: float = 0.98          # volume / hull volume must fall below this
    base_weights: dict = _dcfield(default_factory=lambda: {
        "star_sh": 0.17, "ellipsoid": 0.08, "polytope": 0.10, "lobes": 0.15,
        "prism": 0.10, "rubble": 0.12, "arch": 0.05, "slab": 0.07,
        "contact_binary": 0.16})
    n_modifiers: tuple = (2, 6)          # inclusive range, drawn per body
    mod_weights: dict = _dcfield(default_factory=lambda: {
        "craters": 0.18, "pitted": 0.10, "basin": 0.12, "cuts": 0.14, "groove": 0.10,
        "waist": 0.10, "bite": 0.12, "scallops": 0.08, "boulders": 0.06,
        "roughness": 0.06, "tunnel": 0.02})
    max_attempts: int = 12


@dataclass
class Body:
    verts: np.ndarray
    faces: np.ndarray
    recipe: dict
    info: dict

    @property
    def convexity(self) -> float:
        return self.info["convexity"]


def _rand_rot(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.diag(r))


def _base(rng: np.random.Generator, kind: str, s: float) -> tuple:
    """A starting field of overall size ~s, plus the parameters that made it."""
    R = _rand_rot(rng)
    if kind == "star_sh":
        L = int(rng.integers(3, 9))
        ls = np.concatenate([[l] * (2 * l + 1) for l in range(1, L + 1)])
        a = rng.normal(0.0, rng.uniform(0.15, 0.5) / (1.0 + ls) ** rng.uniform(1.0, 1.8))
        return sd_star_sh(a, L, scale=s), {"L": L, "amp": float(np.abs(a).max())}
    if kind == "ellipsoid":
        ax = s * rng.uniform(0.55, 1.45, 3)
        return sd_ellipsoid(axes=ax, rot=R), {"axes": ax.tolist()}
    if kind == "polytope":
        n = int(rng.integers(8, 40))
        u = rng.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        d = s * rng.uniform(0.55, 1.15, n)
        return sd_convex(u, d), {"n_planes": n}
    if kind == "contact_binary":
        # Two ellipsoids forced APART, with the fillet capped low. `lobes` cannot produce this
        # shape: its centres are drawn N(0,I) * U(0.25,0.55) against semi-axes U(0.35,0.75),
        # so the components fuse, and op_smooth_union with a fillet up to 0.18 fills whatever
        # crease survives. Measured over the library as shipped, the MEDIAN body has a neck
        # ratio of about 1.0 -- deep necks exist only in the tail -- while a deep central cut
        # is exactly what the hard public body needs. Separation >= 0.9 (a1x + a2x) puts the
        # components at or past tangency, so the waist is a real pinch rather than a dimple.
        a1 = s * rng.uniform(0.42, 0.62, 3)
        a2 = s * rng.uniform(0.34, 0.55, 3)
        sep = rng.uniform(0.90, 1.02) * (a1[0] + a2[0])
        k_fill = float(rng.uniform(0.01, 0.05))
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        c = 0.5 * sep * u
        f = op_smooth_union(sd_ellipsoid(centre=-c, axes=a1, rot=R),
                            sd_ellipsoid(centre=c, axes=a2, rot=R), k=k_fill)
        return f, {"axes1": a1.tolist(), "axes2": a2.tolist(),
                   "separation": float(sep), "fillet": k_fill}
    if kind == "lobes":
        k = int(rng.integers(2, 5))
        fs, cs = [], []
        for _ in range(k):
            c = rng.normal(size=3) * s * rng.uniform(0.25, 0.55)
            ax = s * rng.uniform(0.35, 0.75, 3)
            fs.append(sd_ellipsoid(centre=c, axes=ax, rot=_rand_rot(rng)))
            cs.append(c.tolist())
        out = fs[0]
        kk = s * rng.uniform(0.02, 0.18)
        for g in fs[1:]:
            out = op_smooth_union(out, g, k=kk)
        return out, {"n_lobes": k, "fillet": float(kk)}
    if kind == "prism":
        n = int(rng.integers(3, 11))
        ang = np.arange(n) * 2 * np.pi / n + rng.uniform(0, 2 * np.pi)
        jit = 1.0 + rng.normal(0, 0.10, n) if rng.random() < 0.6 else np.ones(n)
        nrm = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1)
        d = s * rng.uniform(0.45, 0.95) * jit
        caps = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        hz = s * rng.uniform(0.5, 1.4)
        f = sd_convex(np.vstack([nrm, caps]), np.concatenate([d, [hz, hz]]))
        return (lambda p, f=f, R=R: f(p @ R.T)), {"n_sides": n, "half_height": float(hz)}
    if kind == "rubble":
        k = int(rng.integers(5, 14))
        fs = []
        for _ in range(k):
            c = rng.normal(size=3) * s * rng.uniform(0.15, 0.65)
            ax = s * rng.uniform(0.18, 0.5, 3)
            fs.append(sd_ellipsoid(centre=c, axes=ax, rot=_rand_rot(rng)))
        out = fs[0]
        kk = s * rng.uniform(0.01, 0.08)
        for g in fs[1:]:
            out = op_smooth_union(out, g, k=kk)
        return out, {"n_grains": k}
    if kind == "arch":
        body = sd_ellipsoid(axes=s * rng.uniform(0.7, 1.2, 3), rot=R)
        hole = sd_cylinder(centre=rng.normal(size=3) * s * 0.15,
                           axis=R[rng.integers(0, 3)], radius=s * rng.uniform(0.15, 0.35),
                           half_height=3.0 * s)
        return op_subtract(body, hole), {"genus": 1}
    if kind == "slab":
        half = s * np.array([rng.uniform(0.6, 1.2), rng.uniform(0.5, 1.1),
                             rng.uniform(0.25, 0.6)])
        return (sd_box(half=half, rot=R, round_r=s * rng.uniform(0.0, 0.18)),
                {"half": half.tolist()})
    raise ValueError(kind)


def min_feature_radius(res: int, extent: float, voxels_across: float = 2.5) -> float:
    """Smallest sphere radius marching cubes can render as a round bowl rather than a
    blocky/aliased lump, at a grid of `res` samples across `2*extent`.

    A feature narrower than a couple of grid cells doesn't get enough sample points on its
    boundary for marching cubes to reconstruct a round surface -- it comes out as jagged
    voxel-aligned facets instead, which is the opposite of "small and clean." 2.5 voxels
    across the radius (5 across the diameter) is a practical floor, not a hard
    mathematical one: below it, quality visibly degrades before the feature disappears.
    """
    spacing = 2.0 * extent / max(res - 1, 1)
    return voxels_across * spacing


def _apply_modifier(f: Field, rng: np.random.Generator, kind: str, s: float,
                    strength: float, res: int = 96, extent: float = 1.6) -> tuple:
    """One geometric edit. `strength` >= 1 intensifies it when the gate has to be retried.

    `res`/`extent` are only used by size-sensitive modifiers (`pitted`) to keep features
    from being drawn smaller than marching cubes can actually resolve on the grid they'll
    be extracted at -- see `min_feature_radius`.
    """
    if kind == "craters":
        k = int(rng.integers(2, 9))
        cuts = []
        for _ in range(k):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            rad = s * rng.uniform(0.15, 0.45) * strength
            depth = rng.uniform(0.25, 0.85)          # fraction of the cutter inside
            cuts.append(sd_sphere(u * s * (1.0 - depth * 0.55) + u * rad * (1 - depth), rad))
        return op_subtract(f, *cuts), {"n_craters": k}
    if kind == "pitted":
        # a permuted lattice, not independent random directions -- see _fibonacci_sphere
        n_lattice = int(rng.integers(200, 361))
        lattice = _fibonacci_sphere(n_lattice)
        order = rng.permutation(n_lattice)
        min_r = min_feature_radius(res, extent)
        # coverage (~ k * r^2) is what should stay roughly constant across resolutions,
        # not k itself: when the floor inflates r at a coarse grid, k has to shrink or the
        # craters overlap into a jagged mass instead of staying distinct small bowls.
        # The reference point is res=96's OWN floor, not an independent constant smaller
        # than it -- using a smaller constant here previously meant count was shrunk even
        # at res=96, silently undoing the "many small craters" behaviour at the one
        # resolution meant to show it off cleanly.
        reference_r = min_feature_radius(96, extent)
        area_scale = 1.0 if min_r <= reference_r else (reference_r / min_r) ** 2
        k = max(6, int(rng.integers(40, 91) * area_scale))
        k = min(k, n_lattice)
        cuts = []
        for idx in order[:k]:
            u = lattice[idx]
            # target radius is deliberately small (asteroid regolith, not lunar maria);
            # the max() with min_r keeps it from being drawn smaller than this grid can
            # actually render -- at low res that floor dominates and craters come out a
            # bit larger than the target (with k reduced above to compensate), at high
            # res the target dominates and craters come out genuinely tiny.
            target = s * rng.uniform(0.020, 0.055) * strength
            rad = max(min_r, target)
            depth = rng.uniform(0.25, 0.55)      # shallow bowls, not deep bites
            cuts.append(sd_sphere(u * s * (1.0 - depth * 0.55) + u * rad * (1 - depth), rad))
        return op_subtract(f, *cuts), {"n_pits": k, "lattice_n": n_lattice,
                                       "min_r": float(min_r)}
    if kind == "basin":
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        rad = s * rng.uniform(0.7, 1.5) * strength
        off = s * rng.uniform(0.75, 1.25) + rad * rng.uniform(0.35, 0.8)
        return op_subtract(f, sd_sphere(u * off, rad)), {"basin_radius": float(rad)}
    if kind == "cuts":
        k = int(rng.integers(1, 5))
        u = rng.normal(size=(k, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        d = s * rng.uniform(0.45, 0.95, k)
        return op_intersect(f, sd_convex(u, d)), {"n_cuts": k}
    if kind == "groove":
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        maj = s * rng.uniform(0.55, 1.0)
        mnr = s * rng.uniform(0.06, 0.22) * strength
        return op_subtract(f, sd_torus(axis=u, major=maj, minor=mnr)), {"groove": float(mnr)}
    if kind == "waist":
        z0 = rng.uniform(-0.5, 0.5) * s
        w = s * rng.uniform(0.2, 0.6)
        amp = s * rng.uniform(0.15, 0.5) * strength

        def d(p, z0=z0, w=w, amp=amp):
            return amp * np.exp(-((p[..., 2] - z0) ** 2) / (2 * w * w))
        return op_displace(f, d), {"waist_amp": float(amp)}
    if kind == "bite":
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        rad = s * rng.uniform(0.45, 0.9) * strength
        return op_subtract(f, sd_sphere(u * (s * rng.uniform(0.6, 1.0)), rad)), {"bite": 1}
    if kind == "scallops":
        k = int(rng.integers(2, 6))
        cuts = []
        for _ in range(k):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            ax = s * rng.uniform(0.2, 0.55, 3) * strength
            cuts.append(sd_ellipsoid(u * s * rng.uniform(0.8, 1.15), ax, _rand_rot(rng)))
        return op_subtract(f, *cuts), {"n_scallops": k}
    if kind == "boulders":
        k = int(rng.integers(2, 8))
        adds = []
        for _ in range(k):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            adds.append(sd_sphere(u * s * rng.uniform(0.7, 1.0), s * rng.uniform(0.08, 0.22)))
        return op_union(f, *adds), {"n_boulders": k}
    if kind == "roughness":
        L = int(rng.integers(4, 12))
        return op_displace(f, _sh_displacement(rng, L, s * rng.uniform(0.02, 0.10))), {"rough_L": L}
    if kind == "tunnel":
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        return op_subtract(f, sd_cylinder(centre=rng.normal(size=3) * s * 0.2, axis=u,
                                          radius=s * rng.uniform(0.10, 0.25),
                                          half_height=4.0 * s)), {"tunnel": 1}
    raise ValueError(kind)


def _draw(d: dict, rng: np.random.Generator) -> str:
    ks = list(d)
    w = np.array([d[k] for k in ks], float)
    return ks[int(rng.choice(len(ks), p=w / w.sum()))]


def sample_body(rng: np.random.Generator, spec: LibrarySpec | None = None) -> Body:
    """One posed, closed, single-component, strictly non-convex body.

    The non-convexity gate is enforced by resampling with a larger `strength`, so the
    returned body is guaranteed to satisfy it rather than merely likely to.
    """
    spec = spec or LibrarySpec()
    last = None
    # Drawn ONCE, outside the retry loop. Redrawing it per attempt biases the realised base
    # distribution towards whatever survives the non-convexity gate first: measured, prism
    # fell from its nominal 0.12 to 0.037 and polytope rose from 0.12 to 0.225, while
    # write_report printed base_weights as though it had been honoured.
    base_kind = _draw(spec.base_weights, rng)
    for attempt in range(spec.max_attempts):
        strength = 1.0 + 0.25 * attempt
        s = 1.0
        f, rec = _base(rng, base_kind, s)
        recipe = {"base": base_kind, **rec, "mods": []}
        n_mod = int(rng.integers(spec.n_modifiers[0], spec.n_modifiers[1] + 1))
        for _ in range(n_mod):
            mk = _draw(spec.mod_weights, rng)
            f, mrec = _apply_modifier(f, rng, mk, s, strength, res=spec.res, extent=spec.extent)
            recipe["mods"].append({"kind": mk, **mrec})
        try:
            v, fc, info = extract(f, extent=spec.extent, res=spec.res)
        except (ValueError, RuntimeError) as e:
            last = f"extract failed: {e}"
            continue
        if len(fc) < 100:
            last = "degenerate: too few faces"
            continue
        v = pose(v, radius=spec.radius, faces=fc)
        c = convexity_ratio(v, fc)
        info.update({"convexity": c, "attempt": attempt, "n_faces": len(fc),
                     "n_verts": len(v)})
        if c >= spec.convexity_max:
            last = f"convexity {c:.3f} >= {spec.convexity_max}"
            continue
        if not is_edge_manifold(fc):
            last = "not edge-manifold (marching-cubes pinch or crack)"
            continue
        if n_components(v, fc) != 1:
            last = "surface split into >1 component (voxel-scale pinch)"
            continue
        return Body(v, fc, recipe, info)
    raise RuntimeError(f"no body passed the gate in {spec.max_attempts} attempts ({last})")


def build_library(n: int, seed: int = 0, spec: LibrarySpec | None = None,
                  progress: bool = False) -> list:
    """`n` independent bodies. Each draws its own generator, so the library is
    reproducible from (seed, n) and any single body can be rebuilt without the rest."""
    spec = spec or LibrarySpec()
    out = []
    for i in range(n):
        b = sample_body(np.random.default_rng([seed, i]), spec)
        out.append(b)
        if progress and (i % 10 == 0 or i == n - 1):
            print(f"  body {i + 1}/{n}  base={b.recipe['base']:<9} "
                  f"conv={b.convexity:.3f}  faces={b.info['n_faces']}", flush=True)
    return out


# --------------------------------------------------------------------------- ingestion

def body_from_mesh(verts: np.ndarray, faces: np.ndarray,
                   rng: np.random.Generator | None = None,
                   spec: LibrarySpec | None = None,
                   add_modifiers: int = 0) -> Body:
    """Bring an external mesh (Thingi10K) into the library.

    The source is voxelised by ray-parity along z, which needs the source to be closed but
    tolerates self-intersection and duplicated faces; the occupancy is then repaired and
    re-meshed exactly as a generated body is, so an ingested body carries the same
    guarantees as a sampled one.

    `add_modifiers` is a MINIMUM, not a fixed count: if the source is already non-convex
    enough on its own, it is admitted with that many edits (0 is allowed, and keeps a
    genuinely non-convex source unmodified); if it is not -- most Thingi10K objects that
    happen to be printable housewares are close to convex -- edits are added and
    intensified exactly as `sample_body` does, so ingestion is held to the same gate as
    every other source rather than a weaker one.
    """
    spec = spec or LibrarySpec()
    rng = rng or np.random.default_rng(0)
    v = np.asarray(verts, float)
    v = v - v.mean(0)
    v = v / max(float(np.abs(v).max()), 1e-12)
    occ = _parity_occupancy(v, np.asarray(faces, np.int64), spec.extent, spec.res)
    f0 = _field_from_occupancy(occ, spec.extent, spec.res)

    last = None
    for attempt in range(spec.max_attempts):
        f = f0
        n_mod = add_modifiers + attempt
        recipe = {"base": "mesh", "source_faces": int(len(faces)), "mods": []}
        for _ in range(n_mod):
            mk = _draw(spec.mod_weights, rng)
            f, mrec = _apply_modifier(f, rng, mk, 1.0, 1.0 + 0.25 * attempt,
                                      res=spec.res, extent=spec.extent)
            recipe["mods"].append({"kind": mk, **mrec})
        try:
            vv, ff, info = extract(f, extent=spec.extent, res=spec.res)
        except (ValueError, RuntimeError) as e:
            last = f"extract failed: {e}"
            continue
        vv = pose(vv, radius=spec.radius, faces=ff)
        c = convexity_ratio(vv, ff)
        if c >= spec.convexity_max:
            last = f"convexity {c:.3f} >= {spec.convexity_max}"
            continue
        if not is_edge_manifold(ff):
            last = "not edge-manifold"
            continue
        if n_components(vv, ff) != 1:
            last = "surface split into >1 component"
            continue
        info.update({"convexity": c, "n_faces": len(ff), "n_verts": len(vv),
                     "attempt": attempt})
        return Body(vv, ff, recipe, info)
    raise RuntimeError(f"ingested mesh never passed the gate in {spec.max_attempts} "
                       f"attempts ({last})")


def body_from_convex_points(points: np.ndarray, rng: np.random.Generator,
                            spec: LibrarySpec | None = None,
                            n_modifiers: int = 4) -> Body:
    """DAMIT path: a convex body used ONLY as a basis, with non-convexity added.

    DAMIT shapes are convex inversions -- they carry no concavity at all, so using them
    directly would train the prior on exactly the geometry the challenge is about
    recovering. The hull enters as the starting field and the gate is enforced afterwards,
    so a body that came from DAMIT cannot reach the library still convex.
    """
    spec = spec or LibrarySpec()
    p = np.asarray(points, float)
    p = p - p.mean(0)
    p = p / max(float(np.abs(p).max()), 1e-12)
    hull = ConvexHull(p)
    nrm = hull.equations[:, :3]
    off = -hull.equations[:, 3]
    f = sd_convex(nrm, off)
    recipe = {"base": "damit_convex", "n_planes": int(len(nrm)), "mods": []}
    for attempt in range(spec.max_attempts):
        g = f
        recipe["mods"] = []
        for _ in range(max(1, n_modifiers)):
            mk = _draw(spec.mod_weights, rng)
            g, mrec = _apply_modifier(g, rng, mk, 1.0, 1.0 + 0.3 * attempt,
                                      res=spec.res, extent=spec.extent)
            recipe["mods"].append({"kind": mk, **mrec})
        v, fc, info = extract(g, extent=spec.extent, res=spec.res)
        v = pose(v, radius=spec.radius, faces=fc)
        c = convexity_ratio(v, fc)
        if c < spec.convexity_max and is_edge_manifold(fc) and n_components(v, fc) == 1:
            info.update({"convexity": c, "n_faces": len(fc), "n_verts": len(v)})
            return Body(v, fc, recipe, info)
    raise RuntimeError("convex basis never brought below the non-convexity gate")


def _parity_occupancy(verts: np.ndarray, faces: np.ndarray, extent: float,
                      res: int) -> np.ndarray:
    """Occupancy by counting triangle crossings along +z through each (x, y) column.

    Parity, not winding: a point is inside when the number of crossings above it is odd.
    Correct for any closed surface regardless of orientation, and it is the reason ingestion
    does not need trimesh.
    """
    a = np.linspace(-extent, extent, res)
    zs = a
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    X, Y = np.meshgrid(a, a, indexing="ij")
    px, py = X.ravel(), Y.ravel()
    count = np.zeros((len(px), res), dtype=np.int32)
    for t in range(0, len(faces), 3000):
        A, B, C = v0[t:t + 3000], v1[t:t + 3000], v2[t:t + 3000]
        d = ((B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0])
             + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1]))
        ok = np.abs(d) > 1e-14
        if not ok.any():
            continue
        A, B, C, d = A[ok], B[ok], C[ok], d[ok]
        l1 = ((B[None, :, 1] - C[None, :, 1]) * (px[:, None] - C[None, :, 0])
              + (C[None, :, 0] - B[None, :, 0]) * (py[:, None] - C[None, :, 1])) / d
        l2 = ((C[None, :, 1] - A[None, :, 1]) * (px[:, None] - C[None, :, 0])
              + (A[None, :, 0] - C[None, :, 0]) * (py[:, None] - C[None, :, 1])) / d
        l3 = 1.0 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not inside.any():
            continue
        zh = l1 * A[None, :, 2] + l2 * B[None, :, 2] + l3 * C[None, :, 2]
        col, tri = np.nonzero(inside)
        zc = zh[col, tri]
        idx = np.clip(np.searchsorted(zs, zc), 0, res - 1)      # crossings above each plane
        np.add.at(count, (col, idx), 1)
    occ = (np.cumsum(count[:, ::-1], axis=1)[:, ::-1] % 2 == 1)
    return occ.reshape(res, res, res)


def _field_from_occupancy(occ: np.ndarray, extent: float, res: int) -> Field:
    """A smooth signed field from a binary occupancy, by the distance transform.

    Distance-to-boundary inside minus distance-to-boundary outside, trilinearly
    interpolated. Meshing this rather than the raw mask is what keeps an ingested body from
    coming out with voxel stairsteps.
    """
    sp = 2.0 * extent / (res - 1)
    din = ndimage.distance_transform_edt(occ, sampling=sp)
    dout = ndimage.distance_transform_edt(~occ, sampling=sp)
    sdf = dout - din
    lo = -extent

    def f(p):
        idx = (np.asarray(p) - lo) / sp
        return ndimage.map_coordinates(sdf, idx.reshape(-1, 3).T, order=1,
                                       mode="nearest").reshape(p.shape[:-1])
    return f


def load_thingi10k(directory: str, limit: int | None = None,
                   spec: LibrarySpec | None = None,
                   rng: np.random.Generator | None = None,
                   add_modifiers: int = 2) -> list:
    """Every .stl under `directory`, ingested. Files that fail are skipped with a warning.

    Thingi10K is not redistributed here and is not required: the procedural families alone
    clear the diversity gate. Point this at a local copy to fold real scanned geometry in.
    """
    from pathlib import Path
    from .stl_io import load_stl

    rng = rng or np.random.default_rng(0)
    out = []
    for i, p in enumerate(sorted(Path(directory).rglob("*.stl"))):
        if limit is not None and len(out) >= limit:
            break
        try:
            v, f = load_stl(str(p))
            out.append(body_from_mesh(v, f, rng, spec, add_modifiers))
        except Exception as e:                                   # noqa: BLE001
            warnings.warn(f"skipped {p.name}: {e}")
    return out

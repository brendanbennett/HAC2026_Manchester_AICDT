"""End-to-end reconstruction and evaluation utilities.

Pipeline: normalized curves (+mask) -> LPD -> scale-free EGI p -> closure projection
-> Minkowski polytope -> challenge pose (z in [-1,1], xy-centroid on the axis) -> STL.

The body frame equals the pose at frame 0 by construction of the operator, which is
exactly the pose the submission requires ("same position as at the beginning of the
lightcurves"); no extra registration is needed.

Evaluation: voxel Dice — identical to the challenge voxel measure by the identity
1 - (#(A\\B)+#(B\\A))/(#A+#B) = 2#(A&B)/(#A+#B).
"""
from __future__ import annotations

import numpy as np

from .geometry import project_closure
from hac26.solvers.minkowski import solve_minkowski
from .shapes import face_normals_areas, rescale_touch_z
from .stl_io import save_stl


def reconstruct_from_curves(net, grid, curves56: np.ndarray, mask56: np.ndarray,
                            device: str = "cpu", drop_tol: float = 2e-4,
                            radius: float | None = None) -> dict:
    """Curves -> EGI -> Minkowski polytope -> challenge pose.

    `radius` is the a-priori bounding radius, required only when the checkpoint was
    trained with r_cond (the network reads it as an input, exactly as in training)."""
    import torch  # local import: the numpy side of the package works without torch

    net.eval()
    with torch.no_grad():
        d = torch.as_tensor(curves56, dtype=torch.float32, device=device)[None]
        mk = torch.as_tensor(mask56, dtype=torch.float32, device=device)[None]
        if getattr(net, "r_cond", False):
            if radius is None:
                raise ValueError("this checkpoint needs the bounding radius")
            lr = torch.tensor([float(np.log(radius))], dtype=torch.float32, device=device)
            p = net(d, mk, lr)[0]
        else:
            p = net(d, mk)[0]
    p = p[0].cpu().numpy().astype(float)
    p = project_closure(p, grid.normals)
    sol = solve_minkowski(grid.normals, p, drop_tol=drop_tol)
    verts = rescale_touch_z(sol["verts"])
    return {"p": p, "verts": verts, "faces": sol["faces"], "minkowski": sol}


def body_from_support(normals: np.ndarray, h: np.ndarray, eps: float = 1e-3) -> tuple:
    """Convex body as the intersection of half-spaces {x : <x,u> <= h(u)}.

    Total replacement for `solve_minkowski` on the support-function path, and far
    better behaved: it is a closed-form polytope construction valid for ANY positive
    h, with no closure constraint, no optimisation, and no failure mode. (If h is not
    literally a support function the result is the body whose support function is the
    convex envelope of h -- a projection, not an error.)"""
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    h = np.maximum(np.asarray(h, dtype=float), eps)
    halfspaces = np.hstack([normals, -h[:, None]])
    hs = HalfspaceIntersection(halfspaces, np.zeros(3))
    pts = hs.intersections
    hull = ConvexHull(pts)
    return pts, hull.simplices


def smooth_support(h: np.ndarray, n_theta: int, n_phi: int, k: int = 1) -> np.ndarray:
    """Box-average h over a (2k+1)^2 neighbourhood; circular in phi, clamped in theta.

    The half-space decoder is a MIN over constraints, so one spuriously low h_n shears
    a slab off the whole body -- it is max-norm sensitive, not L2 sensitive. Genuine
    support functions of bounded bodies are Lipschitz on the sphere while the error is
    not, so a mild low-pass removes exactly the outliers that do the damage. Measured
    on held-out shapes: at 40% relative error this recovers Dice 0.57 -> 0.74."""
    H = np.asarray(h, dtype=float).reshape(n_theta, n_phi)
    acc, cnt = np.zeros_like(H), 0
    for dt in range(-k, k + 1):
        rows = np.clip(np.arange(n_theta) + dt, 0, n_theta - 1)
        for dp in range(-k, k + 1):
            acc += H[rows][:, (np.arange(n_phi) + dp) % n_phi]
            cnt += 1
    return (acc / cnt).reshape(-1)


def reconstruct_from_support(net, grid, curves56: np.ndarray, mask56: np.ndarray,
                             device: str = "cpu", smooth: int = 0,
                             radius: float | None = None) -> dict:
    """Curves -> support function -> polytope -> challenge pose.

    `radius` is the model's a-priori bounding radius, required when the checkpoint was
    trained with r_cond (the network reads it as an input, exactly as in training)."""
    import torch

    net.eval()
    with torch.no_grad():
        d = torch.as_tensor(curves56, dtype=torch.float32, device=device)[None]
        mk = torch.as_tensor(mask56, dtype=torch.float32, device=device)[None]
        if getattr(net, "r_cond", False):
            if radius is None:
                raise ValueError("this checkpoint needs the bounding radius (--fit-cylinder "
                                 "supplies it from CYLINDER_R)")
            lr = torch.tensor([float(np.log(radius))], dtype=torch.float32, device=device)
            out = net(d, mk, lr)
        else:
            out = net(d, mk)
    h = out[2][0].cpu().numpy().astype(float)
    if smooth:
        h = smooth_support(h, grid.n_theta, grid.n_phi, k=smooth)
    verts, faces = body_from_support(grid.normals, h)
    from .shapes import hull_mesh
    verts, faces = hull_mesh(verts)
    return {"h": h, "verts": rescale_touch_z(verts), "faces": faces}


def fit_to_cylinder(verts: np.ndarray, radius: float) -> np.ndarray:
    """Scale x,y so the body's max axis distance equals the a-priori cylinder radius.

    The challenge publishes a bounding-cylinder base radius R per model. Measured on
    the public models, that bound is *tight* in the challenge pose (z in [-1,1]):
    official r/R = 0.99 / 1.03 / 1.01 for models 1/2/3. So R is not a loose box, it
    pins the aspect ratio -- information the scale-free EGI inversion cannot recover
    on its own (normalization cancels overall scale, and the pose then fixes only z).

    Applying it is anisotropic (xy only), which is exactly the missing degree of
    freedom: the EGI fixes the *shape* of the hull up to scale, the pose fixes the
    height, and R fixes the width. z is untouched so the pose stays valid.
    """
    v = verts.copy()
    r = float(np.sqrt((v[:, :2] ** 2).sum(1)).max())
    if r > 1e-12 and radius and radius > 0:
        v[:, :2] *= radius / r
    return v


def save_submission_stl(path: str, verts: np.ndarray, faces: np.ndarray,
                        cylinder_radius: float | None = None) -> dict:
    """Write the posed STL; report the a-priori cylinder check (not enforced)."""
    info = {"zmin": float(verts[:, 2].min()), "zmax": float(verts[:, 2].max()),
            "max_axis_dist": float(np.sqrt((verts[:, :2] ** 2).sum(1)).max())}
    if cylinder_radius is not None:
        info["cylinder_radius_prior"] = cylinder_radius
        info["inside_prior_cylinder"] = bool(info["max_axis_dist"] <= cylinder_radius + 1e-9)
    save_stl(path, verts, faces)
    return info


# ---------- voxel evaluation ---------------------------------------------------------
def voxel_grid(extent_xy: float, n: int = 128) -> np.ndarray:
    """Cell-center points of an n x n x n grid on [-e,e]^2 x [-1,1], shape (n^3, 3)."""
    x = (np.arange(n) + 0.5) / n * 2 * extent_xy - extent_xy
    z = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
    return np.stack([X, Y, Z], axis=-1).reshape(-1, 3)


def voxelize_convex(verts: np.ndarray, faces: np.ndarray, pts: np.ndarray,
                    max_elems: int = 1 << 24) -> np.ndarray:
    """Inside test for a convex mesh via its facet halfspaces (exact for convex).

    The `pts @ n.T` temporary is (block, n_facets), so the block size has to shrink
    as the facet count grows — a fixed block overflows RAM on the challenge STLs
    (asteroid1's hull has 2.4e4 facets: 2.6e5 x 2.4e4 float64 = 47 GiB). Cap the
    temporary at max_elems entries (default 2^24 ~ 128 MB) instead.
    """
    n, a = face_normals_areas(verts, faces)
    keep = a > 1e-14
    n = n[keep]
    d = (n * verts[faces[keep, 0]]).sum(1)
    inside = np.ones(len(pts), dtype=bool)
    block = max(1, min(262144, max_elems // max(1, len(n))))
    for i in range(0, len(pts), block):
        blk = pts[i:i + block]
        inside[i:i + block] = np.all(blk @ n.T <= d[None, :] + 1e-9, axis=1)
    return inside


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Soerensen-Dice = the challenge voxel measure."""
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else 1.0


# Scoring utility: mesh -> occupancy -> signed distance, to voxelise a reconstruction and
# a ground truth onto a common grid. numpy, scipy and trimesh only.

def mesh_to_sdf(verts: np.ndarray, faces: np.ndarray, n: int, extent: float) -> np.ndarray:
    """Signed distance field of a mesh on a cubic grid, negative inside.

    Occupancy by point-in-mesh test, then a Euclidean distance transform on each side.
    Accurate to about one voxel, which is the resolution the field is stored at anyway.
    """
    import trimesh
    from scipy.ndimage import distance_transform_edt

    ax = (np.arange(n) + 0.5) / n * 2.0 * extent - extent
    pts = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    inside = np.zeros(len(pts), dtype=bool)
    for i in range(0, len(pts), 200_000):         # contains() is memory-hungry
        sl = slice(i, i + 200_000)
        inside[sl] = mesh.contains(pts[sl])
    inside = inside.reshape(n, n, n)
    voxel = 2.0 * extent / n
    d_out = distance_transform_edt(~inside) * voxel
    d_in = distance_transform_edt(inside) * voxel
    return (d_out - d_in).astype(np.float32)

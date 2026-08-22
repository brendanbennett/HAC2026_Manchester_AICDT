"""Radiosity, solved once per shape rather than once per phase.

The body, mount and turntable are mutually rigid, and the solve is done in the body frame, so the
geometry between facets never changes as the turntable turns. Everything that depends only
on that geometry -- the form-factor matrix F, and therefore the factorisation of
(I - rho F) -- is computed once per candidate shape. The phases are then back-substitutions
against that one factorisation, and can go through it as a single right-hand-side block; the
adjoint solve (I - rho F)^T lambda = dJ/dB reuses the same factors.

    F_ij = (1/A_i) int int V(x,y) cos(theta_x) cos(theta_y) / (pi r^2) dA_i dA_j
    A_i F_ij = A_j F_ji                          (reciprocity)
    (I - rho F) B = rho e(psi)
    e_i(psi) = (E0/K) sum_k (n_i . omega_k(psi))+ V_i(omega_k(psi))
    L_i = B_i / pi                               (radiance leaving facet i)

Only the emission e depends on the phase, and only through which source samples each facet
can see. That is the whole reason this decomposition is worth the trouble: the expensive
object is phase-independent and the phase-dependent one is a triangular solve.

Two checks, both asserted rather than repaired:
  * F is symmetrised through reciprocity before use, since the centroid quadrature below
    does not produce a reciprocal matrix by itself;
  * sum_j F_ij <= 1 for every i, because a facet cannot see more than the whole hemisphere.
    A row sum above 1 means the quadrature has broken down, and clamping it would hide that.
"""
from __future__ import annotations

import numpy as np

__all__ = ["facet_geometry", "form_factors", "RadiositySolver", "emission"]


def facet_geometry(verts: np.ndarray, faces: np.ndarray):
    """Centroids, unit normals and areas of every facet."""
    tv = verts[faces]
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    a2 = np.linalg.norm(n, axis=1)
    keep = a2 > 1e-14
    tv, n, a2 = tv[keep], n[keep], a2[keep]
    return tv.mean(1), n / a2[:, None], 0.5 * a2


def _visibility_matrix(centroids: np.ndarray, normals: np.ndarray,
                       verts: np.ndarray, faces: np.ndarray,
                       backface_only: bool = False) -> np.ndarray:
    """V(i,j): can facet centroids i and j see each other through the exterior?

    Facets that do not face each other are excluded first -- that is exact and free -- and
    only the survivors are ray-tested, which is what keeps this affordable.
    """
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    n_f = len(centroids)
    d = centroids[None, :, :] - centroids[:, None, :]
    r = np.linalg.norm(d, axis=2)
    np.fill_diagonal(r, np.inf)
    u = d / np.maximum(r, 1e-12)[:, :, None]
    cos_i = (u * normals[:, None, :]).sum(2)
    cos_j = -(u * normals[None, :, :]).sum(2)
    facing = (cos_i > 1e-9) & (cos_j > 1e-9)
    if backface_only:
        return facing.astype(np.float64), cos_i, cos_j, r
    ii, jj = np.nonzero(np.triu(facing, 1))
    V = np.zeros((n_f, n_f), dtype=bool)
    if len(ii):
        eps = 1e-4 * np.maximum(r[ii, jj], 1e-9)
        o = centroids[ii] + normals[ii] * eps[:, None]
        dirs = u[ii, jj]
        # a hit beyond the partner does not block, so the distance is tested explicitly.
        # One cast: an intersects_any pass over the same rays answers a weaker question and
        # its result was thrown away.
        loc, idx_ray, _ = m.ray.intersects_location(o, dirs, multiple_hits=False)
        blocked = np.zeros(len(ii), dtype=bool)
        if len(idx_ray):
            dist = np.linalg.norm(loc - o[idx_ray], axis=1)
            blocked[idx_ray] = dist < r[ii, jj][idx_ray] * (1 - 1e-3)
        ok = ~blocked
        V[ii[ok], jj[ok]] = True
        V[jj[ok], ii[ok]] = True
    return V.astype(np.float64), cos_i, cos_j, r


def _barycentric_samples(verts: np.ndarray, faces: np.ndarray, n_samples: int):
    """n_samples points per facet, at fixed barycentric positions (centroid for n=1)."""
    tv = verts[faces]
    if n_samples == 1:
        return tv.mean(1)[:, None, :]
    bary = {
        3: [(2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)],
        4: [(1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)],
    }[n_samples]
    return np.stack([b[0] * tv[:, 0] + b[1] * tv[:, 1] + b[2] * tv[:, 2] for b in bary], 1)


def form_factors(verts: np.ndarray, faces: np.ndarray, occlusion: bool = True,
                 n_samples: int = 4):
    """F with reciprocity imposed, plus the row-sum diagnostic.

    Quadrature of the double integral with n_samples points per facet. Single-point
    (centroid) quadrature is the usual first-order approximation and is exact only for
    facets small relative to their separation; on a coarse mesh it OVERESTIMATES near-field
    pairs badly enough to push row sums above 1, which the diagnostic in RadiositySolver
    then correctly rejects. Four points per facet integrates the 1/r^2 kernel over the
    facet instead of sampling it at one point,
    which is what fixes it. Visibility is still evaluated at the centroids: it is a binary
    quantity that varies far more slowly than 1/r^2.
    """
    c, n, a = facet_geometry(verts, faces)
    V, _, _, r_c = _visibility_matrix(c, n, verts, faces, backface_only=not occlusion)
    P = _barycentric_samples(verts, faces[np.arange(len(faces))], n_samples)
    if len(P) != len(c):                      # facet_geometry drops degenerate faces
        P = _barycentric_samples(verts, faces, n_samples)[: len(c)]
    # No (i,j,si,sj,3) difference is formed. cos_i * cos_j / r^2 is
    #     clip(d.n_i, 0) * clip(-d.n_j, 0) / (pi * r^4)
    # and both dot products factorise -- d.n_i = P_j.n_i - P_i.n_i -- while r^2 comes from
    # the expanded square. The largest array is then (i,j,si,sj) instead of that times three,
    # and no square root is taken. Same result to the last bit.
    nf, S = len(c), P.shape[1]
    flat = P.reshape(nf * S, 3)
    sq = (flat ** 2).sum(1)
    rr2 = (sq[:, None] + sq[None, :]
           - 2.0 * (flat @ flat.T)).reshape(nf, S, nf, S).transpose(0, 2, 1, 3)
    np.maximum(rr2, 0.0, out=rr2)
    rr2[np.arange(nf), np.arange(nf)] = np.inf
    dni = (np.einsum("jbk,ik->ijb", P, n)[:, :, None, :]
           - np.einsum("iak,ik->ia", P, n)[:, None, :, None])
    dnj = (np.einsum("jbk,jk->jb", P, n)[None, :, None, :]
           - np.einsum("iak,jk->ija", P, n)[:, :, :, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.clip(dni, 0, None) * np.clip(-dnj, 0, None) / (np.pi * rr2 ** 2)
    k[~np.isfinite(k)] = 0.0
    K = V * k.mean(axis=(2, 3))
    F = K * a[None, :]                       # F_ij = K_ij A_j
    np.fill_diagonal(F, 0.0)

    # Reciprocity: A_i F_ij = A_j F_ji. The quadrature does not deliver this exactly, so
    # symmetrise the reciprocal form G_ij = A_i F_ij and read F back off it.
    G = a[:, None] * F
    G = 0.5 * (G + G.T)
    F = G / np.maximum(a[:, None], 1e-300)
    return F, a, n, c


class RadiositySolver:
    """Factor (I - rho F) once; every phase and every adjoint is a back-substitution."""

    def __init__(self, F: np.ndarray, rho: float, check_rows: bool = True,
                 row_tol: float = 1e-6):
        self.rho = float(rho)
        self.F = F
        if check_rows:
            rs = F.sum(1).max()
            if rs > 1.0 + row_tol:
                raise ValueError(
                    f"form-factor row sum {rs:.6f} exceeds 1; the quadrature has broken "
                    "down. This is asserted rather than clamped on purpose -- clamping "
                    "turns a broken transport model into a plausible-looking one.")
            self.max_row_sum = float(rs)
        from scipy.linalg import lu_factor
        self._lu = lu_factor(np.eye(len(F)) - self.rho * F)

    def solve(self, e: np.ndarray) -> np.ndarray:
        """B from emission e: (I - rho F) B = rho e. e may be (F,) or (F, n_rhs)."""
        from scipy.linalg import lu_solve
        return lu_solve(self._lu, self.rho * np.asarray(e, dtype=float))

    def solve_adjoint(self, dJ_dB: np.ndarray) -> np.ndarray:
        """(I - rho F)^T lambda = dJ/dB, reusing the same factors."""
        from scipy.linalg import lu_solve
        return lu_solve(self._lu, np.asarray(dJ_dB, dtype=float), trans=1)

    @staticmethod
    def radiance(B: np.ndarray) -> np.ndarray:
        return np.asarray(B) / np.pi


def emission(normals: np.ndarray, source_dirs: np.ndarray, vis: np.ndarray | None = None,
             e0: float = 1.0) -> np.ndarray:
    """e_i = (E0/K) sum_k (n_i . omega_k)+ V_i(omega_k).

    source_dirs (K, 3) with vis (n_facets, K) gives one emission vector. source_dirs
    (P, K, 3) with vis (n_facets, P, K) gives (n_facets, P) -- every phase at once, so
    RadiositySolver.solve can take them as one right-hand-side block. None means unoccluded.
    """
    d = np.asarray(source_dirs)
    if d.ndim == 2:
        mu = np.clip(np.asarray(normals) @ d.T, 0.0, None)
        if vis is not None:
            mu = mu * np.asarray(vis, dtype=float)
        return e0 / d.shape[0] * mu.sum(-1)
    mu = np.clip(np.einsum("fk,pjk->fpj", np.asarray(normals), d), 0.0, None)
    if vis is not None:
        mu = mu * np.asarray(vis, dtype=float)
    return e0 / d.shape[1] * mu.sum(-1)

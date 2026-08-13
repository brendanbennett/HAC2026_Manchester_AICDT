"""From posterior samples to one submitted body.

The medoid rather than the mean. The samples are draws from a posterior that provably
contains indistinguishable pairs, so averaging them is not a summary but a new body that
none of them is. Concretely: a crater whose longitude is uncertain appears at a different
place in each sample, and the mean smears it into a shallow depression everywhere, which is
to say back into a filled convex body. Thresholding a marginal occupancy field does the same
thing by a different route. The metric medoid

    x* = argmax_s (1/S) sum_s' Dice(B_s, B_s')

is instead an actual sample -- the one most typical under the very metric being scored --
so it keeps a crater somewhere rather than nowhere.

The planar snap is evidence-gated. Every ground truth is a printed polytope, so snapping
near-planar patches flat is usually right, and on the cube it is worth a great deal. But it
is wrong on a genuinely smooth body, so each candidate plane is accepted only if the data
misfit does not rise beyond the calibrated model-error floor eta. That makes the step
self-rejecting: on a smooth body no plane is accepted and nothing happens.

Constraints come last. The z-extent equality and the radius bound are restored as the final
operation, after the snap, because the snap moves vertices and would otherwise break them.
"""
from __future__ import annotations

import numpy as np

__all__ = ["dice_volumes", "metric_medoid", "ransac_planes", "planar_snap",
           "restore_constraints", "export_stl"]


def dice_volumes(occ_a: np.ndarray, occ_b: np.ndarray) -> float:
    inter = np.logical_and(occ_a, occ_b).sum()
    return float(2.0 * inter / max(occ_a.sum() + occ_b.sum(), 1))


def metric_medoid(occupancies) -> int:
    """Index of the sample with the highest mean Dice to all the others.

    The medoid is taken under the scored metric itself, not under a proxy.
    """
    s = len(occupancies)
    if s == 1:
        return 0
    mean_dice = np.zeros(s)
    for i in range(s):
        acc = 0.0
        for j in range(s):
            if i != j:
                acc += dice_volumes(occupancies[i], occupancies[j])
        mean_dice[i] = acc / (s - 1)
    return int(np.argmax(mean_dice))


def ransac_planes(verts: np.ndarray, faces: np.ndarray, n_planes: int = 12,
                  tol: float = 0.02, min_frac: float = 0.02, iters: int = 400,
                  seed: int = 0):
    """Candidate planes by RANSAC on the face centroids, largest support first."""
    rng = np.random.default_rng(seed)
    tv = verts[faces]
    cen = tv.mean(1)
    nrm = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    a = np.linalg.norm(nrm, axis=1)
    keep = a > 1e-14
    cen, nrm, area = cen[keep], nrm[keep] / a[keep, None], 0.5 * a[keep]
    # A plane needs enough support to be evidence, but the floor must scale with the mesh:
    # an absolute minimum of 8 triangles rejected every face of a 12-triangle cube, which is
    # the one shape this step exists to win.
    floor = max(2, int(min_frac * len(cen)))
    remaining = np.ones(len(cen), bool)
    out = []
    for _ in range(n_planes):
        idx = np.flatnonzero(remaining)
        if len(idx) < floor:
            break
        best = (0, None)
        for _ in range(iters):
            i = rng.choice(idx)
            n, d = nrm[i], float(nrm[i] @ cen[i])
            inl = remaining & (np.abs(cen @ n - d) < tol) & (nrm @ n > 0.9)
            w = float(area[inl].sum())
            if w > best[0]:
                best = (w, (n, d, inl))
        if best[1] is None or best[0] <= 0:
            break
        n, d, inl = best[1]
        if inl.sum() < floor:
            break
        # refit the plane to its inliers by area-weighted least squares
        c0 = (cen[inl] * area[inl, None]).sum(0) / area[inl].sum()
        u, s_, vt = np.linalg.svd((cen[inl] - c0) * np.sqrt(area[inl])[:, None],
                                  full_matrices=False)
        n = vt[-1] / np.linalg.norm(vt[-1])
        if n @ nrm[inl].mean(0) < 0:
            n = -n
        out.append((n, float(n @ c0), np.flatnonzero(keep)[inl]))
        remaining &= ~inl
    return out


def planar_snap(verts, faces, planes, misfit_fn=None, eta: float = None, tol: float = 0.02):
    """Project vertices onto accepted planes; a plane is accepted only on evidence.

    misfit_fn(verts) -> float is the data misfit of a candidate body. A plane is kept only
    if it does not raise that misfit past the calibrated floor eta. With no misfit_fn the
    snap is applied unconditionally, which is only appropriate in tests.
    """
    v = np.asarray(verts, dtype=np.float64).copy()
    base = None if misfit_fn is None else float(misfit_fn(v))
    accepted = 0
    for n, d, face_ids in planes:
        vid = np.unique(np.asarray(faces)[face_ids].reshape(-1))
        vid = vid[vid < len(v)]
        if len(vid) == 0:
            continue
        trial = v.copy()
        off = trial[vid] @ n - d
        near = np.abs(off) < tol
        trial[vid[near]] -= np.outer(off[near], n)
        if misfit_fn is None:
            v, accepted = trial, accepted + 1
            continue
        new = float(misfit_fn(trial))
        if new <= base + (eta if eta is not None else 0.0):
            v, base, accepted = trial, new, accepted + 1
    return v, accepted


def restore_constraints(verts: np.ndarray, radius: float, tol: float = 0.03) -> np.ndarray:
    """The LAST operation: z-extent equality, then the radius bound with its tolerance.

    The three public bodies measure r/R = 1.0079, 1.0271 and 0.9940, so two exceed their
    published R. The radius is treated as an approximation with tolerance `tol`.
    """
    v = np.asarray(verts, dtype=np.float64).copy()
    zmin, zmax = v[:, 2].min(), v[:, 2].max()
    v[:, 2] = 2.0 * (v[:, 2] - zmin) / max(zmax - zmin, 1e-12) - 1.0
    cap = radius * (1.0 + tol)
    r = float(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).max())
    if r > cap:
        v[:, :2] *= cap / r
    return v


def export_stl(path: str, verts: np.ndarray, faces: np.ndarray) -> dict:
    """Watertight binary STL with outward normals. Reports what it had to repair."""
    import trimesh
    m = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=True)
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    m.fix_normals()                      # consistent winding, outward
    report = {"watertight": bool(m.is_watertight), "volume": float(m.volume),
              "faces": int(len(m.faces))}
    if not m.is_watertight:
        m.fill_holes()
        report["filled_holes"] = True
        report["watertight"] = bool(m.is_watertight)
    m.export(path, file_type="stl")      # trimesh writes binary STL by default
    return report

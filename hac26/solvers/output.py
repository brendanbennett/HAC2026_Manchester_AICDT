"""From posterior samples to one submitted body.

The medoid rather than the mean. The samples are draws from a posterior that provably
contains indistinguishable pairs, so averaging them is not a summary but a new body that
none of them is. Concretely: a crater whose longitude is uncertain appears at a different
place in each sample, and the mean smears it into a shallow depression everywhere, which is
to say back into a filled convex body. Thresholding a marginal occupancy field does the same
thing by a different route. The metric medoid

    x* = argmin_s rank(-mean Dice_s) + rank(mean side-view ASSD_s)

is instead an actual sample -- the one most typical under the metrics being scored -- so it
keeps a crater somewhere rather than nowhere. If no side-view outlines are supplied, the same
function falls back to the old Dice-only medoid.

The planar snap helper is optional. It can help faceted/polyhedral targets, but it is wrong on
a genuinely smooth body, so the reconstruction script leaves it off unless requested. When it
is used, each candidate plane is accepted only if the data misfit does not rise beyond the
calibrated model-error floor eta.

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


def metric_medoid(occupancies, outlines=None, side_n_dirs: int = 36,
                  side_res: int = 512, side_mode: str = "side") -> int:
    """Index of the sample most central under the scored metrics, over the samples.

    The medoid is taken under the scored metrics themselves, not under a proxy, and it is
    computed sample against sample -- never against a truth -- so it transfers unchanged to
    the secret bodies.

    BOTH MEASURES, COMBINED BY RANK. The challenge scores each body twice, by volume overlap
    and by the distance between the boundary curves of its projections, and sums them: "the
    highest sum of all scores over the 7 secret asteroid models (minimum score 0, maximum
    score 14)". Selecting on volume alone optimises half the score.

    The two cannot simply be added: Dice is a similarity in [0, 1] and the boundary distance
    is an unbounded distance in model units, so any direct blend needs a scale constant, and
    a tuned constant is a prior nobody measured. Ranks avoid it. Each sample is ranked by
    mean Dice to the others and, separately, by mean boundary distance to the others; the
    winner minimises the sum of the two ranks. That weights the measures equally, which is
    what the challenge does, and introduces no free parameter.

    `outlines` is an optional list of surface-point arrays, one per sample, in the same order
    as `occupancies`. Without it this reduces to the volume-only medoid. The side-view
    settings are passed through to hac26.scoring.side_view.side_view_measure.
    """
    s = len(occupancies)
    if s == 1:
        return 0
    if outlines is not None and len(outlines) != s:
        raise ValueError(f"got {len(outlines)} outlines for {s} occupancy grids")

    mean_dice = np.zeros(s)
    for i in range(s):
        for j in range(i + 1, s):
            d = dice_volumes(occupancies[i], occupancies[j])
            mean_dice[i] += d
            mean_dice[j] += d
    mean_dice /= (s - 1)
    if outlines is None:
        return int(np.argmax(mean_dice))

    from hac26.scoring.side_view import side_view_measure
    mean_bd = np.zeros(s)
    for i in range(s):
        for j in range(i + 1, s):
            bd = side_view_measure(outlines[i], outlines[j], n_dirs=side_n_dirs,
                                   res=side_res, mode=side_mode)["assd_mean"]
            mean_bd[i] += bd
            mean_bd[j] += bd
    mean_bd /= (s - 1)

    # rank 0 = best under each measure; Dice high is good, boundary distance low is good
    r_dice = np.argsort(np.argsort(-mean_dice))
    r_bd = np.argsort(np.argsort(mean_bd))
    return int(np.argmin(r_dice + r_bd))


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

    The published radius is treated as an approximation with tolerance `tol`: two of the
    three public bodies exceed their own R when posed this way.
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

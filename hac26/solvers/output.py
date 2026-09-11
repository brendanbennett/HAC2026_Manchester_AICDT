"""From several reconstructed meshes to one submitted body.

The submitted body is never the mean of the samples. A feature whose position is uncertain,
such as a crater, sits at a different place in each sample; averaging the samples in code
space smears it into a shallow depression everywhere, which is a body none of the samples is.

It is chosen instead from a set of candidates, by which of them scores best on average
against the samples under the two measures the challenge scores. The candidates are the
samples themselves and the level sets of the fraction of samples occupying each voxel
(reconstruct_lpd.consensus_bodies). The two kinds are there because the two measures want
different bodies. Under voxel overlap the best single answer against an uncertain truth is a
level set, at the level that is half the overlap achievable; under the side-view boundary
distance a level set is penalised, because filling a concavity shortens the outline and it is
the concave stretches that no convex reconstruction can produce. Which wins is therefore
decided by measurement rather than by argument, and metric_medoid does that.

Ranks are added rather than scores. The challenge sums the two measures after normalising
each into [0, 1], but the normalisation of the boundary distance is not published, and a rank
is what survives any monotone choice of it. The cost is that margins are discarded: a
candidate can lose the voxel measure by a wide margin and win on a narrow lead in the other.

Without side-view outlines the choice is made under volume overlap alone.

Planar snapping is optional. It helps a faceted target and harms a smooth one, so the
reconstruction script leaves it off unless asked. When it runs, a candidate plane is accepted
only if the data misfit does not rise by more than the calibrated model error eta.

restore_constraints runs after the snap, because the snap moves vertices and would otherwise
break the z-extent equality and the radius bound.
"""
from __future__ import annotations

import numpy as np

__all__ = ["dice_volumes", "metric_medoid", "ransac_planes", "planar_snap",
           "restore_constraints", "export_stl"]


def dice_volumes(occ_a: np.ndarray, occ_b: np.ndarray) -> float:
    """Dice overlap of two boolean voxel grids."""
    inter = np.logical_and(occ_a, occ_b).sum()
    return float(2.0 * inter / max(occ_a.sum() + occ_b.sum(), 1))


def metric_medoid(occupancies, outlines=None, side_n_dirs: int = 36,
                  side_res: int = 512, side_mode: str = "side", n_ref=None) -> int:
    """Index of the candidate that scores best against the draws under the two measures the
    challenge scores.

    The challenge scores each body by volume overlap (Dice) and by the distance between the
    boundary curves of its side-view projections, and adds the two. The draws are samples of
    the bodies that fit the data, so a candidate's mean score against them estimates its
    expected score against the truth. Each candidate is ranked by its mean Dice to the draws
    and, separately, by its mean boundary distance to them, a draw never counted against
    itself; the winner minimises the sum of the two ranks. Ranks are used because Dice is a
    similarity in [0, 1] and the boundary distance is a length in model units, so adding them
    directly would need a scale constant nobody has measured; ranks weight the two equally
    with no free parameter. Everything is computed body against body, never against a truth,
    so the same rule applies to the secret bodies.

    `occupancies` are boolean voxel grids on one shared grid: the first `n_ref` are the
    draws, the rest further candidates (None: every entry is a draw, and the winner is the
    medoid of the draws). `outlines` is an optional list of surface-point arrays in the same
    order; without it the choice is by mean Dice alone. The side-view settings are passed to
    hac26.scoring.side_view.outline_set.
    """
    s = len(occupancies)
    n_ref = s if n_ref is None else int(n_ref)
    if s == 1 or n_ref < 1:
        return 0
    if outlines is not None and len(outlines) != s:
        raise ValueError(f"got {len(outlines)} outlines for {s} occupancy grids")

    def against_draws(pair):
        # mean over the draws of pair(candidate, draw), the candidate itself left out
        out = np.zeros(s)
        for i in range(s):
            others = [j for j in range(n_ref) if j != i]
            out[i] = np.mean([pair(i, j) for j in others]) if others else 0.0
        return out

    mean_dice = against_draws(lambda i, j: dice_volumes(occupancies[i], occupancies[j]))
    if outlines is None:
        return int(np.argmax(mean_dice))

    from hac26.scoring.side_view import measure_outlines, outline_extent, outline_set

    # Project each cloud once, on one shared extent: the contours come back in model units,
    # and a per-pair extent would put each pair on a different pixel pitch.
    ext = outline_extent(outlines, side_mode)
    sets = [outline_set(o, ext, n_dirs=side_n_dirs, res=side_res, mode=side_mode)
            for o in outlines]
    cache = {}

    def bd(i, j):
        key = (min(i, j), max(i, j))
        if key not in cache:
            cache[key] = measure_outlines(sets[key[0]], sets[key[1]])["assd_mean"]
        return cache[key]
    mean_bd = against_draws(bd)

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
    # The support a plane needs scales with the mesh: a fixed minimum face count would reject
    # every face of a coarsely meshed body.
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
    """Project vertices onto the accepted planes.

    `planes` come from ransac_planes. For each plane, the vertices of its faces that lie
    within `tol` of it are moved onto it. misfit_fn(verts) -> float is the data misfit of a
    candidate body; a plane is kept only if the misfit of the snapped mesh stays within `eta`
    of the misfit of the unsnapped mesh (`eta` None allows no rise). Without a misfit_fn every
    plane is applied, which is only appropriate in tests. Returns the new vertices and the
    number of planes accepted.
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
            # base is not advanced: eta bounds the total rise over the unsnapped mesh, not
            # the rise per plane.
            v, accepted = trial, accepted + 1
    return v, accepted


def restore_constraints(verts: np.ndarray, radius: float, tol: float = 0.03) -> np.ndarray:
    """Rescale z to span exactly [-1, 1], then cap the xy radius at radius * (1 + tol).

    The published radius is an approximation, so it is enforced with a tolerance rather than
    as a hard bound; see hac26.conventions.
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
    """Write a watertight binary STL with outward normals; returns a report of the repairs.

    Raises ValueError rather than writing a body that is not a closed solid of positive
    volume. An open or inside-out mesh is not an answer: the voxel measure is read off a
    parity scan, which a hole inverts over the whole column through it, and a negative
    volume means the winding is reversed. The repairs run in the order that can actually
    fix one -- holes are filled first, so that fix_normals sees a closed surface and
    orients the patches with everything else -- and the volume is measured after them
    rather than before, so the number in the report describes the body on disk.
    """
    import trimesh
    m = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=True)
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())   # a zero-area face carries no orientation
    report = {"watertight_as_built": bool(m.is_watertight)}
    if not m.is_watertight:
        m.fill_holes()
        report["filled_holes"] = True
    m.fix_normals()                      # consistent winding, outward
    report.update(watertight=bool(m.is_watertight), volume=float(m.volume),
                  faces=int(len(m.faces)))
    if not report["watertight"] or report["volume"] <= 0.0:
        raise ValueError(
            f"refusing to write {path}: not a closed solid after repair "
            f"(watertight={report['watertight']}, volume={report['volume']:.4g}, "
            f"faces={report['faces']}). A level set taken exactly on the k/n_draws "
            f"lattice is the usual cause; see reconstruct_lpd.off_lattice_level.")
    m.export(path, file_type="stl")      # trimesh writes binary STL by default
    return report

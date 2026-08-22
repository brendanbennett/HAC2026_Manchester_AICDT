"""Calibration against the real curves of models 1, 2 and 3, using their true STLs.

Fitted, and at what scope:

    shared, transfers to all ten   rho, source angular radius, PSF width, vignetting,
                                   OETF knots, clip knee
    per curve                      pedestal C, model error eta
    per body                       psi0

Held fixed rather than fitted, despite looking fittable: tau_I and tau_B (a hard threshold
passes no gradient), rho and the source angular radius (passed in on the command line), and
psi0 (never applied at all, so it is always zero). See scripts/calibrate.py.

A pedestal and not a gain. With val = g L + C the two reductions give I = g I_raw + C N and
N = N. Mean normalisation divides each curve by its own mean and so removes g entirely, which
makes a per-curve gain unidentifiable. C does not cancel, because it enters I weighted by the
pixel count, so the two channels together pin C/g.

The mount geometry and any z-gradient in the beam are pinned by measurement and never fitted.
Under any transport model the z -> -z mirror is exact for the elevation-0 geometries, so
fitting them would be fitting the mirror ambiguity itself.

Weights are computed in unnormalised space. Photon noise is roughly constant in absolute
terms, so on a mean-normalised curve it becomes large where the body is faint, and applying
1/sigma^2 there would drive the high-phase azimuths -- the most concavity-informative
geometries -- to zero weight. The whitening therefore uses s^2 = sigma^2 + eta^2 with eta the
fitted model error, which bounds the weight from above.

Radiosity runs on a decimated mesh out of necessity: a dense form-factor matrix on a full
ground-truth mesh is not storable. Interreflection is smooth and low-frequency,
unlike the silhouette, which is still rasterised from the full mesh.
"""
from __future__ import annotations

import numpy as np

from .conventions import source_directions, to_body
from hac26.forward.mesh.radiosity import RadiositySolver, emission, form_factors

__all__ = ["decimate", "light_visibility", "body_radiance", "residual_to_noise"]

N_CURVES = 56


def decimate(verts: np.ndarray, faces: np.ndarray, target: int = 1200,
             subdivide_up: bool = True):
    """Bring a mesh TO a facet budget -- decimating when above it, subdividing when below.

    Subdivision matters as well as decimation: a box has 12 facets, and a token carries one
    constant value across whatever pixels it covers, so a coarse mesh gives tokens that are
    large relative to the image. Subdivision is exact, adding vertices on existing faces
    without changing the geometry.
    """
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    if len(faces) > target:
        d = m.simplify_quadric_decimation(face_count=target)
        return np.asarray(d.vertices, dtype=np.float64), np.asarray(d.faces, dtype=np.int64)
    if subdivide_up:
        while len(m.faces) * 4 <= target:
            m = m.subdivide()
        if len(m.faces) < target // 2:
            m = m.subdivide()
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def light_visibility(verts, faces, centroids, normals, source_dirs, eps=1e-4):
    """V_i(omega_k): can facet i see source sample k?

    source_dirs (K, 3) -> (facets, K); (P, K, 3) -> (facets, P, K). Every ray goes in one
    cast either way -- the ray tracer is far happier with one large batch than with K small
    ones, and it is the same set of rays.
    """
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    d = np.asarray(source_dirs, dtype=float)
    tail = d.shape[:-1]
    o = np.repeat(centroids + normals * eps, int(np.prod(tail)), axis=0)
    dirs = np.tile(d.reshape(-1, 3), (len(centroids), 1))
    return (~m.ray.intersects_any(o, dirs)).astype(float).reshape(len(centroids), *tail)


def body_radiance(verts, faces, rho: float, delta_rad: float, psi: np.ndarray,
                  psi0: float = 0.0, n_source: int = 8, target_faces: int = 1200):
    """Per-facet radiance at every phase, on the decimated radiosity mesh.

    The form-factor matrix and its factorisation are built ONCE; each phase costs one
    back-substitution, which is why the solve is done in the body frame.
    """
    dv, df = decimate(verts, faces, target_faces)
    F_, area, nrm, cen = form_factors(dv, df, occlusion=True)
    solver = RadiositySolver(F_, rho=max(rho, 1e-9))
    s_dirs_lab = source_directions(delta_rad, n_source)
    # One right-hand-side block, not one solve per phase: the factorisation is shared.
    dirs = np.stack([np.stack([to_body(d, np.array([p]), psi0)[0] for d in s_dirs_lab])
                     for p in psi])                                   # (phases, K, 3)
    vis = light_visibility(dv, df, cen, nrm, dirs)                    # (facets, phases, K)
    e = emission(nrm, dirs, vis)                                      # (facets, phases)
    return dv, df, solver.radiance(solver.solve(e)).T


def residual_to_noise(pred: np.ndarray, real: np.ndarray, sigma: np.ndarray,
                      eta: np.ndarray | None = None) -> np.ndarray:
    """Per-geometry residual-to-noise. NEVER aggregated across geometries.

    Aggregating hides which geometries the model cannot reproduce.
    A single number is dominated by whichever geometries are brightest.
    """
    s = sigma if eta is None else np.sqrt(sigma ** 2 + eta ** 2)
    return np.sqrt(((pred - real) ** 2 / np.maximum(s, 1e-12) ** 2).mean(axis=-1))

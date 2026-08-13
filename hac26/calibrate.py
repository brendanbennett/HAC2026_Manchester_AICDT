"""M4 -- calibration against the real curves of models 1, 2 and 3, using their true STLs.

WHAT IS FITTED, AND AT WHAT SCOPE

  shared, once, transfers to all ten   rho, source angular radius delta, PSF width,
                                       vignetting coefficients, OETF knots, clip knee
  per curve (56)                       tau_I, tau_B, pedestal C
  per body                             psi0

WHY A PEDESTAL AND NOT A GAIN. With val = g L + C, the two reductions give
I = g I_raw + C N and N = N. Mean normalisation divides each curve by its own mean, which
removes g entirely -- a per-curve gain is unidentifiable and fitting one adds a direction
the data cannot constrain. C does not cancel, because it enters I weighted by the pixel
COUNT, so the intensity and binary channels together pin the ratio C/g. Keep the pedestal,
drop the gain.

WHAT IS PINNED BY MEASUREMENT AND NEVER FITTED. The mount geometry and any z-gradient in
the beam. Under any transport model the z -> -z mirror is exact for all fourteen
elevation-0 geometries, so those two are the only things that break the degeneracy; fitting
them means fitting the mirror ambiguity itself, and the fit will happily choose whichever
mirror suits the residual.

WEIGHTS ARE COMPUTED IN UNNORMALISED SPACE. Photon noise is roughly constant in absolute
terms, so on a mean-normalised curve it becomes sigma_c = sigma_photon / mean_c, i.e. LARGE
where the body is faint. Applying 1/sigma^2 to the normalised curves then drives azimuths
135 and 225 to zero weight -- and those are the two most concavity-informative geometries
in the set, being the ones at high phase where shadow dominates. The whitening therefore
uses s^2 = sigma^2 + eta^2 with eta the fitted residual model error, which bounds the
weight from above and keeps those geometries in the fit.

RADIOSITY RUNS ON A DECIMATED MESH, of necessity: model 1's ground truth carries 800,000
facets, and a dense form-factor matrix on that is 5,120 GB. Decimating to ~1,200 facets
costs 0.27% in volume and 0.93% in area, and interreflection is a smooth, low-frequency
quantity -- far smoother than the silhouette, which is still rasterised from the full mesh.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conventions import S_LAB, cameras, psi_grid, source_directions, to_body
from .radiosity import RadiositySolver, emission, facet_geometry, form_factors
from .sensor import SensorModel

__all__ = ["CalibrationParams", "decimate", "light_visibility", "body_radiance",
           "residual_to_noise"]

N_CURVES = 56


def decimate(verts: np.ndarray, faces: np.ndarray, target: int = 1200,
             subdivide_up: bool = True):
    """Bring a mesh TO a facet budget -- decimating when above it, subdividing when below.

    Subdividing up matters and was missing. A box decimates to 12 facets and stayed there,
    so its tokens were enormous relative to the image, and a token carries one constant
    value across whatever pixels it covers. Measured on the held-out set: the five worst
    shapes had 96, 320, 96, 320 and 96 tokens while all five best had 320, and the error
    distribution was heavy-tailed around a median of 0.040 rather than uniformly poor.
    Subdivision is exact -- it adds vertices on existing faces and changes no geometry.
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


class CalibrationParams(nn.Module):
    """Every fitted quantity, at its correct scope. Nothing here is per-curve gain."""

    def __init__(self, n_bodies: int = 3, n_curves: int = N_CURVES,
                 rho_prior: float = 0.85, rho_sd: float = 0.05):
        super().__init__()
        self.sensor = SensorModel()
        self.raw_rho = nn.Parameter(torch.tensor(float(np.log(rho_prior / (1 - rho_prior)))))
        self.raw_delta = nn.Parameter(torch.tensor(-4.0))       # source angular radius, rad
        self.raw_tau_i = nn.Parameter(torch.full((n_curves,), -2.0))
        self.raw_tau_b = nn.Parameter(torch.full((n_curves,), -1.0))
        self.pedestal = nn.Parameter(torch.zeros(n_curves))
        self.psi0 = nn.Parameter(torch.zeros(n_bodies))
        self.rho_prior, self.rho_sd = rho_prior, rho_sd

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_rho)

    @property
    def delta(self) -> torch.Tensor:
        return F.softplus(self.raw_delta)

    @property
    def tau_i(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_tau_i)

    @property
    def tau_b(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_tau_b)

    def rho_penalty(self) -> torch.Tensor:
        return ((self.rho - self.rho_prior) / self.rho_sd) ** 2


def light_visibility(verts, faces, centroids, normals, source_dirs, eps=1e-4):
    """V_i(omega_k): can facet i see source sample k? Ray cast per facet per sample."""
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    vis = np.ones((len(centroids), len(source_dirs)))
    o = centroids + normals * eps
    for k, d in enumerate(source_dirs):
        vis[:, k] = (~m.ray.intersects_any(o, np.tile(d, (len(o), 1)))).astype(float)
    return vis


def body_radiance(verts, faces, rho: float, delta_rad: float, psi: np.ndarray,
                  psi0: float = 0.0, n_source: int = 8, target_faces: int = 1200):
    """Per-facet radiance at every phase, on the decimated radiosity mesh.

    The form-factor matrix and its factorisation are built ONCE; each phase costs one
    back-substitution, which is the whole point of working in the body frame.
    """
    dv, df = decimate(verts, faces, target_faces)
    F_, area, nrm, cen = form_factors(dv, df, occlusion=True)
    solver = RadiositySolver(F_, rho=max(rho, 1e-9))
    s_dirs_lab = source_directions(delta_rad, n_source)
    out = np.zeros((len(psi), len(nrm)))
    for j, p in enumerate(psi):
        dirs = np.stack([to_body(d, np.array([p]), psi0)[0] for d in s_dirs_lab])
        vis = light_visibility(dv, df, cen, nrm, dirs)
        e = emission(nrm, dirs, vis)
        out[j] = solver.radiance(solver.solve(e))
    return dv, df, out


def residual_to_noise(pred: np.ndarray, real: np.ndarray, sigma: np.ndarray,
                      eta: np.ndarray | None = None) -> np.ndarray:
    """Per-geometry residual-to-noise. NEVER aggregated across geometries.

    Aggregating hides the thing worth knowing: which geometries the model cannot reproduce.
    A single number is dominated by whichever geometries are brightest.
    """
    s = sigma if eta is None else np.sqrt(sigma ** 2 + eta ** 2)
    return np.sqrt(((pred - real) ** 2 / np.maximum(s, 1e-12) ** 2).mean(axis=-1))

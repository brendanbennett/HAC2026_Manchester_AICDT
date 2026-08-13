"""Measurement geometry of HAC 2026 (all constants from the challenge page).

World frame: e3 = rotation axis (challenge z-axis). Light source at (-inf, 0, 0)
=> illumination direction (object -> source) OMEGA0 = -e1, constant (parallel beam).

Camera view direction (object -> camera), for azimuth theta, elevation eps,
handedness delta in {+1,-1} (to be fixed on public models; see data_io.fit_conventions):

    omega_c = R3(delta*theta) @ (-cos(eps) e1 + sin(eps) e3)
            = (-cos(eps)cos(theta), -delta cos(eps)sin(theta), sin(eps))

Check: theta=0, eps=0 gives omega_c = -e1 = OMEGA0 (beam-splitter view, phase angle 0).

Rotation state: psi(t) = sigma * 2*pi * k/m for frame k of an m-frame full revolution,
sense sigma in {+1,-1} (fixed on public models). Body frame == world frame at frame 0
(challenge convention: marked point faces the light source at the initial time).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# --- challenge constants (fips.fi HAC 2026 page) ---------------------------------
AZIMUTHS_DEG: tuple = (0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0)
TOP_ALPHA_DEG: dict = {0.0: 21.0, 45.0: 26.0, 90.0: 26.0, 135.0: 26.0,
                       225.0: 24.0, 270.0: 24.0, 315.0: 24.0}
CAM_KINDS: tuple = ("hor_a", "hor_b", "top", "bottom")  # column order within each azimuth group
CURVE_TYPES: tuple = ("intensity", "binary")
OMEGA0 = np.array([-1.0, 0.0, 0.0])
BLENDER_FRAMES = 360  # frames per revolution in the Blender-simulated curves


@dataclass(frozen=True)
class Camera:
    azimuth_deg: float
    elevation_deg: float
    kind: str  # one of CAM_KINDS

    def omega(self, delta: float = 1.0) -> np.ndarray:
        th = np.deg2rad(self.azimuth_deg)
        ep = np.deg2rad(self.elevation_deg)
        return np.array([-np.cos(ep) * np.cos(th),
                         -delta * np.cos(ep) * np.sin(th),
                         np.sin(ep)])

    @property
    def phase_angle_deg(self) -> float:
        """alpha_c = arccos<omega_c, OMEGA0> = arccos(cos eps cos theta); constant in time."""
        th = np.deg2rad(self.azimuth_deg)
        ep = np.deg2rad(self.elevation_deg)
        return float(np.rad2deg(np.arccos(np.clip(np.cos(ep) * np.cos(th), -1.0, 1.0))))


def build_cameras() -> list:
    """The 28 curves of one file, in the documented column order:
    for each azimuth in AZIMUTHS_DEG: (hor_a, hor_b, top, bottom)."""
    cams = []
    for az in AZIMUTHS_DEG:
        a = TOP_ALPHA_DEG[az]
        cams.append(Camera(az, 0.0, "hor_a"))
        cams.append(Camera(az, 0.0, "hor_b"))
        cams.append(Camera(az, +a, "top"))
        cams.append(Camera(az, -a, "bottom"))
    return cams


def r3(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def psi_grid(m: int, sigma: float = 1.0, psi0: float = 0.0) -> np.ndarray:
    """Rotation angles of the m frames of one full revolution.

    psi0 is a constant phase offset (frame 0 = aligned pose has psi0 = 0; nonzero
    values model residual start-phase misalignment, and are also used in tests to
    stay off the measure-zero set {mu0 = 0} where the discontinuous binary kernel
    makes float-level sign noise visible)."""
    return sigma * 2.0 * np.pi * np.arange(m) / m + psi0


def body_frame_dirs(omega_world: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """v_k = R3(-psi_k) @ omega_world, shape (m, 3).

    Identity used: <R3(psi) u, w> = <u, R3(-psi) w>, so photometric cosines of the
    rotating body against a fixed world direction w equal cosines of the *fixed*
    body normals against these rotated directions.
    """
    c, s = np.cos(psi), np.sin(psi)
    wx, wy, wz = omega_world
    return np.stack([c * wx + s * wy, -s * wx + c * wy, np.full_like(c, wz)], axis=1)


# --- EGI normal grid ---------------------------------------------------------------
@dataclass(frozen=True)
class NormalGrid:
    n_theta: int
    n_phi: int
    normals: np.ndarray   # (N, 3), N = n_theta * n_phi, row-major (theta major)
    theta: np.ndarray     # (n_theta,)
    phi: np.ndarray       # (n_phi,)

    @property
    def n(self) -> int:
        return self.n_theta * self.n_phi


def make_grid(n_theta: int = 24, n_phi: int = 48) -> NormalGrid:
    """Equirectangular grid of unit normals; cell centers avoid the exact poles."""
    theta = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    phi = np.arange(n_phi) * 2.0 * np.pi / n_phi
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    normals = np.stack([np.sin(tt) * np.cos(pp),
                        np.sin(tt) * np.sin(pp),
                        np.cos(tt)], axis=-1).reshape(-1, 3)
    return NormalGrid(n_theta, n_phi, normals, theta, phi)


def cell_index(grid: NormalGrid, u: np.ndarray) -> np.ndarray:
    """Flat grid index of the cell containing unit vector(s) u, shape (..., 3)."""
    u = np.asarray(u, dtype=float)
    th = np.arccos(np.clip(u[..., 2], -1.0, 1.0))
    ph = np.mod(np.arctan2(u[..., 1], u[..., 0]), 2.0 * np.pi)
    p = np.clip((th / np.pi * grid.n_theta).astype(int), 0, grid.n_theta - 1)
    q = (ph / (2.0 * np.pi) * grid.n_phi + 0.5).astype(int) % grid.n_phi
    return p * grid.n_phi + q


def project_closure(g: np.ndarray, normals: np.ndarray, iters: int = 200,
                    tol: float = 1e-12) -> np.ndarray:
    """Alternating projections onto {sum_i g_i u_i = 0} (affine) and {g >= 0}.

    Both sets are convex and their intersection C (the Minkowski-feasible cone)
    is nonempty; POCS converges to a point of C. Projection onto the affine set:
    g - U^T (U U^T)^{-1} U g with U = normals^T (3 x N).
    """
    U = normals.T                      # (3, N)
    M = U @ U.T                        # (3, 3)
    Minv = np.linalg.inv(M)
    g = np.clip(np.asarray(g, dtype=float).copy(), 0.0, None)
    for _ in range(iters):
        r = U @ g                      # (3,)
        if float(r @ r) <= tol * max(1.0, float(g @ g)):
            break
        g = g - U.T @ (Minv @ r)
        g = np.clip(g, 0.0, None)
    return g

"""M0 -- conventions. Frames, light, cameras, rotation, geometric constraints.

Everything downstream depends on these being right, so each one is stated with the check
that pins it, and `tests/test_conventions.py` asserts the checks.

FRAME. z is the rotation axis. We work in the BODY frame: the mesh never moves, and the
light and cameras rotate around it. That is what makes the radiosity form-factor matrix of
M2 phase-independent -- body, mount and turntable are mutually rigid in this frame.

LIGHT. The source is at (-inf, 0, 0), so the direction from the body toward it is

    s_lab = (-1, 0, 0)

with finite angular radius delta, represented as K = 8 directions on a disc about s_lab
each carrying E0/K.

CAMERAS. Tabulated azimuths are measured FROM THE LIGHT DIRECTION, so the lab azimuth is
phi_c = 180 deg + azimuth, and

    v_c = (cos e_c cos phi_c,  cos e_c sin phi_c,  sin e_c)

Expanding phi_c = 180 + a gives v_c = (-cos e cos a, -cos e sin a, sin e), hence

    cos alpha = v_c . s_lab = cos(e_c) cos(azimuth)

which is the identity the whole convention hangs on: azimuth 0 elevation 0 gives alpha = 0
(coaxial, which is why a beam splitter sits there), azimuth 135 elevation 26 gives
alpha = 129.4 deg, and azimuth 180 would give alpha = 180 deg -- the camera staring into
the beam, which is why that azimuth is absent from the table.

ROTATION. Directions are carried into the body frame by

    s_body(psi) = R_z(-psi - psi0) s_lab        v_body(psi) = R_z(-psi - psi0) v_c

psi0 is one fitted scalar per body (measured near -2 deg).

SENSE -- CORRECTED AGAINST THE DATA. The specification gives psi_k = +2 pi k / FRAMES. The
turntable runs the other way. Forward-modelling the three public STLs through this exact
convention and correlating against the real curves, each allowed its own best phase origin
so that only the SENSE is being compared:

    model      psi_k = +2 pi k/F      psi_k = -2 pi k/F
      1              0.2951                 0.5669
      2              0.2955                 0.7837
      3              0.2211                 0.8240

The negative sense wins on all three by 0.27 to 0.60 in correlation, which is far outside
anything a phase origin or a photometric detail could produce. It also agrees with the
rotation sense independently fitted on these same bodies earlier in the project. So SENSE
is -1 and psi_grid returns psi_k = -2 pi k / FRAMES; the R_z(-psi - psi0) formula above is
used exactly as written.
"""
from __future__ import annotations

import numpy as np

__all__ = ["S_LAB", "AZIMUTHS_DEG", "TOP_ELEVATION_DEG", "CAM_KINDS", "FRAMES",
           "Camera", "cameras", "camera_vector", "lab_azimuth_deg", "phase_angle_deg",
           "R_z", "to_body", "source_directions", "psi_grid", "SENSE"]

S_LAB = np.array([-1.0, 0.0, 0.0])

AZIMUTHS_DEG = (0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0)

# Top-camera elevation per azimuth; the "virtual bottom" camera sits at its negative.
TOP_ELEVATION_DEG = {0.0: 21.0, 45.0: 26.0, 90.0: 26.0, 135.0: 26.0,
                     225.0: 24.0, 270.0: 24.0, 315.0: 24.0}

# Column order within each azimuth group, matching the released curve files.
CAM_KINDS = ("hor_a", "hor_b", "top", "bottom")

FRAMES = 360

# Turntable sense, measured against the real curves -- see the module docstring.
SENSE = -1.0


def lab_azimuth_deg(azimuth_deg: float) -> float:
    """Tabulated azimuth is measured from the light; the lab frame is 180 deg away."""
    return 180.0 + azimuth_deg


def camera_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector from the body centre toward the camera."""
    phi = np.radians(lab_azimuth_deg(azimuth_deg))
    e = np.radians(elevation_deg)
    return np.array([np.cos(e) * np.cos(phi), np.cos(e) * np.sin(phi), np.sin(e)])


def phase_angle_deg(azimuth_deg: float, elevation_deg: float) -> float:
    """Solar phase angle: cos alpha = cos(elevation) cos(azimuth). Constant in time."""
    c = np.cos(np.radians(elevation_deg)) * np.cos(np.radians(azimuth_deg))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


class Camera:
    __slots__ = ("azimuth_deg", "elevation_deg", "kind")

    def __init__(self, azimuth_deg: float, elevation_deg: float, kind: str):
        self.azimuth_deg = float(azimuth_deg)
        self.elevation_deg = float(elevation_deg)
        self.kind = kind

    @property
    def v(self) -> np.ndarray:
        return camera_vector(self.azimuth_deg, self.elevation_deg)

    @property
    def phase_angle_deg(self) -> float:
        return phase_angle_deg(self.azimuth_deg, self.elevation_deg)

    def __repr__(self) -> str:
        return (f"Camera(az={self.azimuth_deg:g}, el={self.elevation_deg:g}, "
                f"{self.kind}, alpha={self.phase_angle_deg:.1f})")


def cameras() -> list:
    """The 28 geometries in released column order: per azimuth (hor_a, hor_b, top, bottom)."""
    out = []
    for az in AZIMUTHS_DEG:
        e = TOP_ELEVATION_DEG[az]
        out.append(Camera(az, 0.0, "hor_a"))
        out.append(Camera(az, 0.0, "hor_b"))
        out.append(Camera(az, +e, "top"))
        out.append(Camera(az, -e, "bottom"))
    return out


def R_z(angle_rad: float | np.ndarray) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def to_body(w: np.ndarray, psi: np.ndarray, psi0: float = 0.0) -> np.ndarray:
    """Carry a lab direction into the body frame at each phase: R_z(-psi - psi0) w.

    Vectorised over psi, returning (len(psi), 3). Written out rather than looping R_z so
    that the whole phase axis is one array operation.
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    a = -psi - psi0
    c, s = np.cos(a), np.sin(a)
    wx, wy, wz = w
    return np.stack([c * wx - s * wy, s * wx + c * wy, np.full_like(c, wz)], axis=1)


def source_directions(delta_rad: float, k: int = 8) -> np.ndarray:
    """K directions on a disc of angular radius delta about s_lab, each carrying E0/K.

    Points are placed on a single ring at the radius that makes the ring's mean solid-angle
    weight match the disc's, i.e. at delta/sqrt(2): for a uniform disc the mean squared
    offset is delta^2/2. One ring is enough because the penumbra term only needs the
    second moment of the source right, not its fine structure.
    """
    if delta_rad <= 0:
        return S_LAB[None, :].copy()
    r = delta_rad / np.sqrt(2.0)
    # basis orthogonal to s_lab
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])
    ang = 2.0 * np.pi * np.arange(k) / k
    d = (S_LAB[None, :] * np.cos(r)
         + np.sin(r) * (np.cos(ang)[:, None] * e1[None, :]
                        + np.sin(ang)[:, None] * e2[None, :]))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def psi_grid(frames: int = FRAMES, sense: float = SENSE) -> np.ndarray:
    """psi_k = sense * 2 pi k / frames, with sense = -1 measured (see module docstring)."""
    return sense * 2.0 * np.pi * np.arange(frames) / frames

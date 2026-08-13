"""M0 completion test.

The three phase angles are the gate: if they do not come out, the convention is wrong and
nothing built on it can be trusted. The remaining tests pin the pieces those three numbers
do not touch (rotation sense, source disc, column order), and two of them consult the real
data, because a convention is a claim about the instrument, not about our algebra.
"""
import numpy as np
import pytest

from hac26.conventions import (AZIMUTHS_DEG, FRAMES, S_LAB, SENSE, TOP_ELEVATION_DEG,
                               Camera, R_z, camera_vector, cameras, lab_azimuth_deg,
                               phase_angle_deg, psi_grid, source_directions, to_body)

DATA = "data/raw"


# ---------------------------------------------------------------- the gate

def test_three_phase_angles():
    """The check the specification says to stop on if it fails."""
    assert phase_angle_deg(0.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    # cos(26) cos(135) = -0.635450 -> 129.4603 deg. The specification quotes "129.4",
    # i.e. the exact value truncated to one decimal, so the tolerance must admit that.
    assert phase_angle_deg(135.0, 26.0) == pytest.approx(129.46, abs=0.01)
    assert phase_angle_deg(180.0, 0.0) == pytest.approx(180.0, abs=1e-9)


def test_phase_angle_identity_holds_for_every_geometry():
    """cos alpha = v_c . s_lab must equal cos(e) cos(azimuth) identically."""
    for cam in cameras():
        lhs = float(cam.v @ S_LAB)
        rhs = float(np.cos(np.radians(cam.elevation_deg))
                    * np.cos(np.radians(cam.azimuth_deg)))
        assert lhs == pytest.approx(rhs, abs=1e-12)


def test_azimuth_zero_is_coaxial():
    """At azimuth 0, elevation 0 the camera looks along the light: v_c == s_lab."""
    assert camera_vector(0.0, 0.0) == pytest.approx(S_LAB, abs=1e-12)


# ---------------------------------------------------------------- structure

def test_camera_table():
    cams = cameras()
    assert len(cams) == 28
    assert lab_azimuth_deg(0.0) == 180.0
    for i, az in enumerate(AZIMUTHS_DEG):
        block = cams[4 * i: 4 * i + 4]
        assert [c.kind for c in block] == ["hor_a", "hor_b", "top", "bottom"]
        assert block[0].elevation_deg == 0.0 and block[1].elevation_deg == 0.0
        assert block[2].elevation_deg == TOP_ELEVATION_DEG[az]
        assert block[3].elevation_deg == -TOP_ELEVATION_DEG[az]
    assert 180.0 not in AZIMUTHS_DEG          # would stare into the beam


def test_camera_vectors_are_unit():
    for cam in cameras():
        assert np.linalg.norm(cam.v) == pytest.approx(1.0, abs=1e-12)


def test_top_and_bottom_are_z_mirrors():
    """The z -> -z mirror maps each top camera onto its bottom counterpart exactly."""
    cams = cameras()
    for i in range(len(AZIMUTHS_DEG)):
        top, bot = cams[4 * i + 2].v, cams[4 * i + 3].v
        assert top * np.array([1, 1, -1]) == pytest.approx(bot, abs=1e-12)


# ---------------------------------------------------------------- rotation

def test_to_body_matches_explicit_rotation():
    psi = psi_grid(FRAMES)[[0, 1, 90, 359]]
    psi0 = np.radians(-2.0)
    got = to_body(S_LAB, psi, psi0)
    want = np.stack([R_z(-p - psi0) @ S_LAB for p in psi])
    assert got == pytest.approx(want, abs=1e-12)


def test_rotation_preserves_z_and_norm():
    v = camera_vector(45.0, 26.0)
    b = to_body(v, psi_grid(37))
    assert np.allclose(np.linalg.norm(b, axis=1), 1.0)
    assert np.allclose(b[:, 2], v[2])          # rotation about z cannot change z


def test_turntable_sense_is_the_measured_one():
    """Fixed by measurement, not by the specification -- see the conventions docstring.

    Kept synthetic and instant: the comparison against the real curves was run once to
    settle the sign and is recorded there; this only guards against the constant being
    flipped back.
    """
    assert SENSE == -1.0
    assert psi_grid(4)[1] < 0.0


def test_full_revolution_is_identity():
    v = camera_vector(90.0, 0.0)
    assert to_body(v, np.array([0.0]))[0] == pytest.approx(
        to_body(v, np.array([2 * np.pi]))[0], abs=1e-12)


def test_relative_geometry_is_phase_independent():
    """Rotating light and camera together cannot change the phase angle."""
    psi = psi_grid(24)
    for cam in cameras()[:8]:
        s, v = to_body(S_LAB, psi), to_body(cam.v, psi)
        assert np.allclose((s * v).sum(1), float(cam.v @ S_LAB), atol=1e-12)


# ---------------------------------------------------------------- source disc

def test_source_disc():
    d = source_directions(np.radians(1.5), k=8)
    assert d.shape == (8, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    # centred on s_lab, and within the stated angular radius
    assert d.mean(0) / np.linalg.norm(d.mean(0)) == pytest.approx(S_LAB, abs=1e-9)
    ang = np.degrees(np.arccos(np.clip(d @ S_LAB, -1, 1)))
    assert ang.max() <= 1.5 + 1e-9
    assert source_directions(0.0).shape == (1, 3)


# ---------------------------------------------------------------- against the data

@pytest.mark.parametrize("model", [1, 2, 3])
def test_published_radius_is_approximate_not_a_bound(model):
    """MEASURED, per the specification's instruction to check before assuming R is tight.

    It is neither tight nor an upper bound. Posed so that z spans exactly [-1, 1]:

        model 1   r_max 1.1288   R 1.120   r/R 1.0079
        model 2   r_max 1.4585   R 1.420   r/R 1.0271
        model 3   r_max 0.8747   R 0.880   r/R 0.9940

    Two of the three public bodies EXCEED their published R, by 0.8% and 2.7%, and the
    third falls 0.6% short. So R is a ~3% approximation, and enforcing r <= R as a hard
    constraint would shrink the true geometry of models 1 and 2. This test asserts what the
    data actually supports; any downstream use of R must respect the same tolerance.
    """
    import glob

    from hac26.shapes import rescale_touch_z
    from hac26.stl_io import load_stl
    from hac26.submission import CYLINDER_R

    f = glob.glob(f"{DATA}/AsteroidModel0{model}_shape_public/asteroid{model}.stl")
    if not f:
        pytest.skip("public STLs not present")
    v = rescale_touch_z(load_stl(f[0])[0])
    r = float(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).max())
    assert r == pytest.approx(CYLINDER_R[model], rel=0.03), (
        f"model {model}: r_max={r:.4f} vs published R={CYLINDER_R[model]}")
    assert abs(v[:, 2].min() + 1.0) < 1e-6 and abs(v[:, 2].max() - 1.0) < 1e-6


R_TOLERANCE = 0.03   # measured above; use this wherever R is enforced

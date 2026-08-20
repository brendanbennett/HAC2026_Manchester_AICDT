"""Tests for the non-convex shape library (`hac26.shape_library`, `hac26.library_metrics`,
`hac26.curves_mesh`).

Fast tests use small grids and few bodies; `slow` runs the actual scale the requirements are
stated at (a real library, hundreds of pairs) and is skippable with `-m "not slow"`.
"""
from __future__ import annotations

import numpy as np
import pytest

from hac26.conventions import cameras
from hac26.curves_mesh import convex_cross_check
from hac26.library_metrics import (check_body, check_library, descriptor_support,
                                   library_descriptors, occupancy, pairwise_dice,
                                   participation_ratio, principal_frame)
from hac26.shape_library import (Body, LibrarySpec, build_library, convexity_ratio,
                                 decimate_mesh, extract, hull_volume, is_edge_manifold,
                                 mesh_volume, n_components, pose, sample_body, sd_box,
                                 sd_ellipsoid, sd_sphere, op_smooth_union, op_subtract)
from hac26.shapes import hull_mesh, icosphere

FAST_SPEC = LibrarySpec(res=48)


# --------------------------------------------------------------------------- primitives

def test_sphere_field_is_a_true_sdf_near_the_surface():
    """|f(x)| should equal the distance to the surface for the simplest primitive."""
    f = sd_sphere(radius=1.0)
    pts = np.array([[2.0, 0, 0], [0, 0, 0], [1.0, 0, 0], [0.5, 0.5, 0.5]])
    got = f(pts)
    want = np.array([1.0, -1.0, 0.0, np.linalg.norm([0.5, 0.5, 0.5]) - 1.0])
    assert np.allclose(got, want, atol=1e-9)


def test_box_field_matches_known_distances():
    f = sd_box(half=(1.0, 1.0, 1.0))
    assert float(f(np.array([2.0, 0.0, 0.0]))) == pytest.approx(1.0)
    assert float(f(np.array([0.0, 0.0, 0.0]))) == pytest.approx(-1.0)
    assert float(f(np.array([1.0, 1.0, 1.0]))) == pytest.approx(0.0, abs=1e-9)


def test_subtract_removes_material_and_union_adds_it():
    a = sd_sphere(radius=1.0)
    b = sd_sphere(centre=(0.5, 0, 0), radius=0.6)
    cut = op_subtract(a, b)
    assert float(cut(np.array([0.5, 0.0, 0.0]))) > 0        # inside the bite: now outside
    assert float(cut(np.array([-0.9, 0.0, 0.0]))) < 0       # untouched region: still inside


def test_smooth_union_has_no_cusp_between_the_two_hard_mins():
    """The smoothed min must sit at or below the hard min everywhere (it can only add
    material at the join), and strictly below near the join itself."""
    a = sd_sphere(centre=(-0.5, 0, 0), radius=0.6)
    b = sd_sphere(centre=(0.5, 0, 0), radius=0.6)
    hard = lambda p: np.minimum(a(p), b(p))                  # noqa: E731
    soft = op_smooth_union(a, b, k=0.15)
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [-2.0, 0.0, 0.0]])
    assert np.all(soft(pts) <= hard(pts) + 1e-9)
    assert float(soft(np.array([0.0, 0.0, 0.0]))) < float(hard(np.array([0.0, 0.0, 0.0])))


# --------------------------------------------------------------------------- mesh measures

def test_mesh_volume_of_a_unit_cube_hull():
    v, f = icosphere(3)
    v, f = hull_mesh(v)
    vol = mesh_volume(v, f)
    assert vol == pytest.approx(4.0 / 3.0 * np.pi, rel=0.02)   # icosphere hull ~ unit ball


def test_convexity_ratio_is_one_for_a_convex_hull():
    v, f = icosphere(2)
    assert convexity_ratio(v, f) == pytest.approx(1.0, abs=1e-6)


def test_edge_manifold_true_for_icosphere_false_for_a_hole():
    v, f = icosphere(1)
    assert is_edge_manifold(f)
    assert not is_edge_manifold(f[1:])            # drop one triangle -> a boundary edge


def test_n_components_counts_two_disjoint_spheres():
    v1, f1 = icosphere(1)
    v2, f2 = icosphere(1)
    v2 = v2 + np.array([10.0, 0.0, 0.0])           # far apart: genuinely disjoint
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1)])
    assert n_components(v, f) == 2


def test_decimate_mesh_shrinks_face_count_and_preserves_occupancy():
    """decimate_mesh makes no manifoldness promise (vertex clustering can create
    non-manifold edges); what it must preserve is the occupancy a caller reads out of it."""
    v, f, _ = extract(sd_sphere(radius=1.0), extent=1.4, res=48)
    v2, f2 = decimate_mesh(v, f, extent=1.35, res=24)
    assert len(f2) < len(f) / 2
    occ_full = occupancy(v, f, res=24, extent=1.35, decimate=False)
    occ_dec = occupancy(v2, f2, res=24, extent=1.35, decimate=False)
    agree = (occ_full == occ_dec).mean()
    assert agree > 0.97


# --------------------------------------------------------------------------- extract + pose

def test_extract_of_a_sphere_field_is_closed_single_component_and_round():
    v, f, info = extract(sd_sphere(radius=1.0), extent=1.5, res=40)
    assert is_edge_manifold(f)
    assert n_components(v, f) == 1
    assert info["n_voids_filled"] == 0
    r = np.linalg.norm(v, axis=1)
    assert r.std() < 0.03                          # close to a perfect sphere


def test_extract_repairs_a_disconnected_field_to_one_component():
    """Two spheres far enough apart that the level set is two disjoint solids; extract must
    keep exactly one (the larger) rather than returning a two-component mesh."""
    f = lambda p: np.minimum(sd_sphere((-1.0, 0, 0), 0.3)(p),     # noqa: E731
                             sd_sphere((1.0, 0, 0), 0.15)(p))
    v, fc, info = extract(f, extent=1.6, res=48)
    assert info["n_solid_components"] == 2
    assert n_components(v, fc) == 1
    assert is_edge_manifold(fc)
    # the kept piece is the bigger sphere, centred near x = -1
    assert v[:, 0].mean() < 0


def test_extract_fills_an_interior_void():
    """A field that is a solid ball everywhere except a small positive (outside) bubble
    dead-centre: the bubble is an enclosed void with no path to the outer boundary, and the
    repair must remove its internal surface rather than leaving a two-shell mesh."""
    outer = sd_sphere(radius=1.0)

    def field_with_void(p):
        r = np.linalg.norm(p, axis=-1)
        return np.where(r < 0.3, 0.5, outer(p))

    v, f, info = extract(field_with_void, extent=1.5, res=40)
    assert info["n_voids_filled"] >= 1
    assert n_components(v, f) == 1
    assert is_edge_manifold(f)


def test_pose_touches_z_exactly_at_plus_and_minus_one_and_centres_xy():
    v, f = icosphere(2)
    v = v * np.array([1.3, 0.8, 2.1]) + np.array([5.0, -3.0, 1.0])   # off-centre, stretched
    p = pose(v, radius=1.0, faces=f)
    assert p[:, 2].min() == pytest.approx(-1.0, abs=1e-9)
    assert p[:, 2].max() == pytest.approx(1.0, abs=1e-9)
    assert abs(p[:, 0].mean()) < 0.35               # volumetric centroid, not vertex mean
    assert abs(p[:, 1].mean()) < 0.35
    assert np.hypot(p[:, 0], p[:, 1]).max() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- sample_body gate

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_sample_body_passes_every_requirement(seed):
    b = sample_body(np.random.default_rng([100, seed]), FAST_SPEC)
    chk = check_body(b, radius=FAST_SPEC.radius, convexity_max=FAST_SPEC.convexity_max)
    assert chk["non_convex"], chk
    assert chk["closed"], chk
    assert chk["single_component"], chk
    assert chk["z_span"], chk
    assert chk["inside_cylinder"], chk


def test_sample_body_is_deterministic_given_its_seed():
    b1 = sample_body(np.random.default_rng([42, 0]), FAST_SPEC)
    b2 = sample_body(np.random.default_rng([42, 0]), FAST_SPEC)
    assert b1.verts.shape == b2.verts.shape
    assert np.allclose(b1.verts, b2.verts)


def test_every_base_archetype_can_pass_the_gate():
    """Each base kind, run with no modifiers beyond what the gate loop adds, must be
    reachable -- i.e. no archetype is silently unusable."""
    spec = LibrarySpec(res=48, base_weights={"star_sh": 1.0}, n_modifiers=(0, 0))
    seen = set()
    for i in range(6):
        b = sample_body(np.random.default_rng([200, i]), spec)
        seen.add(b.recipe["base"])
    assert seen == {"star_sh"}


# --------------------------------------------------------------------------- library-level

@pytest.mark.slow
def test_library_diversity_clears_the_participation_ratio_baseline():
    """The stated baseline is PR ~ 3.7 on the current `shapes()` corpus; the new library
    must sit well above it on the solver's own descriptor (support on the design normals)."""
    lib = build_library(40, seed=7, spec=FAST_SPEC)
    desc = library_descriptors(lib, n_probes=120, res=32)
    assert participation_ratio(desc["support"]) > 6.0
    assert participation_ratio(desc["combined"]) > 8.0


@pytest.mark.slow
def test_library_pairwise_dice_is_varied_not_clustered_near_one():
    lib = build_library(24, seed=8, spec=FAST_SPEC)
    d = pairwise_dice(lib, res=32, max_pairs=120)
    assert d.mean() < 0.85          # not a library of near-copies
    assert d.std() > 0.03           # genuine spread, not a single cluster


@pytest.mark.slow
def test_library_check_passes_every_body():
    lib = build_library(30, seed=9, spec=FAST_SPEC)
    chk = check_library(lib, radius=FAST_SPEC.radius, convexity_max=FAST_SPEC.convexity_max)
    for key, idx in chk["failed"].items():
        assert idx == [], f"{key} failed for bodies {idx}"


def test_dice_of_a_body_against_itself_is_one():
    lib = build_library(2, seed=11, spec=FAST_SPEC)
    from hac26.library_metrics import dice
    assert dice(lib[0], lib[0], res=32) == pytest.approx(1.0, abs=1e-6)


def test_dice_is_invariant_to_a_rigid_rotation_of_one_body():
    """Intersection is not rotation-invariant, but Dice AFTER principal-axis alignment
    must be, which is the entire point of aligning first."""
    from hac26.library_metrics import dice
    b = sample_body(np.random.default_rng([300, 0]), FAST_SPEC)
    q, r = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))
    R = q * np.sign(np.diag(r))
    rotated = Body(b.verts @ R.T, b.faces, b.recipe, b.info)
    d = dice(b, rotated, res=32)
    assert d > 0.75                 # alignment recovers most of the overlap despite rotation


def test_principal_frame_orders_axes_by_decreasing_moment():
    v, f = icosphere(3)
    v = v * np.array([2.0, 1.0, 0.5])               # longest along x, shortest along z
    R = principal_frame(v, f, res=40, extent=2.5)
    aligned = v @ R.T
    spread = aligned.std(axis=0)
    assert spread[0] >= spread[1] >= spread[2]


# --------------------------------------------------------------------------- curve conventions

def test_mesh_curve_renderer_matches_the_exact_convex_operator():
    """For a convex body, ray-cast visibility and the mu>0-and-mu0>0 test must agree, so
    this is the correctness check for rotation sense, camera geometry and thresholds without
    needing real measured data."""
    u, f = icosphere(1)
    v = u * np.array([1.0, 0.7, 1.3])
    hv, hf = hull_mesh(v)
    geoms = cameras()[:4]
    res = convex_cross_check(hv, hf, m=10, geoms=geoms, res=44, c_lambert=0.12, delta=1.0)
    assert res["mean_abs_diff_normalised"] < 0.03


def test_mesh_curve_renderer_produces_a_time_directional_curve():
    """The renderer carries a fixed rotation SENSE (`hac26.conventions.SENSE`), not a
    parameter -- so the check is that an asymmetric body's curve is NOT invariant under
    reversing frame order, which it would be if the renderer had no definite time
    direction (e.g. if visibility only depended on |psi| somehow)."""
    from hac26.curves_mesh import render_curves_mesh
    from hac26.shape_library import sd_box, extract, pose as _pose

    f = op_subtract(sd_sphere(radius=1.0), sd_box((0.6, 0, 0), (0.35, 0.15, 0.15)))
    v, fc, _ = extract(f, extent=1.5, res=36)
    v = _pose(v, radius=1.0, faces=fc)
    geoms = cameras()[4:5]                          # az = 45 deg: not the symmetric az = 0
    c = render_curves_mesh(v, fc, m=10, curve_types=["binary"], geoms=geoms, res=28,
                           decimate_to=2000)
    assert not np.allclose(c, c[:, ::-1])

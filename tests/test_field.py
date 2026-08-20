"""Geometry test.

Encode a cube with h alone, tokens zero, extract, and
require the faces planar to within one grid cell and Dice above 0.99 against the analytic
cube. The rest pin the properties the cube test does not touch -- that the core is a plain
max rather than log-sum-exp, that the token field is signed and zero-mean, and that the
constraints are applied to vertices with the measured radius tolerance.
"""
import numpy as np
import pytest
import torch

from hac26.field import (CORE_SCALE, DESIGN_N, DESIGN_T, TOKEN_SIGMA_FRAC, ConvexCore,
                         ImplicitBody, TokenField, _design_residual, apply_constraints,
                         extract_mesh, spherical_design)

A = 1.0            # cube half-side
RES = 128


def cube_support(normals: np.ndarray, a: float = A) -> np.ndarray:
    """h(n) = max over the cube's vertices of n.v = a(|nx| + |ny| + |nz|)."""
    return a * np.abs(normals).sum(1)


# ------------------------------------------------------------------ the fixed normals

def test_design_is_a_ten_design():
    x = spherical_design()
    assert x.shape == (DESIGN_N, 3)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-9)
    # an exact design has zero energy at every degree; the Fibonacci spiral it starts from
    # sits at 1.5e-3, so this is three orders better and far inside anything downstream sees
    assert _design_residual(x, DESIGN_T) < 1e-5


def test_design_contains_the_axis_directions():
    """Required for a cube to be exactly representable by the convex core."""
    x = spherical_design()
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        assert np.abs(x - e).sum(1).min() < 1e-9
        assert np.abs(x + e).sum(1).min() < 1e-9


# ------------------------------------------------------------------ the core

def test_core_is_max_not_log_sum_exp():
    """A soft max would bias the zero set inward by log(J)/beta. Check exactness instead."""
    n = spherical_design()
    core = ConvexCore(n)
    core.set_support(torch.tensor(cube_support(n), dtype=torch.float32))
    y = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0], [0.0, 0.0, A]], dtype=torch.float32)
    f = core(y).detach().numpy()
    assert f[0] == pytest.approx(-A, abs=1e-5)       # centre: distance to the nearest face
    assert f[1] == pytest.approx(0.0, abs=1e-5)      # face centres sit exactly on the surface
    assert f[2] == pytest.approx(0.0, abs=1e-5)


def test_support_roundtrip():
    n = spherical_design()
    h = cube_support(n)
    core = ConvexCore(n)
    core.set_support(torch.tensor(h, dtype=torch.float32))
    assert core.h.detach().numpy() == pytest.approx(h, rel=1e-4)


def test_core_h_is_non_negative():
    core = ConvexCore(spherical_design())
    with torch.no_grad():
        core.raw_h.copy_(torch.full((DESIGN_N,), -50.0))
    assert (core.h >= 0).all()


# ------------------------------------------------------------------ the token field

def test_token_field_is_signed_and_zero_mean():
    torch.manual_seed(0)
    tf = TokenField(radius=1.0)
    for p in tf.parameters():
        with torch.no_grad():
            p.copy_(torch.randn_like(p) * 0.5)
    y = torch.randn(512, 3) * 0.6
    d = tf(y)
    assert abs(float(d.mean())) < 1e-5          # zero-mean by construction
    assert float(d.min()) < 0 < float(d.max())  # signed: it grows as well as carves


def test_token_sigma_is_fixed_not_learned():
    tf = TokenField(radius=2.0)
    assert tf.sigma == pytest.approx(TOKEN_SIGMA_FRAC * 2.0)
    assert not any(n.endswith("sigma") for n, _ in tf.named_parameters())


def test_core_scale_is_fixed():
    b = ImplicitBody(radius=2.0)
    assert b.s == pytest.approx(CORE_SCALE * 2.0)


# ------------------------------------------------------------------ constraints

def test_constraints_are_applied_to_vertices():
    v = np.array([[0.3, 0.0, -4.0], [0.0, 0.4, 6.0], [2.0, 0.0, 1.0]])
    out = apply_constraints(v, radius=1.0, tol=0.03)
    assert out[:, 2].min() == pytest.approx(-1.0, abs=1e-12)
    assert out[:, 2].max() == pytest.approx(+1.0, abs=1e-12)
    r = np.sqrt(out[:, 0] ** 2 + out[:, 1] ** 2).max()
    assert r == pytest.approx(1.03, rel=1e-9)      # clamped to R(1+tol), not to R


def test_radius_tolerance_does_not_shrink_a_body_inside_it():
    """A body at 1.02 R is left alone: two of the three public bodies genuinely exceed R."""
    v = np.array([[1.02, 0, -1.0], [0, 0, 1.0], [-1.02, 0, 0.0]])
    out = apply_constraints(v, radius=1.0, tol=0.03)
    assert np.sqrt(out[:, 0] ** 2 + out[:, 1] ** 2).max() == pytest.approx(1.02, rel=1e-9)


# ------------------------------------------------------------------ THE GATE

@pytest.mark.slow
def test_cube_extraction_is_planar_and_matches():
    """Encode a cube with h alone, tokens zero, extract at 128^3."""
    n = spherical_design()
    body = ImplicitBody(radius=A * np.sqrt(2.0), normals=n)
    body.core.set_support(torch.tensor(cube_support(n), dtype=torch.float32))

    extent = A * 1.6
    verts, faces = extract_mesh(lambda y: body(y, use_tokens=False), extent, res=RES)
    assert len(verts) > 0 and len(faces) > 0

    cell = 2.0 * extent / RES

    # planarity: every vertex of the cube's zero set must lie on one of the six faces,
    # i.e. its largest coordinate magnitude must be A, to within one grid cell
    dev = np.abs(np.abs(verts).max(1) - A)
    assert dev.max() < cell, f"max deviation {dev.max():.5f} exceeds one cell {cell:.5f}"

    # Dice against the analytic cube on a common voxel grid
    g = (np.arange(96) + 0.5) / 96 * 2 * extent - extent
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    truth = (np.abs(X) <= A) & (np.abs(Y) <= A) & (np.abs(Z) <= A)
    import trimesh
    got = trimesh.Trimesh(verts, faces, process=False).contains(
        np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)).reshape(truth.shape)
    dice = 2.0 * (got & truth).sum() / (got.sum() + truth.sum())
    assert dice > 0.99, f"Dice {dice:.4f}"

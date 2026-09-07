"""Tests of hac26.field, the implicit shape representation.

The gate is the cube: encode it with the support h alone, extract a mesh, and require the
faces planar to within one grid cell and Dice above 0.99 against the analytic cube. The other
tests pin what the cube does not touch: the design of normals, that the core is a plain max,
that the lattice correction is signed and independent of the query batch, that dh is
band-limited, and that the pose constraints allow the published radius its tolerance.
"""
import numpy as np
import pytest
import torch

from hac26.field import (DESIGN_N, DESIGN_T, LATTICE_EXTENT, N_SITES,
                         ConvexCore, GaussianLattice, ImplicitBody, _design_residual,
                         apply_constraints, extract_mesh, spherical_design)

A = 1.0            # cube half-side
RES = 128


def cube_support(normals: np.ndarray, a: float = A) -> np.ndarray:
    """h(n) = max over the cube's vertices of n.v = a(|nx| + |ny| + |nz|)."""
    return a * np.abs(normals).sum(1)


# ------------------------------------------------------------------ the fixed normals

def test_design_is_a_ten_design():
    """The cached design has DESIGN_N unit normals and a small worst-degree residual."""
    x = spherical_design()
    assert x.shape == (DESIGN_N, 3)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-9)
    # an exact design has zero residual at every degree
    assert _design_residual(x, DESIGN_T) < 1e-5


def test_design_contains_the_axis_directions():
    """All six axis directions are in the design; the core needs them to represent a cube
    exactly."""
    x = spherical_design()
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        assert np.abs(x - e).sum(1).min() < 1e-9
        assert np.abs(x + e).sum(1).min() < 1e-9


# ------------------------------------------------------------------ the core

def test_core_is_max_not_log_sum_exp():
    """With the cube's support, the core is exactly -A at the centre and exactly zero at the
    face centres; a log-sum-exp core would put the zero set inside the true surface."""
    n = spherical_design()
    core = ConvexCore(n)
    core.set_support(torch.tensor(cube_support(n), dtype=torch.float32))
    y = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0], [0.0, 0.0, A]], dtype=torch.float32)
    f = core(y).detach().numpy()
    assert f[0] == pytest.approx(-A, abs=1e-5)       # centre: distance to the nearest face
    assert f[1] == pytest.approx(0.0, abs=1e-5)      # face centres sit exactly on the surface
    assert f[2] == pytest.approx(0.0, abs=1e-5)


def test_support_roundtrip():
    """set_support followed by reading h returns the same support."""
    n = spherical_design()
    h = cube_support(n)
    core = ConvexCore(n)
    core.set_support(torch.tensor(h, dtype=torch.float32))
    assert core.h.detach().numpy() == pytest.approx(h, rel=1e-4)


def test_core_h_is_non_negative():
    """h stays non-negative whatever the raw parameter holds."""
    core = ConvexCore(spherical_design())
    with torch.no_grad():
        core.raw_h.copy_(torch.full((DESIGN_N,), -50.0))
    assert (core.h >= 0).all()


# ------------------------------------------------------------------ the correction

def test_correction_is_signed_and_adds_where_kernels_overlap():
    """Amplitudes on the fixed lattice add where their kernels overlap (doubling them doubles
    the field), and the field takes both signs."""
    gl = GaussianLattice()
    with torch.no_grad():
        gl.g.zero_()
        near = torch.cdist(gl.p, torch.zeros(1, 3))[:, 0].argsort()[:8]
        gl.g[near] = 0.1
    d_one = float(gl(torch.zeros(1, 3)))
    with torch.no_grad():
        gl.g[near] = 0.2
    assert float(gl(torch.zeros(1, 3))) == pytest.approx(2 * d_one, rel=1e-5)
    with torch.no_grad():
        gl.g[near[:4]] = -0.2
    d = gl(torch.randn(256, 3) * 0.5)
    assert float(d.min()) < 0 < float(d.max())          # signed: it grows as well as carves


def test_correction_does_not_depend_on_the_query_batch():
    """The correction is a function of the query point alone: evaluating the points in two
    chunks gives the same values as one call. extract_mesh evaluates its grid in chunks."""
    torch.manual_seed(0)
    gl = GaussianLattice()
    with torch.no_grad():
        gl.g.normal_(0, 0.1)
    y = torch.randn(3000, 3) * 0.5
    parts = torch.cat([gl(y[:2000]), gl(y[2000:])])
    assert float((gl(y) - parts).abs().max()) < 1e-6


def test_the_lattice_is_fixed_and_never_travels_with_a_checkpoint():
    """`g` is the only parameter and the only entry of the state dict; the sites and widths
    are constants of the representation, so a saved state cannot redefine another body's
    lattice."""
    gl = GaussianLattice()
    assert [n for n, _ in gl.named_parameters()] == ["g"]
    assert list(gl.state_dict().keys()) == ["g"]
    assert gl.g.numel() == N_SITES
    assert float(gl.p.abs().max()) < LATTICE_EXTENT          # cell centres, not corners


def test_dh_is_band_limited_whatever_the_flow_emits():
    """The expanded dh is band-limited to degree SH_DEGREE exactly in the argument of the
    softplus, and approximately in h itself (the softplus slope varies across normals), and
    the resulting support stays positive. An out-of-band dh would kill facets, and a dead
    facet has a zero row in the area Jacobian and so no gradient at all.
    """
    n = spherical_design(64)
    body = ImplicitBody(normals=n)
    body.set_support(torch.tensor(cube_support(n), dtype=torch.float32))
    from hac26.field import _real_sh
    y5 = torch.tensor(_real_sh(n, 5), dtype=torch.float32)
    torch.manual_seed(0)
    with torch.no_grad():
        body.dh.normal_(0, 0.02)
        arg = body.dh_expand @ body.dh                       # the argument: exactly band-limited
        moved = body.support() - body.core.h                 # h itself: approximately so
    r_arg = arg - y5 @ torch.linalg.lstsq(y5, arg).solution
    assert float(r_arg.norm() / arg.norm()) < 1e-4
    r_h = moved - y5 @ torch.linalg.lstsq(y5, moved).solution
    assert float(r_h.norm() / moved.norm()) < 0.10
    assert bool((body.support() > 0).all())                  # positivity is automatic


def test_every_parameter_block_receives_gradient():
    """Every parameter of ImplicitBody (the support, the lattice amplitudes and dh) receives
    a non-zero gradient from a loss on the field."""
    n = spherical_design(64)
    body = ImplicitBody(normals=n)
    body.set_support(torch.tensor(cube_support(n), dtype=torch.float32))
    (body(torch.randn(512, 3) * 0.5) ** 2).mean().backward()
    for name, prm in body.named_parameters():
        assert prm.grad is not None and float(prm.grad.abs().max()) > 0, f"{name} is dead"


# ------------------------------------------------------------------ constraints

def test_constraints_are_applied_to_vertices():
    """apply_constraints rescales z to [-1, 1] and caps the xy radius at R (1 + tol)."""
    v = np.array([[0.3, 0.0, -4.0], [0.0, 0.4, 6.0], [2.0, 0.0, 1.0]])
    out = apply_constraints(v, radius=1.0, tol=0.03)
    assert out[:, 2].min() == pytest.approx(-1.0, abs=1e-12)
    assert out[:, 2].max() == pytest.approx(+1.0, abs=1e-12)
    r = np.sqrt(out[:, 0] ** 2 + out[:, 1] ** 2).max()
    assert r == pytest.approx(1.03, rel=1e-9)      # clamped to R(1+tol), not to R


def test_radius_tolerance_does_not_shrink_a_body_inside_it():
    """A body whose xy radius is within the tolerance of R is left alone."""
    v = np.array([[1.02, 0, -1.0], [0, 0, 1.0], [-1.02, 0, 0.0]])
    out = apply_constraints(v, radius=1.0, tol=0.03)
    assert np.sqrt(out[:, 0] ** 2 + out[:, 1] ** 2).max() == pytest.approx(1.02, rel=1e-9)


# ------------------------------------------------------------------ the gate

@pytest.mark.slow
def test_cube_extraction_is_planar_and_matches():
    """A cube encoded with h alone extracts with every vertex on a face to within one grid
    cell, and Dice above 0.99 against the analytic cube."""
    n = spherical_design()
    body = ImplicitBody(normals=n)
    body.set_support(torch.tensor(cube_support(n), dtype=torch.float32))
    # g and dh are zero, so this is the core alone
    extent = LATTICE_EXTENT + 0.5
    verts, faces = extract_mesh(lambda y: body(y), extent, res=RES)
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
    # mesh_occupancy rather than trimesh.contains: contains() casts a ray per point and is
    # too expensive on a grid this size without the optional embreex dependency;
    # mesh_occupancy uses the same cell-centre grid and needs nothing optional.
    from hac26.recon import mesh_occupancy
    got = mesh_occupancy(verts, faces, 96, extent)
    dice = 2.0 * (got & truth).sum() / (got.sum() + truth.sum())
    assert dice > 0.99, f"Dice {dice:.4f}"

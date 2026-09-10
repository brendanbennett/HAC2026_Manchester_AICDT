"""The derivative-free fit: what a coordinate means, and that one step of it moves a body
toward the one its curves came from."""
import numpy as np
import pytest
import torch

from hac26.conventions import psi_grid
from hac26.field import (CODE_DIM, LATTICE_SHAPE, N_RADIAL, N_SITES, GaussianLattice,
                         ImplicitBody, lattice_kernel)
from hac26.forward.mesh.exact import RenderConfig
from hac26.forward.mesh.instrument import Instrument
from hac26.shapes import canonicalize_r, icosphere, mesh_support, rescale_touch_z
from hac26.solvers.gauss_newton import (CarveFit, Stage, block_basis, subspace_basis,
                                        waist_amplitudes)
from hac26.solvers.operator import CodeOperator

TINY = RenderConfig(height=24, width=40, supersample=1, sun_res=64, n_source=1,
                    phase_chunk=2, geom_chunk=1, radiosity_faces=48)


def test_the_sparse_lattice_kernel_is_the_field_at_the_sites():
    """K g is the field the amplitudes make at the sites, and it is what tells a solver what
    a coordinate is worth without spending a render on it."""
    K = lattice_kernel()
    lat = GaussianLattice()
    rng = np.random.default_rng(0)
    g = np.zeros(N_SITES)
    g[rng.choice(N_SITES, 12, replace=False)] = rng.standard_normal(12)
    dense = lat(lat.p, g=torch.tensor(g, dtype=torch.float32)).numpy()
    assert np.abs(K @ g - dense).max() < 1e-3       # the kernel is cut at 4 sigma, e^-8


def test_a_block_coordinate_is_a_carve_depth():
    """Every stage's coordinates are scaled so that one unit is one body unit of carve depth
    at the sites. Without it a coarse coordinate, which drives dozens of sites at once, would
    make a carve dozens of times deeper than the same number at the fine stage, and one
    finite-difference step could not be right for both."""
    K = lattice_kernel()
    fit = CarveFit(lambda c, g: None, np.zeros(4), np.ones(4), K, LATTICE_SHAPE)
    for stage in (Stage(2, 0, 1), Stage(6, 0, 1), Stage(12, 0, 1), Stage(0, 8, 1)):
        basis, btb = fit._basis(stage)
        depth = np.abs(K @ basis.toarray()).max(axis=0)
        assert np.allclose(btb, (basis.T @ basis).toarray())
        assert np.allclose(depth, 1.0, atol=1e-6), stage.name

    # the blocks partition the lattice: every site is driven by exactly one coordinate
    raw = block_basis(LATTICE_SHAPE, 6)
    assert raw.shape == (N_SITES, 216)
    assert np.array_equal(np.asarray(raw.sum(1)).ravel(), np.ones(N_SITES))
    q = subspace_basis(LATTICE_SHAPE, 8, np.random.default_rng(0))
    assert np.abs(q.T @ q - np.eye(8)).max() < 1e-6


def test_a_drawn_waist_carves_to_the_depth_it_asks_for():
    """A restart is a waist of a requested depth, and the amplitudes it turns into have to
    make a field of about that depth, or the start is somewhere else than intended."""
    K = lattice_kernel()
    sites = GaussianLattice().p.numpy()
    rng = np.random.default_rng(3)
    for _ in range(8):
        g, rec = waist_amplitudes(sites, K, rng)
        made = float(np.abs(K @ g).max())
        assert 0.5 * rec["depth"] < made < 1.5 * rec["depth"]


def _operator_and_bodies():
    """A small operator, a convex support, and a body carved out of it by a known waist."""
    v, f = icosphere(2)
    v = np.asarray(v) * np.array([1.0, 0.82, 0.72])
    v = canonicalize_r(rescale_touch_z(v, f, centre_xy=False))
    support = torch.tensor(mesh_support(v, ImplicitBody().core.n.numpy()),
                           dtype=torch.float32)
    op = CodeOperator(Instrument.blender_start(), psi_grid(2), res=16, config=TINY,
                      device="cpu", backend="software")
    K = lattice_kernel()
    sites = GaussianLattice().p.numpy()
    # a waist across the spin axis, the feature a convex inversion cannot see
    target = 0.35 * np.exp(-((sites @ np.array([1.0, 0.0, 0.0])) / 0.30) ** 2)
    g_true = target / float(np.asarray(K.sum(1)).ravel().max())
    c_true = np.zeros(N_RADIAL)
    c_true[0] = 0.06                                  # and a hull that is slightly too large
    return op, support, K, g_true, c_true


@pytest.mark.slow
def test_one_step_moves_a_body_toward_the_one_its_curves_came_from():
    """The fit is given the curves of a carved body and started from the convex body it was
    carved out of. A step has to lower the misfit and move the amplitudes toward the ones
    that made the curves; a step that only lowers the misfit could be fitting the curves with
    the wrong shape, which is what every method that moved the carve alone did."""
    op, support, K, g_true, c_true = _operator_and_bodies()
    geoms = [12]                                     # one high phase-angle camera
    code = torch.zeros(CODE_DIM)

    def render(c, g):
        z = code.clone()
        z[-N_SITES:] = torch.tensor(np.asarray(g), dtype=torch.float32)
        cur = op.curves(support, z, 1.0, geoms=geoms,
                        c=torch.tensor(np.asarray(c), dtype=torch.float32))
        return None if cur is None else cur.numpy().ravel()

    data = render(c_true, g_true)
    assert data is not None
    scale = np.full(len(data), 0.01)
    fit = CarveFit(render, data, scale, K, LATTICE_SHAPE, n_radial=N_RADIAL,
                   step_g=0.20, step_c=0.03, seed=0)

    start = fit.residual(render(np.zeros(N_RADIAL), np.zeros(N_SITES)))
    chi0 = float(np.linalg.norm(start))
    c, g, hist = fit.run(np.zeros(N_RADIAL), np.zeros(N_SITES),
                         stages=(Stage(2, 0, 2),), target=0.0)
    assert hist and hist[0]["accepted"], "no damping and no step length lowered the misfit"
    chi1 = hist[-1]["chi"]
    assert chi1 < chi0

    # and the body moved toward the truth, not merely toward the curves
    field_true, field_fit = K @ g_true, K @ g
    assert float(field_fit @ field_true) > 0.0
    assert np.linalg.norm(field_fit - field_true) < np.linalg.norm(field_true)
    assert abs(c[0] - c_true[0]) < abs(c_true[0])

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
from hac26.solvers.gauss_newton import (AREA_WEIGHT, AREA_WINDOW, DEPTH_TRUST,
                                        VOLUME_TRUST, CarveFit, Stage,
                                        block_basis, subspace_basis,
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
    # a waist across the spin axis, the feature a convex inversion cannot see, at a depth
    # measured through the kernel rather than estimated from its overlap, which is what
    # waist_amplitudes does and what makes the depth the same at any lattice size
    target = np.exp(-((sites @ np.array([1.0, 0.0, 0.0])) / 0.30) ** 2)
    g_true = target * (0.35 / float(np.abs(K @ target).max()))
    c_true = np.zeros(N_RADIAL)
    # and a hull too large by about what a convex inversion of a body with a waist leaves
    c_true[0] = 0.10
    return op, support, K, g_true, c_true


@pytest.mark.slow
def test_one_step_moves_a_body_and_leaves_it_a_body():
    """The fit is given the curves of a carved body and started from the convex body it was
    carved out of. One step of the coarsest stage has to lower what is being minimised, and
    the body it leaves has to still be a body: its volume inside the trust region and the
    carve it added no deeper than a carve can be.

    Those are the properties of the machinery and they are what a test can settle. Whether the
    fit moves a body *toward* its truth is a property of a run against that body's whole set
    of geometries, and is measured in notes/representation.md; asked of one coarse stage
    against a handful of cameras it measures a regime no reconstruction is in, where a
    relative change of misfit is small and the penalty sets the step on its own.
    """
    op, support, K, g_true, c_true = _operator_and_bodies()
    geoms = [12]                                     # one high phase-angle camera
    code = torch.zeros(CODE_DIM)

    def render(c, g):
        z = code.clone()
        z[-N_SITES:] = torch.tensor(np.asarray(g), dtype=torch.float32)
        out = op.curves_with_shape(support, z, 1.0, geoms=geoms,
                                   c=torch.tensor(np.asarray(c), dtype=torch.float32))
        if out is None:
            return None
        cur, area, vol = out
        return cur.numpy().ravel(), area, vol

    data = render(c_true, g_true)[0]
    # The model error the fit is given has to be the one it is used with. What is minimised is
    # scale free in the misfit, so a body a hundred model errors from its curves is a regime
    # no reconstruction starts in; a calibrated model error puts the convex answer a few
    # errors away and this scale does the same here.
    scale = np.full(len(data), 0.12)
    fit = CarveFit(render, data, scale, K, LATTICE_SHAPE, n_radial=N_RADIAL, seed=0)

    r0, area0, vol0 = fit._render(np.zeros(N_RADIAL), np.zeros(N_SITES))
    obj0 = fit.objective(r0, area0)
    c, g, hist = fit.run(np.zeros(N_RADIAL), np.zeros(N_SITES),
                         stages=(Stage(2, 0, 2),), target=0.0)
    assert hist and hist[0]["accepted"], "no damping and no step length improved the objective"
    assert hist[-1]["objective"] < obj0
    assert abs(hist[-1]["volume"] - vol0) < 3.0 * VOLUME_TRUST * vol0   # three steps of it
    assert float(np.abs(K @ g).max()) < 3.0 * DEPTH_TRUST
    assert np.isfinite(g).all() and np.isfinite(c).all()


def test_the_objective_charges_surface_and_not_amplitude():
    """A corrugation and a smooth dent of the same depth are the same size in the amplitudes
    and very different in area, and it is area the objective charges.

    That is the whole reason the penalty is on area: a ridge on the amplitudes would rank
    these two the same way round only by accident, and the misfit ranks the corrugation
    first, because a finely resolved surface fits rendered curves better whether or not its
    shape is right."""
    K = lattice_kernel()
    fit = CarveFit(lambda c, g: None, np.zeros(4), np.ones(4), K, LATTICE_SHAPE,
                   area_weight=AREA_WEIGHT)
    r = np.full(4, 0.5)
    smooth, rough = 9.0, 9.6
    assert fit.objective(r, smooth) < fit.objective(r, rough)
    # and the balance is scale free: a body whose misfit is ten times smaller is not thereby
    # allowed ten times the surface
    better = np.full(4, 0.05)
    assert (fit.objective(better, rough) - fit.objective(better, smooth)
            == pytest.approx(fit.objective(r, rough) - fit.objective(r, smooth)))
    # with the penalty off it is the misfit alone
    off = CarveFit(lambda c, g: None, np.zeros(4), np.ones(4), K, LATTICE_SHAPE,
                   area_weight=0.0)
    assert off.objective(r, smooth) == off.objective(r, rough)


def test_the_area_weight_is_held_inside_the_window_it_was_measured_in():
    """Outside it the objective is the wrong one in a way no run would report: below the
    window a corrugation one grid cell wide still lowers it, and above the window the body
    stops being the minimum."""
    K = lattice_kernel()
    for bad in (AREA_WINDOW[0] - 0.1, AREA_WINDOW[1] + 0.1):
        with pytest.raises(ValueError):
            CarveFit(lambda c, g: None, np.zeros(4), np.ones(4), K, LATTICE_SHAPE,
                     area_weight=bad)


def test_a_step_that_moves_the_volume_too_far_is_refused():
    """The cheapest area in this representation is a hull shrink, so an objective that
    charges area walks the body away to nothing unless the volume is held. The trust region
    is what holds it, and a fit whose every trial leaves it must take no step at all."""
    K = lattice_kernel()
    calls = {"n": 0}

    def render(c, g):
        calls["n"] += 1
        # every body after the first is half the volume of the first, and fits perfectly
        if calls["n"] == 1:
            return np.ones(4), 9.0, 1.0
        return np.zeros(4), 1.0, 0.5

    fit = CarveFit(render, np.zeros(4), np.ones(4), K, LATTICE_SHAPE, n_radial=N_RADIAL,
                   volume_trust=0.08)
    _, _, hist = fit.run(np.zeros(N_RADIAL), np.zeros(N_SITES), stages=(Stage(2, 0, 1),),
                         target=0.0)
    assert hist and not hist[0]["accepted"], "a step halving the volume was accepted"


def test_the_search_is_spent_only_where_it_can_move_the_surface():
    """A secant Jacobian costs one render per coordinate, so a coordinate over sites that
    cannot move the surface is a render thrown away. Over half the lattice of a body of this
    size is in the corners of the box, outside the body the fit starts from.

    The band has to be one-sided. The sites a carve needs run from the starting surface all
    the way inward, so a band about the surface throws the carve away; what is null is the
    outside. The mask therefore keeps everything inside the start and a shell beyond it, and
    the coordinates it leaves still carry unit peak depth, or one finite-difference step would
    no longer be right for all of them."""
    from hac26.field import searchable_sites
    from hac26.shapes import canonicalize_r, rescale_touch_z

    v, f = icosphere(2)
    v = canonicalize_r(rescale_touch_z(np.asarray(v) * np.array([1.0, 0.82, 0.72]),
                                       np.asarray(f), centre_xy=False))
    mask = searchable_sites(mesh_support(v, ImplicitBody().core.n.numpy()))
    assert 0.2 * N_SITES < mask.sum() < 0.8 * N_SITES, int(mask.sum())

    # every site the body's own surface passes through is kept
    sites = GaussianLattice().p.numpy()
    core = (sites @ ImplicitBody().core.n.numpy().T
            - mesh_support(v, ImplicitBody().core.n.numpy())[None, :]).max(axis=1)
    assert mask[core <= 0.0].all(), "a site inside the starting body was masked out"

    K = lattice_kernel()
    full = CarveFit(lambda c, g: None, np.zeros(4), np.ones(4), K, LATTICE_SHAPE)
    lean = CarveFit(lambda c, g: None, np.zeros(4), np.ones(4), K, LATTICE_SHAPE,
                    site_mask=mask)
    for stage in (Stage(6, 0, 1), Stage(12, 0, 1)):
        b_full, _ = full._basis(stage)
        b_lean, btb = lean._basis(stage)
        assert b_lean.shape[1] < b_full.shape[1], stage.name
        depth = np.abs(K @ b_lean.toarray()).max(axis=0)
        assert np.allclose(depth, 1.0, atol=1e-6), stage.name
        assert np.allclose(btb, (b_lean.T @ b_lean).toarray())
        # and nothing the mask refuses is driven
        assert not b_lean.toarray()[~mask].any()

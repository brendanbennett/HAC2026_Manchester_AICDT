"""The fit of the lattice amplitudes to a body: it has to reach whatever depth the body has,
not whatever depth a step budget allows.

The estimator is what is under test, and it does not depend on how many sites the lattice
has, so these run on a small one. The production lattice has thousands of sites and the
solve is a dense system in that many unknowns, which needs one sample point per twelve of
them to be determined at all; carrying that here would test the machine it runs on."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hac26.field import EXTRACT_EXTENT, ImplicitBody, extract_mesh          # noqa: E402
from hac26.recon import dice, mesh_occupancy                                # noqa: E402
from fit_shapes import DICE_EXTENT, LatticeFit, POINTS_PER_SITE, sample_arrays  # noqa: E402

SHAPE = (8, 8, 8)
N_PTS = POINTS_PER_SITE * 8 ** 3


def _dumbbell(res=96, extent=1.2, r=0.62, half=0.42):
    """Two overlapping spheres, as the level set of the smaller of their two distances: a
    body with a waist no convex body has."""
    from skimage import measure
    ax = np.linspace(-extent, extent, res)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    d = np.minimum(np.sqrt((x + half) ** 2 + y ** 2 + z ** 2),
                   np.sqrt((x - half) ** 2 + y ** 2 + z ** 2)) - r
    v, f, _, _ = measure.marching_cubes(d, level=0.0, spacing=(ax[1] - ax[0],) * 3)
    return v - extent, np.asarray(f, dtype=np.int64)


def _fit(verts, faces, n_pts=N_PTS, res=48):
    fit = LatticeFit("cpu", lattice_shape=SHAPE)
    normals = ImplicitBody().core.n.numpy()
    pts, sd = sample_arrays(verts, faces, n_pts=n_pts, seed=0)
    h = np.maximum((verts @ normals.T).max(axis=0), 1e-3).astype(np.float32)
    g, before, after = fit.solve(torch.tensor(pts), torch.tensor(sd), h)
    body = ImplicitBody(lattice_shape=SHAPE)
    body.core.set_support(torch.tensor(h))
    with torch.no_grad():
        body.delta.g.copy_(g)
    v, f = extract_mesh(lambda y: body(y), EXTRACT_EXTENT, res=res, device="cpu")
    d = dice(mesh_occupancy(v, f, res, DICE_EXTENT),
             mesh_occupancy(verts, faces, res, DICE_EXTENT)) if len(f) >= 8 else 0.0
    return g, before, after, d


def test_the_fit_reproduces_a_body_with_a_waist():
    """The amplitudes of a two-lobed body are solved, not descended to, so the waist comes
    back rather than being averaged away, and the residual against the signed distance falls
    by a large factor."""
    v, f = _dumbbell()
    g, before, after, d = _fit(v, f)
    assert d > 0.9, d
    assert after < 0.2 * before, (before, after)
    # the hull of this body bridges the waist, so carving it needs real amplitudes
    assert float(g.abs().max()) > 0.02


def test_a_convex_body_needs_almost_no_amplitudes():
    """The core alone is already the answer for a convex body, so the fit leaves the
    amplitudes near zero and does not carve something that is not there."""
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=2, radius=0.8)
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    g, before, after, d = _fit(v, f)
    assert d > 0.95, d
    assert float(g.abs().max()) < 0.02, float(g.abs().max())


def test_blocking_the_solve_does_not_change_it():
    """The normal matrix is a sum over the sample points, so accumulating it a block at a time
    must give the same amplitudes and the same residual."""
    import fit_shapes
    v, f = _dumbbell()
    fit = LatticeFit("cpu", lattice_shape=SHAPE)
    normals = ImplicitBody().core.n.numpy()
    pts, sd = sample_arrays(v, f, n_pts=N_PTS, seed=0)
    h = np.maximum((v @ normals.T).max(axis=0), 1e-3).astype(np.float32)
    whole = fit_shapes.SOLVE_CHUNK
    try:
        fit_shapes.SOLVE_CHUNK = 10 ** 9
        g1, _, a1 = fit.solve(torch.tensor(pts), torch.tensor(sd), h)
        fit_shapes.SOLVE_CHUNK = 700
        g2, _, a2 = fit.solve(torch.tensor(pts), torch.tensor(sd), h)
    finally:
        fit_shapes.SOLVE_CHUNK = whole
    assert abs(a1 - a2) < 1e-4 * max(a1, 1e-9)
    assert float((g1 - g2).abs().max()) < 0.02 * float(g1.abs().max())

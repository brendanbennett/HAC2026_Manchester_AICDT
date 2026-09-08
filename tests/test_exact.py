"""The exact forward model on the software rasteriser: coverage, shadows, agreement with the
convex operator on a convex body, and the adjoint plumbing."""
import numpy as np
import pytest
import torch

from hac26.conventions import SENSE, cameras, psi_grid
from hac26.forward.mesh.exact import (ExactForward, LitCoverage, RenderConfig, normalise,
                                      normalise_vjp)
from hac26.forward.mesh.instrument import Instrument
from hac26.forward.mesh.raster import Rasteriser, flat_faces
from hac26.geometry import build_cameras
from hac26.shapes import hull_mesh, icosphere, mesh_curves_convex

# few patches, so the interreflection runs on a decimated copy of the mesh as it does on a
# real body
SMALL = RenderConfig(height=24, width=40, supersample=1, sun_res=96, phase_chunk=3,
                     radiosity_faces=48)
# fine enough that the missing antialiasing of the software backend is a small effect
MEDIUM = RenderConfig(height=48, width=80, supersample=2, sun_res=96, phase_chunk=4,
                      radiosity_faces=48)


def _two_spheres():
    import trimesh
    a = trimesh.creation.icosphere(subdivisions=1, radius=0.55)
    b = trimesh.creation.icosphere(subdivisions=1, radius=0.55)
    a.apply_translation([-0.375, 0, 0]); b.apply_translation([0.375, 0, 0])
    m = trimesh.util.concatenate([a, b])
    return np.asarray(m.vertices), np.asarray(m.faces)


def _coverage(v, f, direction, res=400):
    vt = torch.tensor(v, dtype=torch.float32); ft = torch.tensor(f)
    fv, ff = flat_faces(vt, ft)
    ras = Rasteriser(res, res, 1, device="cpu", backend="software")
    d = torch.tensor([direction], dtype=torch.float32); d = d / d.norm()
    ext = float(vt.norm(dim=1).max()) * 1.05
    cov = LitCoverage.apply(fv, ff, d, ext, ext, ras)[0].numpy()
    tv = v[f]
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    cos = (n / (2 * area[:, None])) @ (d[0].numpy())
    return cov, area * np.clip(cos, 0, None)


def test_coverage_of_a_convex_body_is_its_projected_area():
    """A convex body casts no shadow on itself, so the lit projected area of every face is
    A cos(theta)+, and their sum is the area of the silhouette."""
    v, f = icosphere(2)
    cov, proj = _coverage(v, f, [0.6, 0.5, 0.3])
    assert np.abs(cov - proj).max() < 0.02 * proj.max()
    assert cov.sum() == pytest.approx(proj.sum(), rel=0.005)


def test_coverage_never_exceeds_the_projected_area_and_shadows_reduce_it():
    """On two overlapping spheres the lit area of each face is at most A cos(theta)+, and the
    faces in the cast shadow of the other sphere get much less."""
    v, f = _two_spheres()
    cov, proj = _coverage(v, f, [1.0, 0.1, 0.0])
    assert (cov <= proj + 1e-3).all()
    shadowed = proj - cov > 0.5 * proj
    assert shadowed.sum() >= 5
    assert cov.sum() < 0.98 * proj.sum()


def test_exact_intensity_matches_the_convex_operator_on_a_convex_body():
    """A convex body has no interreflection (no two faces see each other), so the exact chain
    reduces to Lambert scattering and its intensity curves must agree with the analytic convex
    operator's Lambert curves, which checks the rotation sense, the camera geometry and the
    projection end to end. The threshold is set very low so it does not cut dim faces."""
    u, f = icosphere(1)
    v = u * np.array([1.0, 0.7, 1.3])
    hv, hf = hull_mesh(v)
    P, geoms = 8, [0, 2, 9, 15]
    inst = Instrument(tau_i=1e-3, quantise=False)
    op = ExactForward(inst, psi_grid(P), MEDIUM, device="cpu", backend="software")
    raw = op.raw_curves(torch.tensor(hv, dtype=torch.float32), torch.tensor(hf), geoms=geoms)
    mine = normalise(raw)[:, 0].numpy()                                   # intensity only
    ref_cams = [build_cameras()[g] for g in geoms]
    ref = mesh_curves_convex(hv, hf, ref_cams, P, ["intensity"] * len(geoms),
                             c_lambert=1.0, ls_weight=0.0, sigma=SENSE)
    ref = ref / ref.mean(1, keepdims=True)
    assert np.abs(mine - ref).mean() < 0.05


def test_vjp_returns_the_same_curves_and_finite_gradients():
    """The vector-Jacobian product reproduces raw_curves exactly and gives finite gradients for
    the vertices and the instrument parameters; more albedo means more light, so the
    derivative of the total intensity with respect to rho is positive on a body that reflects
    onto itself."""
    v, f = _two_spheres()
    vt = torch.tensor(v, dtype=torch.float32).requires_grad_(True)
    ft = torch.tensor(f)
    inst = Instrument(quantise=False)
    op = ExactForward(inst, psi_grid(6), SMALL, device="cpu", backend="software")
    raw = op.raw_curves(vt.detach(), ft, geoms=[0, 5])
    cot = torch.zeros_like(raw); cot[:, 0] = 1.0                          # d(sum of I)
    raw2, gv, (g_rho, g_tau, g_ped) = op.vjp(vt, ft, cot, geoms=[0, 5],
                                            params=[inst.raw_rho, inst.raw_tau_i, inst.raw_pedestal])
    assert torch.allclose(raw, raw2)
    assert gv.shape == vt.shape and torch.isfinite(gv).all()
    assert float(g_rho) > 0.0
    assert torch.isfinite(g_tau) and torch.isfinite(g_ped).all()


def test_normalise_vjp_matches_autograd():
    """The hand-written adjoint of the per-curve mean normalisation equals autograd's."""
    raw = (torch.rand(3, 2, 7, dtype=torch.float64) + 0.5).requires_grad_(True)
    cot = torch.randn(3, 2, 7, dtype=torch.float64)
    (normalise(raw) * cot).sum().backward()
    assert torch.allclose(raw.grad, normalise_vjp(raw.detach(), cot))


def test_geometry_subset_matches_the_full_set():
    """Curves of a subset of the geometries equal the same rows of the full set."""
    v, f = _two_spheres()
    vt, ft = torch.tensor(v, dtype=torch.float32), torch.tensor(f)
    op = ExactForward(Instrument(quantise=False), psi_grid(4), SMALL, device="cpu",
                      backend="software")
    full = op.raw_curves(vt, ft)
    part = op.raw_curves(vt, ft, geoms=[3, 20])
    assert full.shape == (len(cameras()), 2, 4)
    assert torch.allclose(full[[3, 20]], part)


def test_a_distant_camera_still_sees_the_body():
    """The clip planes follow the camera distance, so a camera far enough away to be nearly
    orthographic still renders the body rather than clipping it."""
    v, f = _two_spheres()
    vt, ft = torch.tensor(v, dtype=torch.float32), torch.tensor(f)
    op = ExactForward(Instrument(eye_distance=400.0, quantise=False), psi_grid(3), SMALL,
                      device="cpu", backend="software")
    raw = op.raw_curves(vt, ft, geoms=[0, 9])
    assert torch.isfinite(raw).all() and (raw > 0).all()

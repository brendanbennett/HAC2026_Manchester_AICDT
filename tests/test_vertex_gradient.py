"""Finite-difference the vertex gradient of the exact forward model, under nvdiffrast.

This is the derivative everything rests on: the flow's data branch reads it, `polish()`
descends it, and any MAP route is nothing but it. Until this file, nothing checked it.

The reason for the gap is structural rather than an oversight. `LitCoverage` reads each face's
lit projected area off nvdiffrast's antialiasing, and reads the derivative off the same
mechanism -- so the shadow-edge motion that carries the whole concavity signal reaches the
vertices through the antialias and nowhere else. Every other test in this suite runs the
pure-torch fallback, whose `antialias` is the identity and whose `rasterize` runs under
`no_grad`, so under it that gradient is identically zero and a test that passed there would
have proved nothing. These tests therefore require a GPU with nvdiffrast and skip without one.

Measured here (RTX 4090, nvdiffrast 0.4.0, torch 2.5.1+cu121):

    lit coverage      sun_res 512 : corr 0.993, slope 0.921
                      sun_res 1024: corr 0.997, slope 0.967
    curves, intensity             : corr 0.945, slope 1.121
    curves, binary count          : corr 0.950, slope 0.554

The slope is the analytic gradient over the finite-difference one. Coverage converges toward 1
with resolution, which is the signature of the antialias's own discretisation rather than a
mistake. The binary channel sits near 0.55 for a reason worth knowing: `raw_curves` recomputes
Otsu's threshold from the first frame on every call, so a finite difference moves the threshold
and the derivative -- which holds it constant by design (exact.py:25-28) -- does not. The
direction is right and the magnitude is roughly halved, so the binary curves pull about half as
hard on the shape as they should. Thresholds are deliberately loose about the scale for that
reason, and tight about the correlation, which is what a descent direction actually needs.
"""
import numpy as np
import pytest
import torch
import trimesh

from hac26.conventions import psi_grid
from hac26.forward.mesh.exact import ExactForward, LitCoverage, RenderConfig
from hac26.forward.mesh.instrument import Instrument
from hac26.forward.mesh.raster import Rasteriser, flat_faces

pytestmark = pytest.mark.cuda

CFG = RenderConfig(height=54, width=96, supersample=2, sun_res=512, phase_chunk=8,
                   radiosity_faces=300)


def _nvdiffrast_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    try:
        from hac26.forward.shared._nvdr import load
        load()
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"needs nvdiffrast: {exc}")


def _dented_ball(subdiv=3):
    """A star-shaped ball with one deep dent: every vertex is on the visible surface, so no
    comparison is between two near-zeros, and the dent casts a real shadow."""
    ico = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    v = np.asarray(ico.vertices)
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    d = np.array([1.0, 0.25, 0.1]); d /= np.linalg.norm(d)
    r = 0.9 + 0.10 * u[:, 2] ** 2 - 0.42 * np.exp(-((1 - u @ d) / 0.18) ** 2)
    return u * r[:, None], np.asarray(ico.faces)


def _compare(analytic, fd):
    a, b = np.asarray(analytic), np.asarray(fd)
    return float(np.corrcoef(a, b)[0, 1]), float((a @ b) / max(b @ b, 1e-30))


def test_lit_coverage_vertex_gradient_matches_finite_differences():
    """The antialiasing-derived derivative of the lit projected area, against central
    differences, on two spheres where one shadows the other."""
    _nvdiffrast_or_skip()
    dev = "cuda"
    a = trimesh.creation.icosphere(subdivisions=2, radius=0.55)
    b = trimesh.creation.icosphere(subdivisions=2, radius=0.55)
    a.apply_translation([-0.375, 0, 0]); b.apply_translation([0.375, 0, 0])
    m = trimesh.util.concatenate([a, b])
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    ft = torch.tensor(f, device=dev)
    d = torch.tensor([[1.0, 0.12, 0.05]], device=dev); d = d / d.norm()
    torch.manual_seed(0)
    cot = torch.randn(len(f), device=dev)

    def total(vv):
        ras = Rasteriser(512, 512, 1, device=dev, backend="nvdiffrast")
        fv, ff = flat_faces(vv, ft)
        ext = float(vv.detach().norm(dim=1).max()) * 1.05
        return (LitCoverage.apply(fv, ff, d, ext, ext, ras)[0] * cot).sum()

    vt = torch.tensor(v, dtype=torch.float32, device=dev, requires_grad=True)
    total(vt).backward()
    g = vt.grad.detach()
    assert torch.isfinite(g).all() and float(g.norm()) > 0, \
        "the coverage gradient is dead: the backend is not antialiasing"

    h = 0.01
    idx = np.random.default_rng(0).choice(len(v), 10, replace=False)
    an, fd = [], []
    for i in idx:
        for k in range(3):
            vp = torch.tensor(v, dtype=torch.float32, device=dev); vp[i, k] += h
            vm = torch.tensor(v, dtype=torch.float32, device=dev); vm[i, k] -= h
            fd.append((float(total(vp)) - float(total(vm))) / (2 * h))
            an.append(float(g[i, k]))
    corr, slope = _compare(an, fd)
    assert corr > 0.95, f"coverage gradient direction is wrong: corr {corr:.3f}"
    assert 0.75 < slope < 1.25, f"coverage gradient magnitude is off: slope {slope:.3f}"


@pytest.mark.parametrize("channel,min_corr,slope_range",
                         [(0, 0.85, (0.8, 1.4)),     # intensity
                          (1, 0.85, (0.35, 1.3))])   # binary: Otsu is frozen; see the docstring
def test_curve_vertex_gradient_matches_finite_differences(channel, min_corr, slope_range):
    """The whole chain -- coverage, radiosity, raster, sensor, coarea thresholds -- against
    central differences, on the vertices carrying the most gradient."""
    _nvdiffrast_or_skip()
    dev = "cuda"
    v, f = _dented_ball()
    ft = torch.tensor(f, device=dev)
    geoms = [0, 5, 11]
    op = ExactForward(Instrument(quantise=False).to(dev), psi_grid(12), CFG, device=dev,
                      backend="nvdiffrast")

    raw0 = op.raw_curves(torch.tensor(v, dtype=torch.float32, device=dev), ft, geoms=geoms)
    torch.manual_seed(0)
    cot = torch.zeros_like(raw0)
    cot[:, channel, :] = torch.randn(raw0.shape[0], raw0.shape[2], device=dev)

    vt = torch.tensor(v, dtype=torch.float32, device=dev, requires_grad=True)
    _, gv, _ = op.vjp(vt, ft, cot, geoms=geoms, params=[])
    g = gv.detach()
    assert torch.isfinite(g).all() and float(g.norm()) > 0

    h = 0.01
    idx = torch.argsort(g.abs().sum(1), descending=True).cpu().numpy()[:8]
    an, fd = [], []
    for i in idx:
        for k in range(3):
            vp = torch.tensor(v, dtype=torch.float32, device=dev); vp[i, k] += h
            vm = torch.tensor(v, dtype=torch.float32, device=dev); vm[i, k] -= h
            lp = float((op.raw_curves(vp, ft, geoms=geoms) * cot).sum())
            lm = float((op.raw_curves(vm, ft, geoms=geoms) * cot).sum())
            fd.append((lp - lm) / (2 * h)); an.append(float(g[i, k]))
    corr, slope = _compare(an, fd)
    assert corr > min_corr, f"channel {channel} gradient direction is wrong: corr {corr:.3f}"
    lo, hi = slope_range
    assert lo < slope < hi, f"channel {channel} gradient magnitude is off: slope {slope:.3f}"

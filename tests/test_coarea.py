"""M3 completion test -- analytic coarea derivative against finite differences.

The specification's gate: 100 randomly chosen parameters, analytic versus FD within 1%.

The subtlety in testing this at all is that N(tau) is an integer, so a finite difference of
it is quantised. The derivative being checked is that of the CONTINUUM area whose pixel
count is an approximation, so the FD step has to be large enough that the count moves by
many pixels (making quantisation a small relative error) while still small enough to stay
in the linear regime. The step below is chosen on that basis and the test reports the
achieved agreement rather than only asserting it.
"""
import numpy as np
import pytest
import torch

from hac26.coarea import contour_weights, threshold_count, threshold_sum

N_PARAM = 100
H = W = 256
TAU = 0.5


def _field(theta: torch.Tensor, centres: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    """A smooth image built from N_PARAM broad bumps: u(x) = sum_k theta_k G_k(x).

    Smooth on the pixel scale, which is the same condition the PSF guarantees in the real
    pipeline and which is what makes the finite-difference gradient magnitude meaningful.
    """
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H, dtype=theta.dtype),
                            torch.linspace(0, 1, W, dtype=theta.dtype), indexing="ij")
    d2 = ((yy[None] - centres[:, 0, None, None]) ** 2
          + (xx[None] - centres[:, 1, None, None]) ** 2)
    # Divided by the number of bumps so u spans roughly [0, 1] and the level set at
    # tau = 0.5 is a substantial closed contour INSIDE the frame. Without this the summed
    # bumps put u in [0.27, 9.83], the level set covers 65450 of 65536 pixels, the contour
    # is a sliver at the border, and a finite difference of the count moves by 3 -- the test
    # then measures quantisation rather than the derivative.
    bumps = theta[:, None, None] * torch.exp(-d2 / (2 * widths[:, None, None] ** 2))
    return bumps.sum(0) / theta.shape[0] * 8.0


@pytest.fixture(scope="module")
def setup():
    g = torch.Generator().manual_seed(7)
    centres = torch.rand(N_PARAM, 2, generator=g, dtype=torch.float64)
    widths = 0.08 + 0.05 * torch.rand(N_PARAM, generator=g, dtype=torch.float64)
    theta = 0.5 + 0.5 * torch.rand(N_PARAM, generator=g, dtype=torch.float64)
    return centres, widths, theta


def _reference(fn, field, up=8):
    """Near-continuum value of the reduction, by evaluating it on an up-sampled field.

    THE REFERENCE HAS TO BE THE CONTINUUM QUANTITY. The coarea formula differentiates the
    AREA of the level set; the pixel count is an integer approximation to it. Finite
    differences of that integer are quantised, and at a step small enough to stay linear the
    count moves by only a few, so the comparison measures quantisation rather than the
    derivative: it reported 10% error while the derivative was in fact correct to 0.6%.
    Up-sampling 8x makes the reference 64x finer and the discrepancy converges properly.
    """
    import torch.nn.functional as Fn
    u = Fn.interpolate(field[None, None], scale_factor=up, mode="bicubic",
                       align_corners=True)[0, 0]
    return float(fn(u, TAU)) / up ** 2


def _fd_vs_analytic(fn, setup, step):
    centres, widths, theta = setup
    t = theta.clone().requires_grad_(True)
    fn(_field(t, centres, widths), TAU).backward()
    ana = t.grad.detach().clone()
    fd = torch.zeros_like(ana)
    for k in range(N_PARAM):
        tp = theta.clone(); tp[k] += step
        tm = theta.clone(); tm[k] -= step
        fd[k] = (_reference(fn, _field(tp, centres, widths))
                 - _reference(fn, _field(tm, centres, widths))) / (2 * step)
    return ana, fd


@pytest.mark.slow
def test_count_derivative_matches_finite_differences(setup):
    ana, fd = _fd_vs_analytic(threshold_count, setup, step=1e-1)
    keep = fd.abs() > 0.02 * fd.abs().max()      # parameters that actually move the contour
    rel = ((ana[keep] - fd[keep]).abs() / fd[keep].abs()).median()
    corr = float(np.corrcoef(ana.numpy(), fd.numpy())[0, 1])
    print(f"\n  dN/dtheta : {int(keep.sum())} active params, median rel err {float(rel):.4f}, "
          f"corr {corr:.6f}")
    assert corr > 0.999
    assert float(rel) < 0.01


@pytest.mark.slow
def test_sum_derivative_matches_finite_differences(setup):
    ana, fd = _fd_vs_analytic(threshold_sum, setup, step=1e-1)
    keep = fd.abs() > 0.02 * fd.abs().max()
    rel = ((ana[keep] - fd[keep]).abs() / fd[keep].abs()).median()
    corr = float(np.corrcoef(ana.numpy(), fd.numpy())[0, 1])
    print(f"  dI/dtheta : {int(keep.sum())} active params, median rel err {float(rel):.4f}, "
          f"corr {corr:.6f}")
    assert corr > 0.999
    assert float(rel) < 0.01


def test_contour_weights_reproduce_the_level_set_length(setup):
    """Sanity on the weights themselves: sum w |grad u| must equal the contour length."""
    centres, widths, theta = setup
    u = _field(theta, centres, widths).numpy()
    r, c, w = contour_weights(u, TAU)
    from skimage.measure import find_contours
    length = sum(float(np.sqrt(((k[1:] - k[:-1]) ** 2).sum(1)).sum())
                 for k in find_contours(u, TAU))
    gy, gx = np.gradient(u)
    g = np.sqrt(gx ** 2 + gy ** 2)
    recovered = float((w * g[r, c]).sum())
    assert recovered == pytest.approx(length, rel=0.05)


def test_no_softening_anywhere():
    """The forward pass must be the hard count, not a sigmoid of it."""
    u = torch.tensor([[0.0, 0.4], [0.6, 1.0]], dtype=torch.float64, requires_grad=True)
    assert float(threshold_count(u, 0.5)) == 2.0
    assert float(threshold_sum(u, 0.5)) == pytest.approx(1.6)

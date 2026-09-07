"""The last step of a reconstruction and the training term that mirrors it: the polish of a
draw on the exact misfit, and the data-fit term's value and gradient."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hac26.conventions import psi_grid                                         # noqa: E402
from hac26.field import CODE_DIM, N_DIR, ImplicitBody                          # noqa: E402
from hac26.forward.mesh.exact import RenderConfig                              # noqa: E402
from hac26.forward.mesh.instrument import Instrument                           # noqa: E402
from hac26.shapes import icosphere, mesh_support                              # noqa: E402
from hac26.solvers.lpd_flow import LPDFlow                                     # noqa: E402
from hac26.solvers.operator import CodeOperator                                # noqa: E402
from reconstruct_lpd import POLISH_TARGET, polish, whitened_misfit             # noqa: E402
from train_lpd import data_fit, with_gradient                                  # noqa: E402

SMALL = RenderConfig(height=24, width=40, supersample=1, sun_res=64, phase_chunk=4,
                     radiosity_faces=48)
GEOMS = [0, 7]


def _setup():
    op = CodeOperator(Instrument(quantise=False), psi_grid(4), res=16, config=SMALL,
                      device="cpu", backend="software")
    v, _ = icosphere(2)
    v = v * [0.9, 0.7, 1.0]
    h = torch.tensor(mesh_support(v, ImplicitBody().core.n.numpy()), dtype=torch.float32)
    target = torch.zeros(CODE_DIM)
    g = torch.zeros(12, 12, 12)
    g[7, 4, 6] = 0.3                                    # the dent the data know about
    target[N_DIR:] = g.reshape(-1)
    data = op.curves(h, target, 1.2, geoms=GEOMS)       # (2, 2, P) of the geometries used
    full = torch.ones(28, 2, data.shape[-1])
    full[GEOMS] = data
    scale = torch.full((28, 2), 0.01)
    net = LPDFlow(n_experts=1)
    gen = torch.Generator().manual_seed(0)
    codes = torch.zeros(8, CODE_DIM)
    codes[:, :N_DIR] = 0.05 * torch.randn(8, N_DIR, generator=gen)
    codes[:, N_DIR:] = 0.1 * torch.randn(8, CODE_DIM - N_DIR, generator=gen)
    net.codec.fit(codes)
    start = torch.zeros(CODE_DIM)                       # a generic body without the dent
    start[N_DIR:] = 0.03 * torch.randn(CODE_DIM - N_DIR, generator=gen)
    return op, h, net, full, scale, net.codec.encode(start)


def test_polish_lowers_the_misfit_and_stops_at_the_noise_level():
    op, h, net, data, scale, z = _setup()
    z2, before, after, n_it = polish(net, op, z, h, 1.2, data, scale, GEOMS, steps=3)
    assert before > POLISH_TARGET
    assert after < before and 1 <= n_it <= 3
    cur = op.curves(h, net.codec.decode(z2), 1.2, geoms=GEOMS)
    assert abs(whitened_misfit(cur, data, scale, GEOMS) - after) < 1e-4
    # a draw already at the noise level is left where it is
    fine = torch.full((28, 2), 100.0)                   # a scale so large that chi < 1
    z3, b3, a3, n3 = polish(net, op, z, h, 1.2, data, fine, GEOMS, steps=3)
    assert n3 == 0 and torch.equal(z3, z) and a3 == b3 <= POLISH_TARGET


def test_data_fit_is_zero_at_the_noise_level_and_its_gradient_points_downhill():
    op, h, net, data, scale, z = _setup()
    z = z[None]
    mask = torch.zeros(1, 28)
    mask[0, GEOMS] = 1.0
    val, grad, dropped = data_fit(net, op, z, h[None], torch.tensor([1.2]), data[None],
                                  scale[None], GEOMS, mask)
    assert dropped == 0 and float(val) > 0 and torch.isfinite(grad).all()
    # steps against the gradient land lower than the same steps along it, and the best of
    # them below the start (the test renderer is coarse and its misfit rough at small scales,
    # so the check is over a few step sizes rather than one)
    direction = grad / grad.pow(2).mean().sqrt()
    downs, ups = [], []
    for eps in (0.003, 0.01, 0.03):
        downs.append(float(data_fit(net, op, z - eps * direction, h[None], torch.tensor([1.2]),
                                    data[None], scale[None], GEOMS, mask)[0]))
        ups.append(float(data_fit(net, op, z + eps * direction, h[None], torch.tensor([1.2]),
                                  data[None], scale[None], GEOMS, mask)[0]))
    assert all(d < u for d, u in zip(downs, ups)) and min(downs) < float(val)
    # at the noise level the term and its gradient vanish
    fine = torch.full((1, 28, 2), 100.0)
    val3, grad3, _ = data_fit(net, op, z, h[None], torch.tensor([1.2]), data[None], fine,
                              GEOMS, mask)
    assert float(val3) == 0.0 and float(grad3.abs().max()) == 0.0
    # with_gradient carries the value and the gradient into autograd
    x = z.clone().requires_grad_(True)
    s = with_gradient(val, x, grad)
    s.backward()
    assert torch.allclose(x.grad, grad) and float(s) == float(val)

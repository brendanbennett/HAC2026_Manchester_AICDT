"""The training loss end to end on the software rasteriser: the operator at the prior's
endpoint estimate, the reader and expert in the loop, the data-fit term for late t, turned
bodies, rolled-out states and the ablation arm."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hac26.conventions import cameras, psi_grid                                # noqa: E402
from hac26.field import CODE_DIM, N_DIR, ImplicitBody                          # noqa: E402
from hac26.forward.mesh.exact import RenderConfig                              # noqa: E402
from hac26.forward.mesh.instrument import Instrument                           # noqa: E402
from hac26.shapes import icosphere, mesh_support                              # noqa: E402
from hac26.solvers.lpd_flow import LPDFlow, geometry_tags                     # noqa: E402
from hac26.solvers.operator import CodeOperator                                # noqa: E402
from train_lpd import FIT_FROM, Corpus, Diag, flow_loss, model_error_scale     # noqa: E402

SMALL = RenderConfig(height=24, width=40, supersample=1, sun_res=64, phase_chunk=4,
                     radiosity_faces=48)


def _corpus(op):
    """Two bodies with all their curves and turned counts, at P = 4 phases."""
    gen = torch.Generator().manual_seed(0)
    n = ImplicitBody().core.n.numpy()
    codes, curves, turned, sups = [], [], [], []
    for scale in ([0.9, 0.7, 1.0], [1.0, 0.8, 1.1]):
        v, _ = icosphere(2)
        h = torch.tensor(mesh_support(v * scale, n), dtype=torch.float32)
        code = torch.zeros(CODE_DIM)
        code[N_DIR:] = 0.05 * torch.randn(CODE_DIM - N_DIR, generator=gen)
        c, t = op.curves_turned(h, code, 1.2)
        codes.append(code); curves.append(c); turned.append(t); sups.append(h)
    stack = lambda xs: torch.stack(xs)                                           # noqa: E731
    return Corpus(stack(codes), stack(curves), stack(turned), stack(sups), stack(sups),
                  torch.tensor([1.2, 1.2]), torch.tensor([0, 1]))


def test_flow_loss_trains_reader_and_expert_with_every_term():
    inst = Instrument(quantise=False)
    op = CodeOperator(inst, psi_grid(4), res=16, config=SMALL, device="cpu",
                      backend="software")
    corpus = _corpus(op)
    net = LPDFlow(n_experts=1)
    net.codec.fit(corpus.codes)
    net.prior.requires_grad_(False)        # as train_lpd.load_prior leaves it
    with torch.no_grad():                  # open the zero-initialised output paths, so the
        for p in net.experts.parameters():   # reader's gradient is not zero by construction
            p.add_(0.01 * torch.randn_like(p))
    eta = model_error_scale(inst)
    C = len(cameras())
    tag, mask = geometry_tags(), torch.ones(1, C)
    idx = torch.tensor([0, 1])
    x0 = torch.randn(2, CODE_DIM, generator=torch.Generator().manual_seed(1))
    t = torch.tensor([0.3, 0.5 * (FIT_FROM + 1.0)])         # one early draw, one in the fit interval
    turns = torch.tensor([0, 1])
    loss, diag = flow_loss(net, op, corpus, eta, idx, x0, t, 2, tag, mask, train_geoms=2,
                           turns=turns, return_diag=True)
    assert isinstance(diag, Diag) and torch.isfinite(loss)
    assert diag.fit > 0 and diag.flow > 0 and diag.occ > 0 and diag.dropped == 0
    loss.backward()
    reader_grad = sum(float(p.grad.abs().sum()) for p in net.reader.parameters()
                      if p.grad is not None)
    expert_grad = sum(float(p.grad.abs().sum()) for p in net.experts[0].parameters()
                      if p.grad is not None)
    assert reader_grad > 0 and expert_grad > 0
    assert all(p.grad is None for p in net.prior.parameters())     # the prior is not trained here
    # the ablation arm: the prior alone, on the same draws
    full, prior_only, dropped = flow_loss(net, op, corpus, eta, idx, x0, t, 2, tag, mask,
                                          train_geoms=2, turns=turns, ablate=True)
    assert torch.isfinite(full) and torch.isfinite(prior_only) and dropped == 0
    # a rolled-out state: one step of the sampler, then the loss at that step's time
    loss_r = flow_loss(net, op, corpus, eta, idx[:1], x0[:1], t[:1], 2, tag, mask,
                       train_geoms=2, rollout_steps=torch.tensor([1]))
    assert torch.isfinite(loss_r)

"""Is the dead token field an initialisation problem, and does breaking it help?

TokenField initialises p and z to zeros, so every token is identical, the softmax
over them is uniform, and Delta(y) is a y-independent constant that the zero-mean
subtraction then removes exactly. The configuration is a symmetric critical point:
identical tokens receive identical gradients, so they never differentiate and the
correction stays zero for the whole of training.

This fits the same deep targets twice -- as shipped, and with p and z given
distinct random values -- and reports both the SDF error and the variance of the
resulting codes across bodies, which is what the flow regresses onto.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))

from hac26.field import ImplicitBody, TokenField  # noqa: E402
from token_sweep import targets  # noqa: E402

CACHE = Path("/home/claude/fitcache")


def break_symmetry(tf: TokenField, seed: int, radius: float = 1.0):
    """Give each token a distinct position and latent.

    Positions are spread through the body rather than clustered at the centre: a
    token's reach is sigma = 0.25 R, so tokens all at the origin cannot describe a
    neck at the waist even once their latents differ.
    """
    g = torch.Generator().manual_seed(seed)
    n = tf.p.shape[0]
    u = torch.randn(n, 3, generator=g)
    u = u / u.norm(dim=1, keepdim=True)
    r = 0.85 * radius * torch.rand(n, 1, generator=g) ** (1 / 3)
    with torch.no_grad():
        tf.p.copy_(u * r)
        tf.z.copy_(0.5 * torch.randn(tf.z.shape, generator=g))


def build(store, n_tokens, broken):
    torch.manual_seed(1234)
    shared = TokenField(radius=1.0, n_tokens=n_tokens)
    if broken:
        break_symmetry(shared, seed=0)
    bodies = []
    for i, h0 in enumerate(store["h0"]):
        b = ImplicitBody(radius=1.0)
        b.core.set_support(torch.tensor(h0))
        b.tokens = TokenField(radius=1.0, n_tokens=n_tokens)
        if broken:
            break_symmetry(b.tokens, seed=100 + i)
        b.tokens.q, b.tokens.k, b.tokens.v, b.tokens.mlp = (
            shared.q, shared.k, shared.v, shared.mlp)
        bodies.append(b)
    return shared, bodies


def run(broken, steps, batch, budget, tokens):
    tag = "broken" if broken else "shipped"
    store = targets()
    data = [(torch.tensor(P), torch.tensor(S)) for P, S in store["data"]]
    shared, bodies = build(store, tokens, broken)

    per_body = [p for b in bodies for p in (b.core.raw_h, b.tokens.p, b.tokens.z)]
    dec = (list(shared.q.parameters()) + list(shared.k.parameters())
           + list(shared.v.parameters()) + list(shared.mlp.parameters()))
    opt = torch.optim.Adam([{"params": per_body, "lr": 0.02},
                            {"params": dec, "lr": 2e-3}])

    ck = CACHE / f"init_{tag}_t{tokens}.pt"
    done = 0
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        for b, sd in zip(bodies, st["bodies"]):
            b.load_state_dict(sd)
        shared.load_state_dict(st["shared"], strict=False)
        opt.load_state_dict(st["opt"])
        done = st["done"]
        print(f"  [{tag}] resumed at {done}", flush=True)

    t0, s = time.time(), done
    while s < steps and time.time() - t0 < budget:
        idx = np.random.default_rng(s).integers(0, len(bodies), batch)
        loss = sum(((bodies[j](data[j][0]) - data[j][1]) ** 2).mean() for j in idx) / len(idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 300 == 0:
            print(f"    [{tag}] step {s:>5}  loss {float(loss.detach()):.5f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
        s += 1
    torch.save({"bodies": [b.state_dict() for b in bodies], "shared": shared.state_dict(),
                "opt": opt.state_dict(), "done": s}, ck)

    if s < steps:
        print(f"  [{tag}] at {s}/{steps}, re-run to continue")
        return

    rows, codes = [], []
    with torch.no_grad():
        for b, (P, S), meta in zip(bodies, data, store["meta"]):
            pred = b(P)
            delta = b.tokens(P)
            rows.append({"name": meta["name"], "D_rms": meta["D_rms"],
                         "mse": float(((pred - S) ** 2).mean()),
                         "sign_acc": float(((pred < 0) == (S < 0)).float().mean()),
                         "max_delta": float(delta.abs().max())})
            codes.append(torch.cat([b.tokens.p.reshape(-1), b.tokens.z.reshape(-1)]).numpy())
    codes = np.stack(codes)
    out = {"tag": tag, "tokens": tokens, "steps": s, "rows": rows,
           "code_var": float(codes.var(0).mean())}
    (CACHE / f"init_{tag}_t{tokens}.json").write_text(json.dumps(out, indent=2))
    print(f"  [{tag}] wrote report")


def report(tokens):
    print(f"{'init':>9} {'corpus mse':>11} {'corpus acc':>11} {'m3 mse':>9} "
          f"{'m3 acc':>8} {'max|Delta|':>11} {'code var':>10}")
    for tag in ("shipped", "broken"):
        p = CACHE / f"init_{tag}_t{tokens}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        corp = [r for r in d["rows"] if r["name"] != "TRUTH_model3"]
        m3 = next(r for r in d["rows"] if r["name"] == "TRUTH_model3")
        print(f"{tag:>9} {np.mean([r['mse'] for r in corp]):>11.5f} "
              f"{np.mean([r['sign_acc'] for r in corp]):>11.4f} {m3['mse']:>9.5f} "
              f"{m3['sign_acc']:>8.4f} {np.mean([r['max_delta'] for r in d['rows']]):>11.3e} "
              f"{d['code_var']:>10.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--broken", action="store_true")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--budget", type=float, default=200.0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a.tokens) if a.report else run(a.broken, a.steps, a.batch, a.budget, a.tokens)

#!/usr/bin/env python3
"""Fit the shape library as an autodecoder: one shared token decoder, per-body codes.

The code the LPD generates is (p, z) -- token positions and latents. That is not the whole
token field: the cross-attention weights and the output MLP live in TokenField too, so a code
only means something together with the decoder it was fitted against. Fitting each body
separately would give every body its own decoder and leave the codes mutually meaningless.

The decoder is therefore shared across the library and trained jointly with the per-body
codes, then saved beside them and loaded by the operator. Freezing a decoder at a fixed seed
would also make codes portable, but a random decoder is not reliably expressive: its features
happen to span what one body needs and not another's.

Public bodies are never in the library.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.field import ImplicitBody          # noqa: E402


def samples(verts, faces, n_pts=6000, seed=0):
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    rng = np.random.default_rng(seed)
    ext = float(np.abs(verts).max()) * 1.3
    pts = rng.uniform(-ext, ext, (n_pts, 3))
    surf, _ = trimesh.sample.sample_surface(m, n_pts // 2)
    pts = np.vstack([pts, surf + rng.normal(0, 0.03, surf.shape)])
    sd = -m.nearest.signed_distance(pts)       # trimesh: positive inside
    return (torch.tensor(pts, dtype=torch.float32),
            torch.tensor(sd, dtype=torch.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=40)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4, help="bodies per step")
    ap.add_argument("--out", default="runs/corpus_codes.npz")
    ap.add_argument("--decoder", default="runs/token_decoder.pt")
    a = ap.parse_args()

    from train_surrogate import shapes
    print(f"[1] sampling SDF for {a.bodies} bodies", flush=True)
    data, h0s = [], []
    ref = ImplicitBody(radius=1.0)
    nrm = ref.core.n.detach().cpu().numpy()
    for i, (v, f) in enumerate(shapes(a.bodies, seed=0)):
        P, S = samples(v, f, seed=i)
        data.append((P, S))
        h0s.append(np.array([max(1e-3, float((v @ n).max())) for n in nrm], dtype=np.float32))
        if i % 10 == 0:
            print(f"    body {i}", flush=True)

    # ONE decoder, shared by every body: the modules are assigned, not copied, so the
    # parameters are literally the same tensors.
    torch.manual_seed(1234)
    shared = ImplicitBody(radius=1.0).tokens
    bodies = []
    for i in range(len(data)):
        b = ImplicitBody(radius=1.0)
        b.core.set_support(torch.tensor(h0s[i]))
        b.tokens.q, b.tokens.k, b.tokens.v, b.tokens.mlp = (
            shared.q, shared.k, shared.v, shared.mlp)
        bodies.append(b)

    # raw_h, not h: h is softplus(raw_h), a property and therefore not a leaf tensor
    per_body = [p for b in bodies for p in (b.core.raw_h, b.tokens.p, b.tokens.z)]
    dec = list(shared.q.parameters()) + list(shared.k.parameters()) \
        + list(shared.v.parameters()) + list(shared.mlp.parameters())
    opt = torch.optim.Adam([{"params": per_body, "lr": 0.02},
                            {"params": dec, "lr": 2e-3}])
    print(f"[2] joint fit: {len(per_body)} per-body tensors, "
          f"{sum(p.numel() for p in dec)} shared decoder parameters", flush=True)
    t0 = time.time()
    for s in range(a.steps):
        idx = np.random.default_rng(s).integers(0, len(bodies), a.batch)
        loss = 0.0
        for j in idx:
            P, S = data[j]
            loss = loss + ((bodies[j](P) - S) ** 2).mean()
        loss = loss / len(idx)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 250 == 0 or s == a.steps - 1:
            print(f"    step {s:>5}  sdf loss {float(loss):.5f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    codes = np.stack([torch.cat([b.tokens.p.reshape(-1), b.tokens.z.reshape(-1)])
                      .detach().numpy() for b in bodies])
    sup = np.stack([b.core.h.detach().numpy() for b in bodies])   # the property
    Path("model").mkdir(exist_ok=True)
    np.savez(a.out, codes=codes, support=sup)
    torch.save({k: v for k, v in shared.state_dict().items()
                if not k.startswith(("p", "z"))}, a.decoder)
    print(f"  codes {codes.shape}, per-component variance {codes.var(0).mean():.5f}")
    print(f"  wrote {a.out} and {a.decoder}")


if __name__ == "__main__":
    main()

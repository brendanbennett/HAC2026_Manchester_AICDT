"""Does the token field have the capacity for model-3-grade concavity?

Fits the deep tail of the corpus, plus the actual public ground truth for model 3,
at several token counts. The corpus rewrite assumes the field can represent what it
supplies; if error at D_rms ~ 0.3 is set by the representation rather than by
optimisation, more training bodies cannot help and the token count is the thing to
change first.

Resumable per config, because a call here is capped at a few minutes.

  python tools/token_sweep.py --tokens 32 --steps 1200
  python tools/token_sweep.py --report
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

from hac26.field import ImplicitBody, TokenField  # noqa: E402

CACHE = Path("/home/claude/fitcache")
DEEP_MIN = 0.22


def targets():
    """The deep tail of the new corpus, plus ground-truth model 3, SDF-sampled once."""
    path = CACHE / "sweep_targets.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())

    from fit_shapes import samples
    from hac26.shapes_nonconvex import hull_deficit
    import trimesh

    store = pickle.loads((CACHE / "new_data.pkl").read_bytes())
    ref = ImplicitBody(radius=1.0)
    nrm = ref.core.n.detach().cpu().numpy()

    out = {"data": [], "meta": [], "h0": []}
    for i, meta in enumerate(store["meta"]):
        if meta["D_rms"] >= DEEP_MIN:
            out["data"].append(store["data"][i])
            out["meta"].append({**meta, "name": f"corpus{i:02d}"})
            out["h0"].append(store["h0"][i])

    # the real thing: public model 3, posed the way the field expects
    sys.path.insert(0, str(ROOT / "tools"))
    from render import pose
    m = trimesh.load(ROOT / "dataset/raw/AsteroidModel03_shape_public/asteroid3.stl",
                     process=False)
    v = pose(np.asarray(m.vertices, float), np.asarray(m.faces, np.int64))
    f = np.asarray(m.faces, np.int64)
    d, l = hull_deficit(trimesh.Trimesh(v, f, process=False))
    P, S = samples(v, f, seed=999)
    out["data"].append((P.numpy(), S.numpy()))
    out["meta"].append({"name": "TRUTH_model3", "D_rms": float(d), "D_lo": float(l)})
    out["h0"].append(np.array([max(1e-3, float((v @ nn).max())) for nn in nrm],
                              dtype=np.float32))

    path.write_bytes(pickle.dumps(out))
    print(f"  {len(out['data'])} targets ({len(out['data'])-1} corpus + model 3)")
    return out


def build(store, n_tokens):
    torch.manual_seed(1234)
    shared = TokenField(radius=1.0, n_tokens=n_tokens)
    bodies = []
    for h0 in store["h0"]:
        b = ImplicitBody(radius=1.0)
        b.core.set_support(torch.tensor(h0))
        b.tokens = TokenField(radius=1.0, n_tokens=n_tokens)
        b.tokens.q, b.tokens.k, b.tokens.v, b.tokens.mlp = (
            shared.q, shared.k, shared.v, shared.mlp)
        bodies.append(b)
    return shared, bodies


def run(a):
    store = targets()
    data = [(torch.tensor(P), torch.tensor(S)) for P, S in store["data"]]
    shared, bodies = build(store, a.tokens)

    per_body = [p for b in bodies for p in (b.core.raw_h, b.tokens.p, b.tokens.z)]
    dec = (list(shared.q.parameters()) + list(shared.k.parameters())
           + list(shared.v.parameters()) + list(shared.mlp.parameters()))
    opt = torch.optim.Adam([{"params": per_body, "lr": 0.02},
                            {"params": dec, "lr": 2e-3}])

    ck = CACHE / f"sweep_t{a.tokens}.pt"
    done = 0
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        for b, sd in zip(bodies, st["bodies"]):
            b.load_state_dict(sd)
        shared.load_state_dict(st["shared"], strict=False)
        opt.load_state_dict(st["opt"])
        done = st["done"]
        print(f"  resumed at {done}", flush=True)

    t0 = time.time()
    s = done
    while s < a.steps and time.time() - t0 < a.budget:
        idx = np.random.default_rng(s).integers(0, len(bodies), a.batch)
        loss = sum(((bodies[j](data[j][0]) - data[j][1]) ** 2).mean() for j in idx) / len(idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 200 == 0:
            print(f"    t{a.tokens} step {s:>5}  loss {float(loss.detach()):.5f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
        s += 1
    torch.save({"bodies": [b.state_dict() for b in bodies], "shared": shared.state_dict(),
                "opt": opt.state_dict(), "done": s}, ck)
    print(f"  t{a.tokens}: at step {s}/{a.steps}")

    if s >= a.steps:
        rows = []
        with torch.no_grad():
            for b, (P, S), meta in zip(bodies, data, store["meta"]):
                pred = b(P)
                rows.append({"name": meta["name"], "D_rms": meta["D_rms"],
                             "mse": float(((pred - S) ** 2).mean()),
                             "sign_acc": float(((pred < 0) == (S < 0)).float().mean())})
        (CACHE / f"sweep_t{a.tokens}.json").write_text(
            json.dumps({"tokens": a.tokens, "steps": s, "rows": rows}, indent=2))
        print(f"  wrote sweep_t{a.tokens}.json")


def report():
    print(f"{'tokens':>7} {'corpus mse':>11} {'corpus acc':>11} "
          f"{'model3 mse':>11} {'model3 acc':>11}")
    for p in sorted(CACHE.glob("sweep_t*.json"), key=lambda q: int(q.stem[7:])):
        d = json.loads(p.read_text())
        corp = [r for r in d["rows"] if r["name"] != "TRUTH_model3"]
        m3 = next(r for r in d["rows"] if r["name"] == "TRUTH_model3")
        print(f"{d['tokens']:>7} {np.mean([r['mse'] for r in corp]):>11.5f} "
              f"{np.mean([r['sign_acc'] for r in corp]):>11.4f} "
              f"{m3['mse']:>11.5f} {m3['sign_acc']:>11.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--budget", type=float, default=200.0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a)

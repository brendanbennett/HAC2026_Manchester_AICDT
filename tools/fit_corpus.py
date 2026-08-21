"""Fit the shape library on a named corpus, in resumable chunks.

Same protocol as scripts/fit_shapes.py -- one shared token decoder, per-body codes,
joint Adam -- but split into `prep` and `fit` so it can run inside a short call
budget, and instrumented to record per-body error against each body's hull deficit.
That last part is the actual question: the field carries 32 tokens, and if its
error grows with D_rms then the new corpus is asking for shapes the representation
cannot hold, which would matter more than anything about the sampler.

  python tools/fit_corpus.py prep --corpus new --bodies 40
  python tools/fit_corpus.py fit  --corpus new --steps 750     # repeat to continue
  python tools/fit_corpus.py report --corpus new
"""
from __future__ import annotations

import argparse
import ast
import json
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hac26.field import ImplicitBody  # noqa: E402

CACHE = Path("/home/claude/fitcache")
CACHE.mkdir(exist_ok=True)


def old_shapes(n, seed=0):
    """The six-archetype generator from main, imported without its torch-heavy module body."""
    import trimesh
    import hac26.shapes as HS

    src = (ROOT.parent / "oldgen/old_gen.py").read_text()
    tree = ast.parse(src)
    keep = [x for x in tree.body if not isinstance(x, (ast.Import, ast.ImportFrom))]
    mod = types.ModuleType("oldgen")
    mod.__dict__.update({"np": np, "trimesh": trimesh, "sys": sys, "Path": Path,
                         "__file__": str(ROOT.parent / "oldgen/old_gen.py")})
    for k in dir(HS):
        if not k.startswith("_"):
            mod.__dict__[k] = getattr(HS, k)
    exec(compile(ast.Module(body=keep, type_ignores=[]), "old", "exec"), mod.__dict__)
    out = []
    for v, f in mod.shapes(n, seed=seed):
        m = trimesh.Trimesh(v, f, process=False)
        try:
            from hac26.shapes_nonconvex import hull_deficit
            d, l = hull_deficit(m)
        except Exception:
            d, l = float("nan"), float("nan")
        out.append((np.asarray(v, float), np.asarray(f, np.int64),
                    {"kind": "old", "R": float("nan"), "D_rms": float(d), "D_lo": float(l)}))
    return out


def get_bodies(corpus, n, seed=0):
    if corpus == "new":
        from hac26.shapes_nonconvex import sample_corpus
        return sample_corpus(n, seed=seed, verbose=True)
    return old_shapes(n, seed=seed)


def cmd_prep(a):
    from fit_shapes import samples

    path = CACHE / f"{a.corpus}_data.pkl"
    store = pickle.loads(path.read_bytes()) if path.exists() else {"meta": [], "data": [], "h0": []}
    if len(store["data"]) >= a.bodies:
        print(f"already have {len(store['data'])} bodies")
        return

    print(f"generating {a.bodies} bodies for '{a.corpus}'...", flush=True)
    bodies = get_bodies(a.corpus, a.bodies, seed=0)
    ref = ImplicitBody(radius=1.0)
    nrm = ref.core.n.detach().cpu().numpy()

    t0 = time.time()
    for i in range(len(store["data"]), min(a.bodies, len(bodies))):
        v, f, meta = bodies[i]
        P, S = samples(v, f, seed=i)
        store["data"].append((P.numpy(), S.numpy()))
        store["meta"].append(meta)
        store["h0"].append(np.array([max(1e-3, float((v @ nn).max())) for nn in nrm],
                                    dtype=np.float32))
        print(f"    body {i}  D_rms {meta['D_rms']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if time.time() - t0 > a.budget:
            print("  budget reached, re-run prep to continue", flush=True)
            break
    path.write_bytes(pickle.dumps(store))
    print(f"  cached {len(store['data'])}/{a.bodies} bodies -> {path}")


def build(store):
    torch.manual_seed(1234)
    shared = ImplicitBody(radius=1.0).tokens
    bodies = []
    for h0 in store["h0"]:
        b = ImplicitBody(radius=1.0)
        b.core.set_support(torch.tensor(h0))
        b.tokens.q, b.tokens.k, b.tokens.v, b.tokens.mlp = (
            shared.q, shared.k, shared.v, shared.mlp)
        bodies.append(b)
    return shared, bodies


def cmd_fit(a):
    store = pickle.loads((CACHE / f"{a.corpus}_data.pkl").read_bytes())
    data = [(torch.tensor(P), torch.tensor(S)) for P, S in store["data"]]
    shared, bodies = build(store)

    per_body = [p for b in bodies for p in (b.core.raw_h, b.tokens.p, b.tokens.z)]
    dec = (list(shared.q.parameters()) + list(shared.k.parameters())
           + list(shared.v.parameters()) + list(shared.mlp.parameters()))
    opt = torch.optim.Adam([{"params": per_body, "lr": 0.02},
                            {"params": dec, "lr": 2e-3}])

    ck = CACHE / f"{a.corpus}_ck.pt"
    done = 0
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        for b, sd in zip(bodies, st["bodies"]):
            b.load_state_dict(sd)
        shared.load_state_dict(st["shared"], strict=False)
        opt.load_state_dict(st["opt"])
        done = st["done"]
        print(f"  resumed at step {done}", flush=True)

    t0 = time.time()
    for s in range(done, done + a.steps):
        idx = np.random.default_rng(s).integers(0, len(bodies), a.batch)
        loss = 0.0
        for j in idx:
            P, S = data[j]
            loss = loss + ((bodies[j](P) - S) ** 2).mean()
        loss = loss / len(idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 250 == 0:
            print(f"    step {s:>5}  sdf loss {float(loss.detach()):.5f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > a.budget:
            print("  budget reached", flush=True)
            done = s + 1
            break
    else:
        done = done + a.steps

    torch.save({"bodies": [b.state_dict() for b in bodies],
                "shared": shared.state_dict(), "opt": opt.state_dict(), "done": done}, ck)
    print(f"  saved at step {done} -> {ck}")


def cmd_report(a):
    store = pickle.loads((CACHE / f"{a.corpus}_data.pkl").read_bytes())
    data = [(torch.tensor(P), torch.tensor(S)) for P, S in store["data"]]
    shared, bodies = build(store)
    st = torch.load(CACHE / f"{a.corpus}_ck.pt", map_location="cpu", weights_only=False)
    for b, sd in zip(bodies, st["bodies"]):
        b.load_state_dict(sd)
    shared.load_state_dict(st["shared"], strict=False)

    rows = []
    with torch.no_grad():
        for i, (b, (P, S)) in enumerate(zip(bodies, data)):
            pred = b(P)
            mse = float(((pred - S) ** 2).mean())
            # sign agreement = does the fitted field put points on the right side?
            acc = float(((pred < 0) == (S < 0)).float().mean())
            rows.append({"i": i, "D_rms": store["meta"][i]["D_rms"],
                         "D_lo": store["meta"][i]["D_lo"], "mse": mse, "sign_acc": acc})

    out = CACHE / f"{a.corpus}_report.json"
    out.write_text(json.dumps({"steps": st["done"], "rows": rows}, indent=2))
    d = np.array([r["D_rms"] for r in rows])
    m = np.array([r["mse"] for r in rows])
    acc = np.array([r["sign_acc"] for r in rows])
    print(f"[{a.corpus}] {len(rows)} bodies, {st['done']} steps")
    print(f"  sdf mse    mean {m.mean():.5f}   median {np.median(m):.5f}   max {m.max():.5f}")
    print(f"  sign acc   mean {acc.mean():.4f}  min {acc.min():.4f}")
    for lo, hi in [(0.0, 0.02), (0.02, 0.12), (0.12, 0.22), (0.22, 0.50)]:
        k = (d >= lo) & (d < hi)
        if k.any():
            print(f"  D_rms [{lo:.2f},{hi:.2f})  n={k.sum():>2}  "
                  f"mse {m[k].mean():.5f}  sign acc {acc[k].mean():.4f}")
    if len(d) > 3 and np.ptp(d) > 0:
        print(f"  corr(D_rms, mse) = {np.corrcoef(d, m)[0,1]:+.3f}   "
              f"corr(D_rms, sign_acc) = {np.corrcoef(d, acc)[0,1]:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("prep", cmd_prep), ("fit", cmd_fit), ("report", cmd_report)):
        p = sub.add_parser(name)
        p.add_argument("--corpus", choices=["new", "old"], required=True)
        p.add_argument("--bodies", type=int, default=40)
        p.add_argument("--steps", type=int, default=750)
        p.add_argument("--batch", type=int, default=4)
        p.add_argument("--budget", type=float, default=200.0, help="seconds per call")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)

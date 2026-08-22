#!/usr/bin/env python3
"""Why is `pitted` under-represented? Measure each modifier's own rejection rate.

The audit showed `pitted` at 0.64 of its declared weight among accepted bodies (z = -5.5),
and showed the deficit is already present among bodies accepted on attempt 0. That rules out
modifier intensification -- `strength` only rises on a retry -- and leaves per-attempt
rejection: a draw containing `pitted` is thrown away by the gate more often than one without,
so the surviving sample under-represents it.

This tests that directly. `mod_weights` is forced to a single kind, so every modifier slot in
every attempt is that kind, and the base distribution is left alone. It replicates
`sample_body`'s retry loop rather than calling it, because the outcome of every REJECTED
attempt is what is being measured and `sample_body` discards it -- it returns only the
attempt that survived.

Everything else is `sample_body`'s logic verbatim: same order of checks, same `strength`
schedule, same `max_attempts`. One JSON line per attempt, so a killed run is still usable and
kinds can be run one per invocation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.shape_library import (LibrarySpec, _apply_modifier, _base, _draw,  # noqa: E402
                                 convexity_ratio, extract, is_edge_manifold,
                                 n_components, pose)


def attempts_for(rng, spec):
    """`sample_body`'s loop, yielding the outcome of EVERY attempt, not just the survivor."""
    base_kind = _draw(spec.base_weights, rng)
    for attempt in range(spec.max_attempts):
        strength = 1.0 + 0.25 * attempt
        s = 1.0
        f, _ = _base(rng, base_kind, s)
        n_mod = int(rng.integers(spec.n_modifiers[0], spec.n_modifiers[1] + 1))
        mods = []
        for _ in range(n_mod):
            mk = _draw(spec.mod_weights, rng)
            f, _r = _apply_modifier(f, rng, mk, s, strength, res=spec.res,
                                    extent=spec.extent)
            mods.append(mk)
        out = {"attempt": attempt, "base": base_kind, "n_mods": n_mod}
        try:
            v, fc, info = extract(f, extent=spec.extent, res=spec.res)
        except (ValueError, RuntimeError) as e:
            yield {**out, "ok": False, "reason": f"extract: {str(e)[:60]}"}
            continue
        if len(fc) < 100:
            yield {**out, "ok": False, "reason": "too few faces"}
            continue
        v = pose(v, radius=spec.radius, faces=fc)
        c = convexity_ratio(v, fc)
        if c >= spec.convexity_max:
            yield {**out, "ok": False, "reason": "convexity gate", "convexity": c}
            continue
        if not is_edge_manifold(fc):
            yield {**out, "ok": False, "reason": "not edge-manifold", "convexity": c}
            continue
        if n_components(v, fc) != 1:
            yield {**out, "ok": False, "reason": "multi-component", "convexity": c}
            continue
        yield {**out, "ok": True, "convexity": c, "n_faces": int(len(fc))}
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--bodies", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/audit/modgate")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.kind}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["body"] for l in path.open()}
    spec = LibrarySpec(mod_weights={a.kind: 1.0})
    fh = path.open("a")
    t0 = time.time()
    for i in range(a.bodies):
        if i in done:
            continue
        rng = np.random.default_rng([a.seed, i])
        for rec in attempts_for(rng, spec):
            fh.write(json.dumps({"body": i, "kind": a.kind, **rec}) + "\n")
        fh.flush()
        print(f"  {a.kind} body {i}  {time.time() - t0:.0f}s", flush=True)
    fh.close()


if __name__ == "__main__":
    main()

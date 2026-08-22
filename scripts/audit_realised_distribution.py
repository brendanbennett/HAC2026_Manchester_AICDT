#!/usr/bin/env python3
"""Measure the REALISED distribution of a shape-library run against its declared weights.

`sample_body` draws `base_kind` once, outside the retry loop, so the realised base
distribution can only differ from `base_weights` through outright failure: a kind whose
bodies exhaust `max_attempts` never reaches the corpus. `n_modifiers` and the modifier
kinds are still drawn INSIDE the loop, so their realised distribution is conditioned on
surviving the gate in exactly the way `base_kind` used to be.

This records, per drawn body: the base kind, whether it was accepted, how many attempts it
took, and the modifier kinds present on the accepted body. Failures are caught here rather
than propagated (`sample_body` raises, and `scripts/build_shape_library.py::_worker` does
not catch it).

Writes one JSON line per drawn body so a killed run is still usable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.shape_library import (LibrarySpec, convexity_ratio, hull_volume,  # noqa: E402
                                 mesh_volume, sample_body)


def aspect(verts: np.ndarray) -> list:
    """PCA extents, largest first, normalised by the largest."""
    c = verts - verts.mean(0)
    s = np.linalg.svd(c, compute_uv=False) / max(len(c), 1) ** 0.5
    s = np.sort(s)[::-1]
    return (s / max(s[0], 1e-12)).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--out", default="runs/audit/realised.jsonl")
    a = ap.parse_args()

    spec = LibrarySpec(res=a.res)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    if out.exists():
        done = sum(1 for _ in out.open())
        print(f"[resume] {done} bodies already recorded", flush=True)

    fh = out.open("a")
    t0 = time.time()
    for i in range(done, a.n):
        rng = np.random.default_rng([a.seed, i])
        t = time.time()
        rec: dict = {"index": i}
        try:
            b = sample_body(rng, spec)
        except RuntimeError as e:
            # The kind is not recoverable from the exception, so redraw it the same way
            # sample_body does -- a fresh generator on the same seed gives the same first
            # draw, which is the base kind.
            from hac26.shape_library import _draw
            rec.update({"base": _draw(spec.base_weights, np.random.default_rng([a.seed, i])),
                        "accepted": False, "reason": str(e)})
        else:
            rec.update({
                "base": b.recipe["base"],
                "accepted": True,
                "attempt": int(b.info["attempt"]),
                "convexity": float(b.convexity),
                "n_faces": int(len(b.faces)),
                "n_mods": len(b.recipe["mods"]),
                "mods": [m["kind"] for m in b.recipe["mods"]],
                "aspect": aspect(b.verts),
                "volume": float(mesh_volume(b.verts, b.faces)),
                "hull_volume": float(hull_volume(b.verts)),
            })
        rec["seconds"] = round(time.time() - t, 3)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if i % 25 == 0:
            el = time.time() - t0
            rate = el / max(i - done + 1, 1)
            print(f"  {i + 1}/{a.n}  {el / 60:.1f} min elapsed, "
                  f"~{rate * (a.n - i - 1) / 60:.1f} min left", flush=True)
    fh.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()

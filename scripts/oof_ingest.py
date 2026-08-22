#!/usr/bin/env python3
"""Stage 1 of the out-of-family test: ingest each source through `body_from_mesh`.

One shape per invocation (`--only`), written to its own npz, so the whole set can be built
across several short runs. In-family controls come from `sample_body` on fixed seeds and are
stored in the same format, so the fit stage cannot treat the two groups differently.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.shape_library import (LibrarySpec, body_from_mesh, convexity_ratio,  # noqa: E402
                                 is_edge_manifold, n_components, sample_body)
import scripts.oof_shapes as oof  # noqa: E402

OOF = {"jack": oof.jack, "cup": oof.cup, "trefoil": oof.trefoil, "steps": oof.steps}


def save(path, verts, faces, meta):
    np.savez_compressed(path, verts=verts, faces=faces, meta=json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, help="an OOF name, or ctrl<k> for a control")
    ap.add_argument("--out", default="runs/audit/bodies")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.only}.npz"
    if path.exists():
        print(f"[skip] {path} exists"); return

    t = time.time()
    if a.only.startswith("ctrl"):
        k = int(a.only[4:])
        b = sample_body(np.random.default_rng([0, 10_000 + k]), LibrarySpec())
        meta = {"group": "in_family", "base": b.recipe["base"],
                "mods": [m["kind"] for m in b.recipe["mods"]],
                "convexity": float(b.convexity), "attempt": int(b.info["attempt"])}
        v, f = b.verts, b.faces
    else:
        sv, sf = OOF[a.only]()
        b = body_from_mesh(sv, sf, rng=np.random.default_rng(0), add_modifiers=0)
        meta = {"group": "out_of_family", "base": a.only,
                "source_faces": int(len(sf)),
                "source_convexity": float(convexity_ratio(sv, sf)),
                "mods": [m["kind"] for m in b.recipe["mods"]],
                "convexity": float(b.convexity), "attempt": int(b.info["attempt"])}
        v, f = b.verts, b.faces

    meta.update({"n_faces": int(len(f)), "n_verts": int(len(v)),
                 "components": int(n_components(v, f)),
                 "edge_manifold": bool(is_edge_manifold(f)),
                 "seconds": round(time.time() - t, 1)})
    save(path, v, f, meta)
    print(f"{a.only:8s} {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()

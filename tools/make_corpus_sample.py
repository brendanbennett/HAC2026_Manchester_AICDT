"""Draw a sample of the new corpus, write the STLs, record the stats.

Deliberately a plain script with a fixed seed: whatever ends up in
figures/ and corpus_sample/ can be regenerated exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hac26.shapes_nonconvex import sample_corpus  # noqa: E402

N, SEED = 20, 7


def main():
    out = ROOT / "corpus_sample"
    out.mkdir(exist_ok=True)
    print(f"sampling {N} bodies (seed {SEED})...", flush=True)
    bodies = sample_corpus(N, seed=SEED, verbose=True)

    order = sorted(range(len(bodies)), key=lambda i: bodies[i][2]["D_rms"])
    meta = []
    for rank, i in enumerate(order):
        v, f, m = bodies[i]
        name = f"body{rank:02d}_{m['kind']}.stl"
        trimesh.Trimesh(v, f, process=False).export(out / name)
        meta.append({"file": name, **{k: (float(x) if isinstance(x, (int, float, np.floating))
                                          else x) for k, x in m.items()}})
        print(f"  {name:<34} D_rms {m['D_rms']:.3f}  D_lo {m['D_lo']:.2f}  R {m['R']:.2f}",
              flush=True)

    (out / "corpus_sample.json").write_text(json.dumps(
        {"n": N, "seed": SEED, "bodies": meta}, indent=2))
    np.save(out / "_meshes.npy",
            np.array([(bodies[i][0], bodies[i][1]) for i in order], dtype=object),
            allow_pickle=True)
    print("wrote", out)


if __name__ == "__main__":
    main()

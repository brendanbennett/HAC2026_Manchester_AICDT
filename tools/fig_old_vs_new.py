"""Previous generator against the new corpus, matched by hull deficit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from render import draw, pose  # noqa: E402

PICK_OLD = [1, 12, 0, 8, 11, 6]      # indices into the sampled old bodies
PICK_NEW = [2, 7, 9, 13, 15, 17]   # ranks in corpus_sample (sorted by D_rms)


def main():
    old = np.load("/tmp/old_bodies.npy", allow_pickle=True)
    meta = json.loads((ROOT / "corpus_sample/corpus_sample.json").read_text())["bodies"]

    n = len(PICK_OLD)
    fig, axs = plt.subplots(2, n, figsize=(1.95 * n, 4.9))

    for c, i in enumerate(PICK_OLD):
        v, f, d, l = old[i]
        draw(axs[0, c], pose(np.asarray(v, float), np.asarray(f, np.int64)), f,
             sub=f"D_rms {d:.3f}  D_lo {l:.2f}", color=(0.86, 0.79, 0.68), az=32 + 14 * c)
        print("  old", i, flush=True)

    for c, r in enumerate(PICK_NEW):
        b = meta[r]
        mesh = trimesh.load(ROOT / "corpus_sample" / b["file"], process=False)
        v = np.asarray(mesh.vertices, float)
        f = np.asarray(mesh.faces, np.int64)
        draw(axs[1, c], pose(v, f), f,
             sub=f"D_rms {b['D_rms']:.3f}  D_lo {b['D_lo']:.2f}",
             color=(0.78, 0.80, 0.86), az=32 + 14 * c)
        print("  new", r, flush=True)

    axs[0, 0].text(-0.08, 0.5, "previous\ngenerator", transform=axs[0, 0].transAxes,
                   rotation=90, ha="center", va="center", fontsize=10)
    axs[1, 0].text(-0.08, 0.5, "new\ncorpus", transform=axs[1, 0].transAxes,
                   rotation=90, ha="center", va="center", fontsize=10)

    fig.suptitle("Same deficit range, different shape families", fontsize=13, y=0.99)
    fig.text(0.5, -0.005,
             "The old draws that do reach deep D_rms get there through surface texture and "
             "machined primitives; several fall below\nthe D_lo floor. The new ones put the "
             "deficit into low-order lobes and necks, which is the form model 3 takes.",
             ha="center", fontsize=8.5, color="0.35")
    fig.subplots_adjust(top=0.9, hspace=0.3, wspace=0.03)
    out = ROOT / "figures/old_vs_new.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()

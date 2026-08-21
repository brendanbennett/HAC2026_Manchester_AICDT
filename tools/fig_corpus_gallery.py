"""Gallery of the sampled corpus, ordered by hull deficit."""
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

TRUTH = {1: 0.003, 2: 0.000, 3: 0.202}


def main():
    src = ROOT / "corpus_sample"
    meta = json.loads((src / "corpus_sample.json").read_text())["bodies"]

    ncol = 5
    nrow = int(np.ceil(len(meta) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.45 * nrow))
    for ax in axs.ravel():
        ax.axis("off")

    for i, b in enumerate(meta):
        ax = axs[i // ncol, i % ncol]
        ax.axis("on")
        mesh = trimesh.load(src / b["file"], process=False)
        v = np.asarray(mesh.vertices, float)
        f = np.asarray(mesh.faces, np.int64)
        conv = b["kind"] == "near_convex"
        draw(ax, pose(v, f), f,
             sub=f"D_rms {b['D_rms']:.3f}  R {b['R']:.2f}",
             color=(0.84, 0.81, 0.75) if conv else (0.78, 0.80, 0.86),
             az=30 + 11 * i, el=20)
        print(f"  rendered {b['file']}", flush=True)

    fig.suptitle("New corpus, 20 bodies (seed 7), ordered by hull deficit", fontsize=13, y=0.985)
    fig.text(0.5, 0.005,
             "Grey-brown = the 40% near-convex quota, kept on purpose. Blue = the non-convex "
             "strata, D_rms binned 0.05-0.45.\nPublic ground truth for scale: model 1 D_rms "
             "0.003, model 2 0.000, model 3 0.202.",
             ha="center", fontsize=8.5, color="0.35")
    fig.subplots_adjust(top=0.94, bottom=0.045, hspace=0.28, wspace=0.02)
    out = ROOT / "figures/corpus_gallery.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()

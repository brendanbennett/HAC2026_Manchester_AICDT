"""Ground truth against both reconstructions, for the three public models."""
from __future__ import annotations

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

# measured with scripts/eval_gate.py --fast against the public ground truth
SCORE = {
    ("convex", 1): 0.9792, ("convex", 2): 0.9128, ("convex", 3): 0.6922,
    ("lpd_fitted", 1): 0.9727, ("lpd_fitted", 2): 0.9346, ("lpd_fitted", 3): 0.6836,
}
CEIL = {1: 0.9913, 2: 1.0000, 3: 0.8669}
TRUTH_D = {1: (0.003, 0.17), 2: (0.000, 0.00), 3: (0.202, 0.91)}
RECON_D = {("convex", 1): 0.000, ("convex", 2): 0.000, ("convex", 3): 0.000,
           ("lpd_fitted", 1): 0.006, ("lpd_fitted", 2): 0.007, ("lpd_fitted", 3): 0.007}

COLS = [("truth", "ground truth"), ("convex", "convex solver"), ("lpd_fitted", "flow solver")]


def load(kind: str, m: int):
    if kind == "truth":
        p = ROOT / f"dataset/raw/AsteroidModel0{m}_shape_public/asteroid{m}.stl"
    else:
        p = ROOT / f"results/{kind}/Asteroid{m:02d}.stl"
    mesh = trimesh.load(p, process=False)
    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, np.int64)
    return pose(v, f), f


def main():
    fig, axs = plt.subplots(3, 3, figsize=(8.4, 9.0))
    for r, m in enumerate((1, 2, 3)):
        for c, (kind, label) in enumerate(COLS):
            ax = axs[r, c]
            v, f = load(kind, m)
            if kind == "truth":
                d, lo = TRUTH_D[m]
                sub = f"D_rms {d:.3f}  hull ceiling {CEIL[m]:.3f}"
                col = (0.74, 0.72, 0.70)
            else:
                sub = f"voxel {SCORE[(kind, m)]:.4f}   D_rms {RECON_D[(kind, m)]:.3f}"
                col = (0.84, 0.80, 0.72) if kind == "convex" else (0.80, 0.82, 0.86)
            draw(ax, v, f, title=label if r == 0 else None, sub=sub, color=col)
            if c == 0:
                ax.text(-0.06, 0.5, f"model {m}", transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=11)
            print(f"  rendered model {m} {kind}", flush=True)

    fig.suptitle("Public models: what the two solvers actually recover", fontsize=13, y=0.965)
    fig.text(0.5, 0.012,
             "Model 2's truth is an exactly convex cube, so its hull ceiling is 1.000; "
             "model 3 needs D_rms 0.202 of concavity and\nboth solvers emit at most 0.007. "
             "Voxel scores from scripts/eval_gate.py --fast.",
             ha="center", fontsize=8, color="0.35")
    fig.subplots_adjust(top=0.93, bottom=0.06, hspace=0.16, wspace=0.04)
    out = ROOT / "figures/truth_vs_recon.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()

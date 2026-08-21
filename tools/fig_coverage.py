"""Where each corpus lands in hull-deficit space, against the public ground truth.

The old points are sampled from the six-archetype generator on `main`. The
seven-archetype revision that CHANGES.md measures at D_rms <= 0.021 for five of
seven archetypes is not in either archive, so its numbers are quoted there, not
re-measured here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hac26.shapes_nonconvex import D_BINS, D_LO_MIN, NEAR_CONVEX_MAX  # noqa: E402

TRUTH = {1: (0.003, 0.17), 2: (0.000, 0.00), 3: (0.202, 0.91)}
EMITTED = {1: 0.006, 2: 0.007, 3: 0.007}   # what the flow solver actually produces


def main():
    old = np.load("/tmp/old_bodies.npy", allow_pickle=True)
    old_d = np.array([b[2] for b in old], float)
    old_l = np.array([b[3] for b in old], float)

    meta = json.loads((ROOT / "corpus_sample/corpus_sample.json").read_text())["bodies"]
    new_d = np.array([b["D_rms"] for b in meta])
    new_l = np.array([b["D_lo"] for b in meta])

    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    lo, hi = D_BINS[0][0], D_BINS[-1][1]
    ax.axhspan(D_LO_MIN, 1.02, xmin=0, xmax=1, color="#eef3f8", zorder=0)
    ax.add_patch(plt.Rectangle((lo, D_LO_MIN), hi - lo, 1.02 - D_LO_MIN,
                               facecolor="#dce8f4", edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((-0.01, -0.03), NEAR_CONVEX_MAX + 0.01, 1.08,
                               facecolor="#f3efe6", edgecolor="none", zorder=0))
    for b0, _ in D_BINS[1:]:
        ax.axvline(b0, color="white", lw=1.4, zorder=1)

    ax.scatter(old_d, old_l, s=44, marker="s", facecolor="none",
               edgecolor="#b08050", lw=1.3, label="previous generator (6 archetypes, main)",
               zorder=3)
    ax.scatter(new_d, new_l, s=52, color="#3d6fa8", alpha=0.85,
               label="new corpus (shapes_nonconvex)", zorder=4)

    for m, (d, l) in TRUTH.items():
        ax.scatter([d], [l], s=190, marker="*", color="#c8102e", zorder=6)
        ax.annotate(f"truth m{m}", (d, l), textcoords="offset points",
                    xytext=(11, 6), fontsize=9.5, color="#c8102e", weight="bold")

    ax.scatter([EMITTED[3]], [0.30], s=110, marker="X", color="0.25", zorder=6)
    ax.annotate("what the flow\nactually emits on m3", (EMITTED[3], 0.30),
                textcoords="offset points", xytext=(16, -6), fontsize=8.5, color="0.25")
    ax.annotate("", xy=(TRUTH[3][0], TRUTH[3][1]), xytext=(EMITTED[3], 0.32),
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.3,
                                connectionstyle="arc3,rad=-0.25"), zorder=5)

    ax.text(hi - 0.005, D_LO_MIN + 0.015, "accepted: D_lo >= 0.70, binned by D_rms",
            ha="right", va="bottom", fontsize=8.5, color="#2f5c8a")
    ax.text(NEAR_CONVEX_MAX / 2, 1.045, "near-convex\nquota (40%)", ha="center",
            va="top", fontsize=8.5, color="#8a6a3a")

    ax.set_xlabel("$D_{rms}$   — how much concavity the body has", fontsize=10.5)
    ax.set_ylabel("$D_{lo}$   — share of that concavity at $\\ell \\leq 4$", fontsize=10.5)
    ax.set_xlim(-0.012, 0.46)
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("Corpus coverage in deficit space", fontsize=13)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.15, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.text(0.5, -0.02,
             "The old generator does reach deep D_rms, but scatters: several draws sit below "
             "the D_lo floor, i.e. their concavity is\nhigh-frequency surface texture rather "
             "than the low-order lobes model 3 is made of. The new sampler enforces both axes.",
             ha="center", fontsize=8.5, color="0.35")
    out = ROOT / "figures/coverage.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()

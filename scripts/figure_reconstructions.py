#!/usr/bin/env python3
"""Render every reconstruction beside its truth, with the scores, into a PDF and PNGs.

    python scripts/figure_reconstructions.py
    python scripts/figure_reconstructions.py --scores results/public_scores.json \
        --out results/figures/reconstructions.pdf

Three pages: the public models against their released truth, the seven secret models with no
truth to show, and a summary of where the score is going. Shapes are drawn with flat shading
from face normals rather than a wireframe, because the thing to see is whether a body has a
waist, a facet or a crater, and a wireframe of a 25k-face mesh shows none of that.

Everything is posed into the challenge frame first (`centre_xy=False`; both the released
truths and our reconstructions are already on the rotation axis) so the three columns of a row
are the same body in the same frame, and a difference between them is a difference in shape.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages                 # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection              # noqa: E402

from hac26.conventions import CYLINDER_R, PUBLIC_MODELS              # noqa: E402
from hac26.data_io import public_stl                                 # noqa: E402
from hac26.shapes import rescale_touch_z                             # noqa: E402

LABELS = {"results/convex": "convex stage",
          "results/lpd": "convex + flow (old)",
          "results/flow_p30": "convex + flow (retrained)",
          "results/flow_p0": "convex + flow (no polish)",
          "results/flow_all": "convex + flow (retrained)"}

LIGHT = np.array([-1.0, 0.35, 0.55])          # the lab's light is at (-inf, 0, 0)
LIGHT = LIGHT / np.linalg.norm(LIGHT)
FACE_CAP = 4000                               # decimation cap: enough to read the silhouette
ELEV, AZIM = 22, -62
BG = "#0e1116"
FG = "#e8eaed"


def load_posed(path, cap: int = FACE_CAP):
    """Vertices and faces in the challenge frame, decimated enough to draw."""
    import trimesh
    from hac26.forward.mesh.exact import decimate
    m = trimesh.load(str(path), process=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    v, f = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
    if len(f) > cap:
        v, f = decimate(v, f, cap)
        v, f = np.asarray(v, float), np.asarray(f, np.int64)
    return rescale_touch_z(v, f, centre_xy=False), f


def draw(ax, v, f, base="#7aa2f7", title=None, subtitle=None):
    """Flat-shaded solid, lit from the lab's light direction."""
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-20)
    shade = np.clip(n @ LIGHT, 0.0, 1.0) * 0.78 + 0.22          # ambient floor
    rgb = np.array(matplotlib.colors.to_rgb(base))
    colours = np.clip(shade[:, None] * rgb[None, :], 0, 1)
    # painter's algorithm: matplotlib's 3D has no z-buffer, so draw far faces first
    view = np.array([np.cos(np.radians(ELEV)) * np.cos(np.radians(AZIM)),
                     np.cos(np.radians(ELEV)) * np.sin(np.radians(AZIM)),
                     np.sin(np.radians(ELEV))])
    order = np.argsort(tri.mean(1) @ view)
    pc = Poly3DCollection(tri[order], facecolors=colours[order], edgecolors="none",
                          linewidths=0, shade=False)
    ax.add_collection3d(pc)
    r = 1.05 * max(float(np.abs(v[:, :2]).max()), 1.0)
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-1.15, 1.15)
    ax.set_box_aspect((1, 1, 2.2 / (2 * r) * r))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    ax.set_facecolor(BG)
    if title:
        ax.set_title(title, color=FG, fontsize=10, pad=-2)
    if subtitle:
        ax.text2D(0.5, -0.02, subtitle, transform=ax.transAxes, ha="center", va="top",
                  color="#9aa5b1", fontsize=8.5)


def page(fig_title, rows, cols, cells, note=None):
    fig = plt.figure(figsize=(3.4 * len(cols), 3.5 * len(rows) + 1.0), facecolor=BG)
    fig.suptitle(fig_title, color=FG, fontsize=14, y=0.985)
    for i, rlab in enumerate(rows):
        for j, clab in enumerate(cols):
            ax = fig.add_subplot(len(rows), len(cols), i * len(cols) + j + 1,
                                 projection="3d", facecolor=BG)
            item = cells.get((rlab, clab))
            if item is None:
                ax.set_axis_off(); ax.set_facecolor(BG)
                ax.text2D(0.5, 0.5, "not released", transform=ax.transAxes, ha="center",
                          color="#5c6370", fontsize=10, style="italic")
                if i == 0:
                    ax.set_title(clab, color=FG, fontsize=10, pad=-2)
                continue
            v, f, sub, colour = item
            draw(ax, v, f, base=colour, title=clab if i == 0 else None, subtitle=sub)
            if j == 0:
                ax.text2D(-0.08, 0.5, rlab, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", color=FG, fontsize=11)
    if note:
        fig.text(0.5, 0.012, note, ha="center", color="#9aa5b1", fontsize=8.5)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.965))
    return fig


def summary_page(scores):
    """Where the score is going: per model, and the two measures against each other."""
    fig = plt.figure(figsize=(12, 7.2), facecolor=BG)
    fig.suptitle("Where the score is going  ·  public models, organisers' own measures",
                 color=FG, fontsize=14)
    models = [1, 2, 3]
    # whatever the caller actually scored, in the order given, rather than two fixed names
    palette = ["#7aa2f7", "#f7768e", "#bb9af7", "#7dcfff"]
    pipes = [(k, LABELS.get(k, Path(k).name), palette[i % len(palette)])
             for i, k in enumerate(scores)]

    ax = fig.add_subplot(1, 2, 1, facecolor=BG)
    w = 0.36
    x = np.arange(len(models))
    for k, (key, lab, col) in enumerate(pipes):
        vals = [scores[key][str(m)]["voxel"] for m in models]
        ax.bar(x + (k - 0.5) * w, vals, w, label=lab, color=col)
        for xi, vv in zip(x + (k - 0.5) * w, vals):
            ax.text(xi, vv + 0.012, f"{vv:.3f}", ha="center", color=FG, fontsize=8)
    hull = [0.9969, 0.9997, 0.8828]
    ax.plot(x, hull, "o--", color="#e0af68", label="convex hull of truth (ceiling)", ms=6)
    ax.set_xticks(x); ax.set_xticklabels([f"model {m}" for m in models], color=FG)
    ax.set_ylabel("voxel measure (Dice)", color=FG)
    ax.set_ylim(0, 1.08); ax.tick_params(colors=FG)
    ax.set_title("the voxel measure decides the competition", color=FG, fontsize=11)
    ax.legend(facecolor="#161b22", edgecolor="none", labelcolor=FG, fontsize=8.5, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.06))
    for s in ax.spines.values():
        s.set_color("#30363d")

    ax2 = fig.add_subplot(1, 2, 2, facecolor=BG)
    for k, (key, lab, col) in enumerate(pipes):
        vx = [scores[key][str(m)]["voxel"] for m in models]
        pj = [scores[key][str(m)]["proj_released"] for m in models]
        ax2.scatter(vx, pj, s=90, color=col, label=lab, zorder=3)
        for m, a, b in zip(models, vx, pj):
            ax2.annotate(str(m), (a, b), textcoords="offset points", xytext=(7, -3),
                         color=FG, fontsize=8)
    ax2.set_xlabel("voxel measure", color=FG); ax2.set_ylabel("projection measure", color=FG)
    ax2.set_xlim(0.6, 1.02); ax2.set_ylim(0.6, 1.02)
    ax2.tick_params(colors=FG)
    ax2.set_title("projection is saturated: it spans 0.02 where voxel spans 0.12",
                  color=FG, fontsize=11)
    ax2.legend(facecolor="#161b22", edgecolor="none", labelcolor=FG, fontsize=8.5, loc="lower right")
    for s in ax2.spines.values():
        s.set_color("#30363d")
    ax2.grid(alpha=0.15, color="#30363d")

    tot = {key: sum(scores[key][str(m)]["voxel"] + scores[key][str(m)]["proj_released"]
                    for m in models) for key, _, _ in pipes}
    parts = "    ·    ".join(f"{lbl} {tot[k]:.3f}" for k, lbl, _ in pipes)
    fig.text(0.5, 0.055,
             f"summed over the three public models (max 6):    "
             f"convex hull of truth 5.775    ·    {parts}",
             ha="center", color="#e0af68", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/public_scores.json")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--out", default="results/figures/reconstructions.pdf")
    ap.add_argument("--recon-dirs", nargs="+", default=["results/convex", "results/lpd"])
    a = ap.parse_args()

    scores = json.loads(Path(a.scores).read_text()) if Path(a.scores).exists() else {}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    def sub(key, m):
        r = scores.get(key, {}).get(str(m))
        if not r:
            return "no truth to score against"
        return f"voxel {r['voxel']:.3f}   projection {r['proj_released']:.3f}   → {r['score']:.3f}"

    labels = dict(LABELS)
    colours = {"results/convex": "#7aa2f7", "results/lpd": "#f7768e",
               "results/flow_p30": "#f7768e", "results/flow_p0": "#bb9af7"}
    figs = []

    cells, rows = {}, []
    for m in PUBLIC_MODELS:
        rlab = f"model {m}   R={CYLINDER_R[m]}"
        rows.append(rlab)
        v, f = load_posed(public_stl(a.data_dir, m))
        cells[(rlab, "released truth")] = (v, f, "ground truth", "#9ece6a")
        for d in a.recon_dirs:
            p = Path(d) / f"Asteroid{m:02d}.stl"
            if p.exists():
                vv, ff = load_posed(p)
                cells[(rlab, labels.get(d, d))] = (vv, ff, sub(d, m), colours.get(d, "#7aa2f7"))
    figs.append(page("Public models — reconstruction against released truth", rows,
                     ["released truth"] + [labels.get(d, d) for d in a.recon_dirs], cells,
                     note="Model 1 is Vesta (near-convex), 2 the sawed-off cube, 3 Mithra — a "
                          "contact binary, the only public body whose shape needs concavity."))

    cells, rows = {}, []
    for m in range(4, 11):
        rlab = f"model {m}   R={CYLINDER_R[m]}"
        rows.append(rlab)
        for d in a.recon_dirs:
            p = Path(d) / f"Asteroid{m:02d}.stl"
            if p.exists():
                vv, ff = load_posed(p)
                cells[(rlab, labels.get(d, d))] = (vv, ff, "", colours.get(d, "#7aa2f7"))
    figs.append(page("Secret models — the seven that are scored", rows,
                     [labels.get(d, d) for d in a.recon_dirs], cells,
                     note="No truth is released for these. Model 10's R = 3.95 makes it much "
                          "wider than tall; the rest lie between 0.67 and 1.48."))

    if scores:
        figs.append(summary_page(scores))

    with PdfPages(out) as pdf:
        for i, fg in enumerate(figs):
            pdf.savefig(fg, facecolor=BG)
            png = out.with_name(f"{out.stem}_p{i + 1}.png")
            fg.savefig(png, dpi=150, facecolor=BG)
            print(f"  wrote {png}")
            plt.close(fg)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

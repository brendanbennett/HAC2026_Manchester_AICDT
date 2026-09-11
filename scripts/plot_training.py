#!/usr/bin/env python3
"""Plot a flow training run: the losses, the validation curve and the cost of a step.

    python scripts/plot_training.py runs/train_metrics.jsonl --out results/figures/training.png
    python scripts/plot_training.py logs/flow.log logs/flow-rollout.log --label "2500 bodies"
    python scripts/plot_training.py runs/*/train_metrics.jsonl --out compare.png

Takes either the JSONL that train_lpd now writes or the text log it prints, and will mix them:
the structured file is the one to prefer, but a run already in flight has only the prose, and
a training curve one cannot see is a training curve one does not check. Several inputs are
drawn on the same axes, labelled, so two configurations can be compared directly.

What to look at, in the order it usually matters:

  validation      the only curve that says whether the run is learning rather than
                  memorising. The last run's failure is visible here and nowhere else: it
                  early-stopped, then the rollout phase resumed and branched into experts,
                  which reset the early-stopping record, and 120 more steps ended at a
                  validation loss 17x worse than the checkpoint it resumed from -- labelled
                  "best" because the record had been cleared.
  dropped bodies  a body whose mesh degenerates has no curves and leaves the batch. A rising
                  share means the sampler is making bodies the operator cannot render, and
                  the loss falling while it rises is not progress.
  s/step          with the operator called once per batch element in a Python loop, this is
                  the budget. Steps, not epochs, are what this model is short of.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

TRAIN_RE = re.compile(
    r"step\s+(\d+)\s+loss\s+([\d.eE+-]+)\s+\(flow\s+([\d.eE+-]+),\s+occupancy\s+([\d.eE+-]+),"
    r"\s+data fit\s+([\d.eE+-]+)\)\s+dropped\s+(\d+)/(\d+).*?([\d.]+)s/step")
VAL_RE = re.compile(r"step\s+(\d+)\s+val\s+([\d.eE+-]+)")

BG, FG, GRID = "#0e1116", "#e8eaed", "#30363d"
COLOURS = ["#7aa2f7", "#f7768e", "#9ece6a", "#e0af68", "#bb9af7", "#7dcfff"]


def read_jsonl(path: Path) -> tuple[list, list]:
    train, val = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        (val if r.get("phase") == "val" or "val" in r else train).append(r)
    return train, val


def read_text(path: Path) -> tuple[list, list]:
    """The printed log. Tolerant of anything else in the file: pipeline banners, warnings,
    the output of other stages, and a line cut in half by a job hitting its wallclock."""
    train, val = [], []
    for line in path.read_text(errors="replace").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            train.append({"step": int(m[1]), "loss": float(m[2]), "flow": float(m[3]),
                          "occupancy": float(m[4]), "data_fit": float(m[5]),
                          "dropped": int(m[6]), "seen": int(m[7]), "step_s": float(m[8])})
            continue
        m = VAL_RE.search(line)
        if m:
            val.append({"step": int(m[1]), "val": float(m[2])})
    return train, val


def load(path: Path) -> tuple[list, list]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    return read_text(path)


def series(rows, key):
    xs = [r["step"] for r in rows if key in r and r[key] is not None]
    ys = [r[key] for r in rows if key in r and r[key] is not None]
    return np.array(xs, float), np.array(ys, float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="train_metrics.jsonl files, or printed logs")
    ap.add_argument("--label", nargs="*", default=None, help="one label per input")
    ap.add_argument("--out", default="results/figures/training.png")
    ap.add_argument("--title", default="Flow training")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for i, raw in enumerate(a.inputs):
        p = Path(raw)
        if not p.exists():
            print(f"  skipping {p}: not found")
            continue
        tr, vl = load(p)
        if not tr and not vl:
            print(f"  skipping {p}: no training lines found in it")
            continue
        label = (a.label[i] if a.label and i < len(a.label) else p.parent.name or p.stem)
        runs.append((label, tr, vl))
        print(f"  {label}: {len(tr)} train points, {len(vl)} validation points, "
              f"to step {max([r['step'] for r in tr + vl], default=0)}")
    if not runs:
        raise SystemExit("nothing to plot")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), facecolor=BG)
    panels = [("loss", "training loss", axes[0][0], True),
              ("val", "validation loss  (the one that matters)", axes[0][1], True),
              ("dropped_frac", "bodies dropped, share of batch", axes[1][0], False),
              ("step_s", "seconds per step", axes[1][1], False)]
    for key, title, ax, logy in panels:
        ax.set_facecolor(BG); ax.set_title(title, color=FG, fontsize=11)
        ax.tick_params(colors=FG); ax.grid(alpha=0.15, color=GRID)
        for s in ax.spines.values():
            s.set_color(GRID)
        for j, (label, tr, vl) in enumerate(runs):
            c = COLOURS[j % len(COLOURS)]
            if key == "val":
                x, y = series(vl, "val")
                if len(x):
                    ax.plot(x, y, "o-", color=c, ms=4, lw=1.8, label=label)
                    k = int(np.argmin(y))
                    ax.plot([x[k]], [y[k]], "*", color=c, ms=15)
            elif key == "dropped_frac":
                x = np.array([r["step"] for r in tr if r.get("seen")], float)
                y = np.array([r["dropped"] / max(r["seen"], 1) for r in tr if r.get("seen")])
                if len(x):
                    ax.plot(x, y, "-", color=c, lw=1.6, label=label)
            else:
                x, y = series(tr, key)
                if len(x):
                    ax.plot(x, y, "-", color=c, lw=1.4, alpha=0.85, label=label)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("step", color=FG)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(facecolor="#161b22", edgecolor="none", labelcolor=FG, fontsize=8.5)

    # the loss components of the first run, which is where a rise in one is visible
    label, tr, _ = runs[0]
    if tr and "flow" in tr[0]:
        ax = axes[0][0]
        for k, name, style in (("flow", "flow", ":"), ("occupancy", "occupancy", "--"),
                               ("data_fit", "data fit", "-.")):
            x, y = series(tr, k)
            if len(x) and np.any(y > 0):
                ax.plot(x, y, style, color="#9aa5b1", lw=1.0, alpha=0.8, label=f"{name} ({label})")
        ax.legend(facecolor="#161b22", edgecolor="none", labelcolor=FG, fontsize=7.5)

    fig.suptitle(a.title, color=FG, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=BG)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()

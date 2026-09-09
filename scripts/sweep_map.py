#!/usr/bin/env python3
"""Run Track A over the public models at several settings and tabulate what it did to Dice.

    python scripts/sweep_map.py --calibration models/instrument_quick.pt --steps 200
    python scripts/sweep_map.py --models 1 2 3 --l2 1e-2 3e-3 1e-3 --out runs/sweep_map

The question is not which setting fits best. It is whether descending the misfit moves the
body toward the truth at all, and the sweep exists because that answer may depend on how hard
the ridge pulls: with no prior 1728 amplitudes can fit noise, and with too much they cannot
carve. Each run prints Dice against truth every checkpoint, so the table below can report both
the best Dice reached and the Dice at the lowest misfit -- and the gap between those two is
the whole question. If the best Dice arrives early and then decays while the misfit keeps
falling, the fit is eating the data rather than recovering the body, and no amount of tuning
fixes that; the honest response is to stop and say so.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_one(model, l2, lr, steps, calibration, phases, out_dir, hold_out, every):
    tag = f"m{model}_l2-{l2:g}_lr-{lr:g}"
    out = Path(out_dir) / tag / f"Asteroid{model:02d}.stl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/reconstruct_map.py", "--model", str(model),
           "--out", str(out), "--steps", str(steps), "--l2", str(l2), "--lr", str(lr),
           "--phases", str(phases), "--every", str(every),
           "--calibration", calibration, "--hold-out-geoms", str(hold_out)]
    print(f"\n=== {tag} ===", flush=True)
    r = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    sys.stdout.write(r.stdout[-2500:])
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-1500:])
        return tag, None
    j = out.with_suffix(".json")
    return tag, (json.loads(j.read_text()) if j.exists() else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--l2", nargs="+", type=float, default=[1e-2, 3e-3, 1e-3])
    ap.add_argument("--lr", nargs="+", type=float, default=[0.01])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--phases", type=int, default=48)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--hold-out-geoms", type=int, default=6)
    ap.add_argument("--calibration", default="models/instrument_calibration.pt")
    ap.add_argument("--out", default="runs/sweep_map")
    a = ap.parse_args()

    rows = []
    for model in a.models:
        for l2 in a.l2:
            for lr in a.lr:
                tag, meta = run_one(model, l2, lr, a.steps, a.calibration, a.phases,
                                    a.out, a.hold_out_geoms, a.every)
                if meta is None:
                    rows.append({"tag": tag, "failed": True})
                    continue
                h = [r for r in meta["history"] if "dice" in r]
                start = h[0] if h else {}
                best_d = max(h, key=lambda r: r["dice"]) if h else {}
                best_x = min(h, key=lambda r: r["chi_fit"]) if h else {}
                rows.append({
                    "tag": tag, "model": meta["model"], "l2": l2, "lr": lr,
                    "dice_start": start.get("dice"), "dice_best": best_d.get("dice"),
                    "dice_best_step": best_d.get("step"),
                    "dice_at_min_chi": best_x.get("dice"), "min_chi": best_x.get("chi_fit"),
                    "chi_start": start.get("chi_fit"),
                    "chi_held_at_min": best_x.get("chi_held"),
                    "convexity_final": meta.get("final_convexity"),
                })

    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{'run':<22} {'dice0':>7} {'dice*':>7} {'@step':>6} {'dice@minchi':>12} "
          f"{'chi0':>7} {'minchi':>7} {'held':>7} {'convex':>7}")
    for r in rows:
        if r.get("failed"):
            print(f"{r['tag']:<22}  FAILED"); continue
        def g(k, w=7, p=4):
            v = r.get(k)
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 1) + "-"
        print(f"{r['tag']:<22} {g('dice_start')} {g('dice_best')} "
              f"{str(r.get('dice_best_step','-')):>6} {g('dice_at_min_chi', 12)} "
              f"{g('chi_start', 7, 2)} {g('min_chi', 7, 2)} {g('chi_held_at_min', 7, 2)} "
              f"{g('convexity_final', 7, 3)}")
    print("\ndice* is the best Dice reached; dice@minchi is the Dice at the lowest misfit.\n"
          "A large gap between them means the misfit and the score disagree, and the misfit\n"
          "is the only one of the two we can see on a secret model.")


if __name__ == "__main__":
    main()

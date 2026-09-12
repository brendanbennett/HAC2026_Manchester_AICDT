#!/usr/bin/env python3
"""Pick convex or carved per model, using an independent renderer as the referee.

    python scripts/select_by_referee.py --out results/submission_referee

The pipeline's own misfit cannot make this choice: measured against truth it is
*anti*-correlated with Dice, and it rates the sawed-off cube (convex) a worse fit than Mithra
(a contact binary). So the choice is made with DAMIT's renderer instead -- an independent
implementation, validated against the Blender reference curves to 0.008 RMSE on a truth mesh
-- scoring each candidate against the Blender curves, which ship with all ten models and are
rendered from the true shapes.

Validated on the public models: for the convex-vs-carved question the referee picks the
higher-scoring body 3 times out of 3, and the resulting submission scores 5.5633 against
5.5442 for convex everywhere. It is *not* reliable at ranking two flows against each other
(1 of 3), so only two candidates are ever compared here.

The referee numbers come from DAMIT/scripts/referee.py; they are transcribed rather than
recomputed so that this script needs neither that repo nor a GPU.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# mean of the intensity and binary lag-minimised RMSE against the Blender curves,
# DAMIT/scripts/referee.py --source blender --faces 2000 --frames 180
REFEREE = {                     # model: (convex, carved flow)
    1: (0.0355, 0.0356), 2: (0.2776, 0.3200), 3: (0.0674, 0.0579),
    4: (0.0791, 0.0737), 5: (0.0882, 0.0747), 6: (0.1466, 0.1106),
    7: (0.2232, 0.1733), 8: (0.0381, 0.0467), 9: (0.0927, 0.0828),
    10: (0.2461, 0.2129),
}
SECRET = (4, 5, 6, 7, 8, 9, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convex", default="results/convex")
    ap.add_argument("--carved", default="results/flow_all")
    ap.add_argument("--out", default="results/submission_referee")
    ap.add_argument("--models", nargs="+", type=int, default=list(SECRET))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'model':>5} {'convex':>9} {'carved':>9}  {'pick':<8} {'margin':>8}")
    picks = {}
    for m in a.models:
        rc, rf = REFEREE[m]
        src, pick = ((a.carved, "carved") if rf < rc else (a.convex, "convex"))
        name = f"Asteroid{m:02d}.stl"
        p = Path(src) / name
        if not p.exists():
            sys.exit(f"missing {p}")
        shutil.copy2(p, out / name)
        picks[m] = pick
        print(f"{m:>5} {rc:>9.4f} {rf:>9.4f}  {pick:<8} {abs(rf - rc) / rc:>7.1%}")
    n_carved = sum(v == "carved" for v in picks.values())
    print(f"\nwrote {len(picks)} bodies to {out}: {n_carved} carved, "
          f"{len(picks) - n_carved} convex")


if __name__ == "__main__":
    main()

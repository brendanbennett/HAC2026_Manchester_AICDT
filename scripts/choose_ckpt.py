"""Pick which training checkpoint to take the late EMA from, then extract it.

Normally the newest checkpoint is the one wanted. But the training loss here is heavy-tailed
-- an earlier phase of this run spiked from ~0.8 to 5.5 before recovering -- and the EMA
window is ~100 steps, so a spike near the end would contaminate the averaged weights. This
scores each candidate by the MEDIAN of (flow + occupancy) over the WINDOW steps ending at its
step, ignoring steps where the sparse data-fit term fired (that term is zero on most steps
and occasionally 3.6, so pooling the two populations measures how often it fired rather than
how the network is doing), and takes the newest candidate that is not much worse than the
best one.

usage: choose_ckpt.py OUT.pt DECAY SNAPDIR LOG [LOG...]
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

WINDOW = 150      # steps of history summarised for each candidate
TOL = 1.25        # a candidate is "good" if within this factor of the best window median

pat = re.compile(r"\] step\s+(\d+)\s+loss\s+[\d.]+\s+\(flow\s+([\d.]+), occupancy\s+([\d.]+), "
                 r"data fit\s+([\d.]+)\)")


def main() -> None:
    out, decay, snapdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    logs = sys.argv[4:]

    rows = []
    for lg in logs:
        try:
            txt = Path(lg).read_text(errors="ignore")
        except OSError:
            continue
        for m in pat.finditer(txt):
            s, f, o, d = m.groups()
            if float(d) == 0.0:                    # clean steps only
                rows.append((int(s), float(f) + float(o)))
    rows.sort()

    cands = []
    for p in sorted(snapdir.glob("step_*.ckpt"), key=lambda q: int(q.stem.split("_")[1])):
        cands.append((int(p.stem.split("_")[1]), p))
    if not cands:
        raise SystemExit(f"no snapshots in {snapdir}")

    def window_median(step):
        v = [x[1] for x in rows if step - WINDOW < x[0] <= step]
        return float(np.median(v)) if v else None

    scored = [(s, p, window_median(s)) for s, p in cands]
    usable = [(s, p, m) for s, p, m in scored if m is not None]
    print(f"  {len(scored)} snapshots, {len(rows)} clean logged steps")
    for s, p, m in scored:
        print(f"    step {s:>5}  window median {'n/a' if m is None else f'{m:.4f}'}")

    if not usable:
        step, path, med = scored[-1][0], scored[-1][1], None
        print("  no loss history for any snapshot; taking the newest")
    else:
        best = min(m for _, _, m in usable)
        good = [(s, p, m) for s, p, m in usable if m <= best * TOL]
        step, path, med = good[-1]
        newest = usable[-1]
        if step != newest[0]:
            print(f"  !!! newest snapshot (step {newest[0]}, median {newest[2]:.4f}) is worse "
                  f"than {TOL:g}x the best ({best:.4f}); falling back to step {step}")
        else:
            print(f"  newest snapshot is within {TOL:g}x of the best; taking step {step}")

    print(f"  chosen: {path} (step {step}"
          + (f", window median {med:.4f}" if med is not None else "") + ")")
    subprocess.run([sys.executable, "scripts/extract_late.py", str(path), out, decay], check=True)


if __name__ == "__main__":
    main()

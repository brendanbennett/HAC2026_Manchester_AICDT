#!/usr/bin/env python3
"""Rebuild a body from the codes a reconstruction saved, optionally without the carving.

    python scripts/decode_codes.py results/final_p30 --out results/final_nocarve --zero-g

A code is `[dh(128), g(1728)]`: `dh` refines the convex core's support function, `g` carves
concavities into it. The reconstruction saves every draw's raw code next to the STL, so the
two halves can be separated after the fact without paying for the inversion again.

`--zero-g` keeps the flow's convex refinement and throws the carving away. That answers a
question the scores raise and cannot settle on their own: when a flow beats the convex stage
on a convex body, is it carving well or is it simply estimating `h + dh` better? The first
would be surprising on a body with no concavities; the second is the ordinary explanation and
is worth having as its own pipeline, because it applies to every model rather than the ones
that happen to be concave.

The draw rebuilt is the one the original run chose, read from the sibling JSON, so the output
is comparable to that run's STL body for body.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.conventions import CYLINDER_R, psi_grid                     # noqa: E402
from hac26.field import N_SITES                                        # noqa: E402
from hac26.solvers.operator import CodeOperator                        # noqa: E402
from hac26.solvers.output import export_stl                            # noqa: E402
from hac26.recon import fit_to_cylinder                                 # noqa: E402
from train_lpd import CALIBRATION, RENDER, load_instrument             # noqa: E402
from reconstruct_lpd import decode                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory holding Asteroid<NN>.codes.npz and .json")
    ap.add_argument("--out", required=True, help="directory to write the rebuilt STLs into")
    ap.add_argument("--models", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--zero-g", action="store_true",
                    help="drop the carving and keep the convex refinement")
    ap.add_argument("--zero-dh", action="store_true",
                    help="drop the convex refinement and keep the carving")
    ap.add_argument("--all-draws", action="store_true",
                    help="write every draw, not just the one the run answered with, as "
                         "Asteroid<NN>.draw<K>.stl. The medoid rule picks among the draws by "
                         "volume and side-view agreement, which has no connection to the "
                         "data; this lets a referee pick instead")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--phases", type=int, default=48)
    ap.add_argument("--operator-res", type=int, default=64)
    ap.add_argument("--calibration", default=CALIBRATION)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inst = load_instrument(a.calibration, dev)

    for M in a.models:
        cp = src / f"Asteroid{M:02d}.codes.npz"
        jp = src / f"Asteroid{M:02d}.json"
        if not cp.exists():
            print(f"model {M:2d}: no codes at {cp}, skipped", flush=True)
            continue
        d = np.load(cp)
        codes, support = d["codes"], torch.as_tensor(d["support"])
        # the draw the original run answered with; a consensus answer has no single code, and
        # rebuilding an arbitrary draw in its place would not be the same body
        k = 0
        if jp.exists():
            meta = json.loads(jp.read_text())
            if not str(meta.get("answer", "")).startswith("draw"):
                print(f"model {M:2d}: answered from {meta.get('answer')}, "
                      f"not a single draw -- rebuilding draw {int(meta['candidate'])}",
                      flush=True)
            k = int(meta["candidate"])
        k = min(k, len(codes) - 1)

        op = CodeOperator(inst, psi_grid(a.phases), res=a.operator_res, config=RENDER,
                          device=dev)
        wanted = range(len(codes)) if a.all_draws else [k]
        for j in wanted:
            code = torch.as_tensor(codes[j]).clone()
            if a.zero_g:
                code[-N_SITES:] = 0.0
            if a.zero_dh:
                code[:-N_SITES] = 0.0
            name = (f"Asteroid{M:02d}.draw{j}.stl" if a.all_draws
                    else f"Asteroid{M:02d}.stl")
            v, f, _ = decode(op, code.to(dev), support.to(dev), res=a.res)
            if v is None:
                print(f"model {M:2d} draw {j}: decoded to nothing", flush=True)
                continue
            # canonical -> physical, xy only. `decode` leaves the body at r_xy ~ 1; the
            # scorer compares it against a truth at the printed model's radius, so skipping
            # this scores a body of the right shape at the wrong size.
            v = fit_to_cylinder(v, CYLINDER_R[M])
            try:
                info = export_stl(out / name, v, f)
            except ValueError as exc:
                print(f"model {M:2d} draw {j}: {exc}", flush=True)
                continue
            print(f"model {M:2d} draw {j}{' (answered)' if j == k else ''}: "
                  f"{info.get('faces', len(f))} faces, "
                  f"volume {info.get('volume', float('nan')):.3f}", flush=True)


if __name__ == "__main__":
    main()

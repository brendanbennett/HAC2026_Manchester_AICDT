#!/usr/bin/env python3
"""Reconstruct a challenge model with the convex LPD, from one checkpoint or a
Minkowski-average ensemble, and write the posed STL.

Uses the decode path of eval_exact.py, so what is written is what that script scores.

    python scripts/reconstruct.py --ckpt a.pt b.pt --ensemble --smooth 1 --fit-cylinder \
        --model 4 --out results/convex/Asteroid04.stl
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_exact import decode, ensemble_body, predict_h  # noqa: E402
from hac26.conventions import CYLINDER_R  # noqa: E402
from hac26.data_io import load_model_curves  # noqa: E402
from hac26.recon import save_submission_stl  # noqa: E402
from hac26.train import load_net  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="checkpoint files")
    ap.add_argument("--model", type=int, required=True, help="challenge model number")
    ap.add_argument("--out", required=True, help="output STL path")
    ap.add_argument("--data-dir", default="dataset/raw", help="challenge data directory")
    ap.add_argument("--ensemble", action="store_true",
                    help="Minkowski-average all checkpoints instead of using the first")
    ap.add_argument("--smooth", type=int, default=0,
                    help="half-width of the support smoothing window; 0 disables it")
    ap.add_argument("--fit-cylinder", action="store_true",
                    help="scale xy to the published bounding-cylinder radius")
    args = ap.parse_args()

    torch.set_num_threads(4)
    nets, grids, prs = [], [], []
    for c in args.ckpt:
        n, p, g = load_net(c, device="cpu")
        nets.append(n)
        grids.append(g)
        prs.append(p)
    pr = prs[0]
    data = load_model_curves(args.data_dir, args.model, m=pr.m)
    if not data["files"]:
        raise SystemExit(f"no lightcurve files for model {args.model}")
    R = CYLINDER_R.get(args.model)

    h_list = [predict_h(n, g, data["curves"], data["mask"], R)
              for n, g in zip(nets, grids)]
    if args.ensemble and len(nets) > 1:
        v, f = ensemble_body(h_list, grids, R, args.smooth, args.fit_cylinder)
    else:
        v, f = decode(h_list[0], grids[0], args.smooth, R, args.fit_cylinder)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    info = save_submission_stl(args.out, v, f, cylinder_radius=R)
    info["n_ckpt"] = len(args.ckpt)
    info["ensemble"] = bool(args.ensemble and len(nets) > 1)
    info["smooth"] = args.smooth
    info["fit_cylinder"] = bool(args.fit_cylinder)
    print(json.dumps(info))


if __name__ == "__main__":
    main()

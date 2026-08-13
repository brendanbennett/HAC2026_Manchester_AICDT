#!/usr/bin/env python3
"""Reconstruct a challenge model from one checkpoint or a Minkowski-average ensemble.

Shares its decode path with eval_exact.py, so what is scored is exactly what is
written -- the two previous shipping bugs in this project were both cases of the
evaluated configuration and the written configuration silently diverging.

    python reconstruct_final.py --ckpt a.pt b.pt --ensemble --smooth 1 --fit-cylinder \
        --model 4 --out results/lpd/Asteroid04.stl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_exact import CYLINDER_R, decode, ensemble_body, predict_h  # noqa: E402
from hac26.data_io import load_model_curves  # noqa: E402
from hac26.recon import save_submission_stl  # noqa: E402
from hac26.train import load_net  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="../data/raw")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--smooth", type=int, default=0)
    ap.add_argument("--fit-cylinder", action="store_true")
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

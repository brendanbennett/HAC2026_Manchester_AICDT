#!/usr/bin/env python3
"""Fully synthetic end-to-end demo: sample a body, simulate its 56 curves with noise,
reconstruct with a trained LPD (+ Minkowski), report the Dice score vs. the truth.

    python scripts/make_demo.py --ckpt checkpoints/lpd_gpu_final.pt --out demo_out
If no checkpoint is given, a quick smoke model is trained first (plumbing check only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forward_models.convex_egi import stack_A  # noqa: E402
from hac26.geometry import build_cameras  # noqa: E402
from hac26.recon import (dice, reconstruct_from_curves, save_submission_stl,  # noqa: E402
                         voxel_grid, voxelize_convex)
from hac26.shapes import sample_training_shape  # noqa: E402
from hac26.train import PRESETS, auto_device, load_net, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--voxels", type=int, default=96)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = auto_device()

    ckpt = args.ckpt
    if ckpt is None:
        print("no checkpoint given -> training the smoke preset (plumbing check)")
        ckpt = str(train(PRESETS["smoke"], out_dir=str(out / "ckpt"), device=dev))
    net, pr, grid = load_net(ckpt, device=dev)

    rng = np.random.default_rng(args.seed)
    truth = sample_training_shape(rng, grid)
    A, _ = stack_A(grid, build_cameras(), pr.m, c_lambert=pr.c_lambert,
                   sigma=pr.sigma, delta=pr.delta)
    raw = np.einsum("cmn,n->cm", A, truth["g"])
    raw += args.noise * raw.mean(1, keepdims=True) * rng.standard_normal(raw.shape)
    d = raw / np.maximum(raw.mean(1, keepdims=True), pr.eps_norm)
    mask = np.ones(A.shape[0], dtype=np.float32)

    rec = reconstruct_from_curves(net, grid, d, mask, device=dev)
    info = save_submission_stl(str(out / "reconstruction.stl"), rec["verts"], rec["faces"])

    ext = 1.1 * max(np.abs(truth["verts"][:, :2]).max(), np.abs(rec["verts"][:, :2]).max())
    pts = voxel_grid(ext, n=args.voxels)
    vt = voxelize_convex(truth["verts"], truth["faces"], pts)
    vr = voxelize_convex(rec["verts"], rec["faces"], pts)
    score = dice(vt, vr)
    p_cos = float(truth["p"] @ rec["p"] /
                  max(np.linalg.norm(truth["p"]) * np.linalg.norm(rec["p"]), 1e-12))
    report = {"kind": truth["kind"], "dice_voxel_measure": score, "egi_cosine": p_cos,
              "minkowski_success": rec["minkowski"]["success"], **info}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

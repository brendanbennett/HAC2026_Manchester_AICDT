#!/usr/bin/env python3
"""Export the shipped model as a slim checkpoint suitable for version control.

A training checkpoint is 95 MB, of which 92.9 MB is `op.A` -- the (56, 360, 1152)
photometric operator. That tensor is a pure function of the preset (grid resolution,
camera list, frame count, c_lambert, sigma, delta), so `build_model` regenerates it bit
for bit on load and there is no reason to carry it. Dropping it and the other two
derived buffers leaves the 608,371 learned parameters, about 2.4 MB.

    python export_model.py --ckpt <training-ckpt> --out models/lpd_convex.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hac26.train import REGENERABLE_BUFFERS, load_net  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_dice2/lpd_gpu_final.pt")
    ap.add_argument("--out", default="models/lpd_convex.pt")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    slim = {k: v for k, v in ck["model"].items() if k not in REGENERABLE_BUFFERS}
    dropped = sorted(set(ck["model"]) - set(slim))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": slim, "preset": ck["preset"], "step": ck.get("step")}, out)

    before = Path(args.ckpt).stat().st_size / 1e6
    after = out.stat().st_size / 1e6
    print(f"dropped (regenerated from the preset): {dropped}")
    print(f"{before:.1f} MB -> {after:.1f} MB")

    # a checkpoint that cannot be loaded is worse than no checkpoint
    net, pr, grid = load_net(str(out), device="cpu")
    n = sum(p.numel() for p in net.parameters())
    print(f"reloaded OK: {n} parameters, grid {pr.n_theta}x{pr.n_phi}, "
          f"support_head={net.support_head}, r_cond={net.r_cond}")


if __name__ == "__main__":
    main()

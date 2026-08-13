#!/usr/bin/env python3
"""Train the LPD model. Examples:
    python scripts/train_lpd.py --preset smoke
    python scripts/train_lpd.py --preset gpu --out checkpoints
    python scripts/train_lpd.py --preset gpu --steps 20000 --resume checkpoints/lpd_gpu_step2000.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.train import PRESETS, Preset, auto_device, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="gpu", choices=sorted(PRESETS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--noise-profile-mode", choices=["measured", "flat"],
                    default=None, help="measured = the replicate-derived "
                    "heteroscedastic profile; flat = homoscedastic")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--warm-start", default=None,
                    help="initialise from an existing (possibly ungated) checkpoint; the "
                         "gate starts closed (a = sigmoid(-9) ~ 1e-4), so the run begins "
                         "at a known-good solution instead of from scratch")
    ap.add_argument("--lr", type=float, default=None,
                    help="peak learning rate. The preset's 1e-3 is a FROM-SCRATCH rate; "
                         "warm-starting at it walks the pretrained weights straight off "
                         "the solution it was handed. Use ~2e-4 to fine-tune.")
    ap.add_argument("--data", default=None,
                    help="root of the team dataset (meshes [+ *_curves.npz]); "
                         "omit to train on the built-in synthetic sampler")
    ap.add_argument("--mix", type=float, default=0.25,
                    help="fraction of built-in synthetic shapes mixed in (breadth reserve)")
    ap.add_argument("--shape-source", default=None, choices=["synthetic", "damit"],
                    help="source for the mixed-in breadth-reserve shapes; 'damit' uses "
                         "real inversion-derived asteroid meshes instead of the "
                         "synthetic SH/polytope sampler")
    ap.add_argument("--damit-dir", default=None, help="root holding DAMIT shape.txt files")
    ap.add_argument("--support", action="store_true",
                    help="add the support-function head and train h(u); the body is "
                         "then a half-space intersection instead of a Minkowski solve")
    ap.add_argument("--canonical-r", action="store_true",
                    help="train on the r_max=1 canonical shape; pair with "
                         "reconstruct.py --fit-cylinder, its exact inverse")
    ap.add_argument("--r-cond", action="store_true",
                    help="feed the a-priori bounding radius R to the network as an "
                         "input channel (test-time R is the published cylinder radius)")
    ap.add_argument("--egi-weight", type=float, default=None,
                    help="weight on the auxiliary EGI objective when --support is set")
    ap.add_argument("--dice-weight", type=float, default=None,
                    help="weight on the EXACT Dice metric (hac26.radial). This is the "
                         "scoring function itself, computed in closed form from the "
                         "support function, so it is a loss and not a surrogate")
    ap.add_argument("--h-mse-weight", type=float, default=None,
                    help="weight on the support MSE; keep non-zero with --dice-weight "
                         "because Dice is scale-invariant and cannot pin the size")
    ap.add_argument("--gate-rank", type=int, default=None,
                    help="rank of the occlusion gate; 0 disables it. R=1 with the "
                         "zero-initialised scale reproduces the ungated solver exactly")
    ap.add_argument("--p-flat", type=float, default=None,
                    help="fraction of flat-faced / few-face training bodies (prisms, "
                         "platonic solids, plane-cut ellipsoids). Challenge model 2 is "
                         "a cube and the original family contained no such shapes")
    ap.add_argument("--n-rays", type=int, default=None,
                    help="sphere quadrature size for the Dice loss")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader workers; the input pipeline (shape sampling + "
                         "einsum) is the bottleneck on a fast GPU, so raise this to "
                         "keep the device fed")
    args = ap.parse_args()
    pr: Preset = PRESETS[args.preset]
    if args.workers is not None:
        pr.num_workers = args.workers
    if args.support:
        pr.support_head = True
    if args.canonical_r:
        pr.canonical_r = True
    if args.r_cond:
        pr.r_cond = True
    if args.egi_weight is not None:
        pr.egi_weight = args.egi_weight
    if args.dice_weight is not None:
        pr.dice_weight = args.dice_weight
    if args.h_mse_weight is not None:
        pr.h_mse_weight = args.h_mse_weight
    if args.p_flat is not None:
        pr.p_flat = args.p_flat
    if args.gate_rank is not None:
        pr.gate_rank = args.gate_rank
    if args.n_rays is not None:
        pr.n_rays = args.n_rays
    if args.shape_source is not None:
        pr.shape_source = args.shape_source
    if args.damit_dir is not None:
        pr.damit_dir = args.damit_dir
    if args.steps is not None:
        pr.steps = args.steps
    if args.batch is not None:
        pr.batch = args.batch
    if args.seed is not None:
        pr.seed = args.seed
    if args.noise_profile_mode is not None:
        pr.noise_profile_mode = args.noise_profile_mode
    if args.lr is not None:
        pr.lr = args.lr
    dev = args.device or auto_device()
    dataset = None
    if args.data:
        from hac26.adapter import FigurineCurves, load_pairs
        from hac26.forward import stack_A
        from hac26.geometry import build_cameras, make_grid
        grid = make_grid(pr.n_theta, pr.n_phi)
        A, _ = stack_A(grid, build_cameras(), pr.m, c_lambert=pr.c_lambert,
                       sigma=pr.sigma, delta=pr.delta)
        from hac26.radial import fibonacci_sphere
        pairs = load_pairs(args.data, grid, A, eps=pr.eps_norm,
                           canonical_r=pr.canonical_r,
                           rays=fibonacci_sphere(pr.n_rays) if pr.dice_weight else None)
        print(f"team dataset: {len(pairs)} mesh/curve pairs from {args.data}")
        dataset = FigurineCurves(pairs, pr, mix_synthetic=args.mix, grid=grid, A=A)
    print(f"preset={pr.name} device={dev} steps={pr.steps} batch={pr.batch} "
          f"grid={pr.n_theta}x{pr.n_phi} I={pr.n_iter} ch={pr.ch}")
    final = train(pr, out_dir=args.out, device=dev, resume=args.resume, dataset=dataset,
                  warm_start_from=args.warm_start)
    print(f"final checkpoint: {final}")


if __name__ == "__main__":
    main()

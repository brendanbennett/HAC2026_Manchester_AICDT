#!/usr/bin/env python3
"""Check the rule that picks the answer, on bodies whose truth is known.

reconstruct_lpd.py chooses the answer to a rock among the draws and their consensus bodies by
expected score against the draws. Whether that rule beats the alternatives cannot be checked
on the challenge models, since only three truths are public and those enter the calibration.
The held-out corpus bodies can check it: their truth is known, they never enter training, and
their curves are the exact model's curves with noise and model error at the sizes training
draws. For each of them this script reconstructs the body as reconstruct_lpd.py does (draws,
polish, candidates) and scores every candidate against the truth with both challenge
measures. It reports, per body and on average, how the rule's pick compares with the rules
one could use instead: the draw that fits the curves best, the draw closest to the other
draws, each consensus level on its own, and the best candidate in hindsight.

The bodies are spread over how deeply carved they are, so the check covers smooth and
deeply carved bodies alike. Writes --out as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.conventions import cameras, psi_grid                             # noqa: E402
from hac26.field import N_DIR, apply_constraints                           # noqa: E402
from hac26.recon import dice, fit_to_cylinder, mesh_occupancy              # noqa: E402
from hac26.scoring.side_view import measure_outlines, outline_set, surface_points   # noqa: E402
from hac26.solvers.lpd_flow import CHURN, N_MODES, N_STEPS, LPDFlow, geometry_tags   # noqa: E402
from hac26.solvers.operator import CodeOperator                            # noqa: E402
from hac26.solvers.output import metric_medoid                             # noqa: E402
from reconstruct_lpd import (CONSENSUS_LEVELS, consensus_bodies, decode,   # noqa: E402
                             make_resid_fn, mesh_misfit_by_geom, polish)
from train_lpd import (CALIBRATION, CORPUS, RENDER, _enable_tf32, cond_channels,   # noqa: E402
                       file_digest, held_out, load_corpus, load_instrument,
                       model_error_scale, noise_sigma, site_field, smooth_noise_like)

RULES = ("vote", "best_fit", "medoid", "oracle") + tuple(f"consensus_{lv:g}" for lv in CONSENSUS_LEVELS)


def carving(corpus) -> np.ndarray:
    """How deeply carved each corpus body is: one minus the share of its hull's lattice sites
    that are inside the body."""
    g = corpus.codes[:, N_DIR:]
    body = (site_field(corpus.support_true, g) < 0).float().sum(1)
    hull = (site_field(corpus.support_true, torch.zeros_like(g)) < 0).float().sum(1)
    return (1.0 - body / hull.clamp_min(1.0)).clamp_min(0.0).cpu().numpy()


def spread_over(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of k entries spread evenly over the sorted values (all of them when k is not
    smaller than their number)."""
    order = np.argsort(values)
    if k >= len(order):
        return order
    return order[np.round(np.linspace(0, len(order) - 1, k)).astype(int)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/lpd_flow.pt")
    ap.add_argument("--corpus", default=CORPUS, help="must match the training run")
    ap.add_argument("--calibration", default=CALIBRATION, help="must match the training run")
    ap.add_argument("--val-bodies", type=int, default=8,
                    help="must match the training run, so the same bodies are held out")
    ap.add_argument("--bodies", type=int, default=8,
                    help="held-out bodies to check, spread over how carved they are")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--churn", type=float, default=CHURN)
    ap.add_argument("--polish-steps", type=int, default=30)
    ap.add_argument("--res", type=int, default=64, help="extraction resolution of the meshes")
    ap.add_argument("--side-points", type=int, default=200000,
                    help="surface samples per body for the side-view measure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/decision_check.json")
    a = ap.parse_args()
    _enable_tf32()
    torch.manual_seed(a.seed)

    data, meta = load_corpus(a.corpus)
    if meta["calibration"] != file_digest(a.calibration):
        raise SystemExit(f"{a.corpus} was built with another calibration than {a.calibration}")
    phases, op_res = int(meta["phases"]), int(meta["operator_res"])
    M = min(N_MODES, phases // 2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inst = load_instrument(a.calibration, dev)
    eta = model_error_scale(inst)
    op = CodeOperator(inst, psi_grid(phases), res=op_res, config=RENDER, device=dev)
    # the network runs on the CPU and the operator on the GPU, as reconstruct_lpd.py runs
    # them: the corpus tensors and the sampler's state stay on one device throughout
    net = LPDFlow.from_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    net.eval()
    C = len(cameras())
    tag, mask = geometry_tags(), torch.ones(1, C)
    geoms = list(range(C))

    n_val = max(0, min(a.val_bodies, len(data.codes) - 1))
    is_val = np.isin(data.index.numpy(), held_out(int(meta["bodies"]), n_val))
    val_rows = np.nonzero(is_val)[0]
    if len(val_rows) == 0:
        raise SystemExit("no held-out bodies; train with --val-bodies above zero")
    carved = carving(data)
    rows = val_rows[spread_over(carved[val_rows], a.bodies)]
    print(f"  {len(rows)} held-out bodies, carved {carved[rows].min():.2f}-{carved[rows].max():.2f}",
          flush=True)

    results = []
    for b in rows.tolist():
        t0 = time.time()
        gen = torch.Generator().manual_seed(a.seed * 100003 + int(data.index[b]))
        R = float(data.radius[b])
        support, sup_true = data.support[b], data.support_true[b]
        sigma = noise_sigma(1, generator=gen)[0]                                  # (G, 2)
        clean = data.curves[b]
        curves = (clean + sigma[..., None] * torch.randn(clean.shape, generator=gen)
                  + eta[..., None] * smooth_noise_like(clean, generator=gen))
        scale = torch.sqrt(sigma ** 2 + eta ** 2)

        cond = cond_channels(support)
        codes = net.sample(make_resid_fn(net, op, curves, scale, mask, M, cond, support, R),
                           tag.expand(a.samples, -1, -1), mask.expand(a.samples, -1), cond, R,
                           batch=a.samples, n_steps=a.steps, churn=a.churn)
        fits = []
        for i in range(a.samples):
            if a.polish_steps > 0:
                codes[i], _, chi, _ = polish(net, op, codes[i], support, R, curves, scale, geoms,
                                             a.polish_steps)
            fits.append(chi if a.polish_steps > 0 else float("nan"))
        raw = net.codec.decode(codes)

        # the truth: the hull support with the body's own amplitudes, posed as the draws are
        true_code = data.codes[b].clone()
        true_code[:N_DIR] = 0.0
        truth = op.mesh(sup_true, true_code, res=a.res)
        if truth is None:
            print(f"  body {int(data.index[b])}: the truth has no surface at this resolution, "
                  f"skipped", flush=True)
            continue
        tv = fit_to_cylinder(apply_constraints(truth[0].cpu().numpy(), 1.0), R)
        tf = truth[1].cpu().numpy()

        meshes, chis = [], []
        for i in range(a.samples):
            v, f, _ = decode(op, raw[i], support, res=a.res)
            if v is None:
                continue
            chis.append(float(mesh_misfit_by_geom(op, v, f, R, curves, scale).pow(2).mean().sqrt()))
            meshes.append((fit_to_cylinder(v, R), f))
        if len(meshes) < 2:
            print(f"  body {int(data.index[b])}: fewer than two draws decoded, skipped", flush=True)
            continue
        n_draws = len(meshes)
        ext = max(float(np.abs(mv).max()) for mv, _ in meshes + [(tv, tf)]) * 1.05
        occs = [mesh_occupancy(mv, mf, 64, ext) for mv, mf in meshes]
        extra = consensus_bodies(occs, ext, R)
        levels = [lv for lv, _, _ in extra]
        candidates = meshes + [(mv, mf) for _, mv, mf in extra]
        occs = occs + [mesh_occupancy(mv, mf, 64, ext) for _, mv, mf in extra]
        outlines = [surface_points(v, f, n=a.side_points, seed=a.seed + i)
                    for i, (v, f) in enumerate(candidates)]
        vote = metric_medoid(occs, outlines, n_ref=n_draws)
        medoid = metric_medoid(occs[:n_draws], n_ref=n_draws)
        best_fit = int(np.argmin(chis))

        # every candidate against the truth, with both measures
        truth_occ = mesh_occupancy(tv, tf, 128, ext)
        truth_out = outline_set(surface_points(tv, tf, n=a.side_points, seed=a.seed), ext)
        scores = []
        for (v, f), pts in zip(candidates, outlines):
            d = dice(mesh_occupancy(v, f, 128, ext), truth_occ)
            s = measure_outlines(outline_set(pts, ext), truth_out)["assd_mean"]
            scores.append((d, s))
        picks = {"vote": vote, "best_fit": best_fit, "medoid": medoid,
                 "oracle": int(np.argmax([d for d, _ in scores]))}
        for lv in CONSENSUS_LEVELS:            # a level with no closed surface has no candidate
            picks[f"consensus_{lv:g}"] = n_draws + levels.index(lv) if lv in levels else None
        row = {"body": int(data.index[b]), "carved": float(carved[b]), "radius": R,
               "draws": n_draws, "misfit_sigma": chis, "polished_misfit_sigma": fits,
               "dice": {r: (None if k is None else scores[k][0]) for r, k in picks.items()},
               "side_assd": {r: (None if k is None else scores[k][1]) for r, k in picks.items()},
               "picked": picks, "seconds": time.time() - t0}
        results.append(row)
        print(f"  body {row['body']:>5} carved {row['carved']:.2f}: dice "
              + ", ".join(f"{r} {row['dice'][r]:.3f}" for r in RULES
                          if row['dice'][r] is not None)
              + f"  ({row['seconds']:.0f}s)", flush=True)

    if not results:
        raise SystemExit("no body could be checked")

    def mean_of(key, r):
        vals = [x[key][r] for x in results if x[key][r] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {r: {"dice": mean_of("dice", r), "side_assd": mean_of("side_assd", r),
                   "bodies": sum(x["dice"][r] is not None for x in results)} for r in RULES}
    print("\n  mean over the bodies:")
    for r in RULES:
        s = summary[r]
        if s["dice"] is not None:
            print(f"    {r:<15} dice {s['dice']:.4f}   side-view distance {s['side_assd']:.4f}"
                  f"   ({s['bodies']} bodies)")
    print("\n  'oracle' is the best candidate in hindsight, the ceiling of any rule.")
    out = {"ckpt": a.ckpt, "corpus": a.corpus, "samples": a.samples, "steps": a.steps,
           "polish_steps": a.polish_steps, "rules": list(RULES), "summary": summary,
           "bodies": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build N shape-library bodies to disk, in parallel, resumably.

    python scripts/build_shape_library.py --n 5000 --out dataset/generated/shapes \
        --workers 16

Each body is independent (its own `np.random.default_rng([seed, i])`), so generation is
embarrassingly parallel across processes; each is also saved to its own file the moment it's
made, so a killed or pre-empted run picks up wherever it left off on restart instead of
starting over -- the property that matters for a multi-hour job on a remote, possibly
pre-emptible, machine.

Output layout:

    {out}/body_00000.npz ... body_04999.npz     verts, faces, recipe, info (see library_io)
    {out}/manifest.json                          index -> file, base kind, convexity
    {out}/report.md                               validity + diversity summary

`scripts/fit_shapes.py --shapes-dir {out}` reads this directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.library_io import load_body, save_body, write_manifest        # noqa: E402
from hac26.library_metrics import (check_library, library_descriptors,   # noqa: E402
                                   pairwise_dice, participation_ratio)
from hac26.shape_library import (Body, LibrarySpec, body_from_convex_points,  # noqa: E402
                                 load_thingi10k, sample_body)


def _worker(args) -> dict:
    """Runs in a subprocess: generate body `i`, save it, return a small summary (not the
    body itself -- shipping meshes back through the pool's pickle channel is the thing that
    would make this slower than serial for a design this cheap per-body)."""
    i, seed, out_dir, spec_kw = args
    path = Path(out_dir) / f"body_{i:05d}.npz"
    if path.exists():
        try:
            b = load_body(str(path))
            return {"index": i, "file": path.name, "base": b.recipe["base"],
                     "convexity": float(b.info["convexity"]), "n_faces": int(len(b.faces)),
                     "skipped": True}
        except Exception:                                    # noqa: BLE001  corrupt: redo
            pass
    spec = LibrarySpec(**spec_kw)
    t0 = time.time()
    b = sample_body(np.random.default_rng([seed, i]), spec)
    save_body(str(path), b)
    return {"index": i, "file": path.name, "base": b.recipe["base"],
             "convexity": float(b.info["convexity"]), "n_faces": int(len(b.faces)),
             "skipped": False, "seconds": time.time() - t0}


def build(n: int, seed: int, out_dir: str, workers: int, spec: LibrarySpec,
          checkpoint_every: int = 100) -> list:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    spec_kw = {"res": spec.res, "extent": spec.extent, "radius": spec.radius,
               "convexity_max": spec.convexity_max, "base_weights": spec.base_weights,
               "n_modifiers": spec.n_modifiers, "mod_weights": spec.mod_weights,
               "max_attempts": spec.max_attempts}
    jobs = [(i, seed, out_dir, spec_kw) for i in range(n)]
    results = [None] * n
    t0 = time.time()
    n_done = 0
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_worker, jobs, chunksize=4):
            results[r["index"]] = r
            n_done += 1
            if n_done % checkpoint_every == 0 or n_done == n:
                done = [r for r in results if r is not None]
                write_manifest(out_dir, done)
                rate = n_done / max(time.time() - t0, 1e-9)
                eta_min = (n - n_done) / max(rate, 1e-9) / 60.0
                n_new = sum(1 for r in done if not r["skipped"])
                print(f"  {n_done:>5}/{n}  ({n_new} generated, {n_done - n_new} resumed)  "
                      f"{rate:.2f} bodies/s  ETA {eta_min:.1f} min", flush=True)
    return [r for r in results if r is not None]


def ingest_extra(out_dir: str, start_index: int, spec: LibrarySpec, seed: int,
                 thingi10k_dir: str | None, thingi10k_limit: int | None,
                 damit_points: str | None) -> list:
    """Append Thingi10K and/or DAMIT-derived bodies after the procedural ones.

    Both write into the SAME directory and manifest as the procedural bodies (contiguous
    indices), so `fit_shapes.py` sees one library regardless of source. Neither source is
    bundled with this repo -- see docs/shape_library.md -- so both are no-ops unless a path
    is given.
    """
    extra = []
    i = start_index
    if thingi10k_dir:
        print(f"[ingest] Thingi10K from {thingi10k_dir}", flush=True)
        for b in load_thingi10k(thingi10k_dir, limit=thingi10k_limit, spec=spec,
                                rng=np.random.default_rng([seed, "thingi"]),
                                add_modifiers=2):
            path = Path(out_dir) / f"body_{i:05d}.npz"
            save_body(str(path), b)
            extra.append({"index": i, "file": path.name, "base": b.recipe["base"],
                         "convexity": float(b.info["convexity"]),
                         "n_faces": int(len(b.faces)), "skipped": False})
            i += 1
        print(f"  ingested {len(extra)} Thingi10K bodies", flush=True)
    if damit_points:
        print(f"[ingest] DAMIT convex bases from {damit_points}", flush=True)
        z = np.load(damit_points, allow_pickle=True)
        pointsets = z["pointsets"] if "pointsets" in z else [z[k] for k in z.files]
        rng = np.random.default_rng([seed, "damit"])
        n_before = len(extra)
        for pts in pointsets:
            try:
                b = body_from_convex_points(np.asarray(pts, float), rng, spec=spec)
            except RuntimeError as e:
                print(f"  skipped one DAMIT shape: {e}", flush=True)
                continue
            path = Path(out_dir) / f"body_{i:05d}.npz"
            save_body(str(path), b)
            extra.append({"index": i, "file": path.name, "base": b.recipe["base"],
                         "convexity": float(b.info["convexity"]),
                         "n_faces": int(len(b.faces)), "skipped": False})
            i += 1
        print(f"  ingested {len(extra) - n_before} DAMIT-based bodies", flush=True)
    return extra


def write_report(out_dir: str, entries: list, spec: LibrarySpec, sample_n: int = 300,
                 seed: int = 0) -> None:
    """Validity + diversity summary, on a random sample for anything O(pairs) or O(voxel)."""
    from hac26.library_io import load_body

    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(entries), sample_n, replace=False)
          if len(entries) > sample_n else np.arange(len(entries)))
    sample = [load_body(str(Path(out_dir) / entries[i]["file"])) for i in idx]

    chk = check_library(sample, radius=spec.radius, convexity_max=spec.convexity_max)
    desc = library_descriptors(sample, n_probes=150, res=32)
    dice = pairwise_dice(sample, res=32, max_pairs=250, seed=seed)
    bases = {}
    for e in entries:
        bases[e["base"]] = bases.get(e["base"], 0) + 1
    conv = np.array([e["convexity"] for e in entries])

    lines = [
        f"# Shape library report: {len(entries)} bodies", "",
        f"Validity, measured on {len(sample)} bodies sampled from the full library:", "",
        "| check | pass |", "|---|---|",
    ]
    for k, v in chk["pass"].items():
        lines.append(f"| {k} | {v}/{len(sample)} |")
    lines += [
        "", f"Convexity (volume/hull volume) over all {len(entries)} bodies: "
            f"mean {conv.mean():.3f}, max {conv.max():.3f}, gate {spec.convexity_max}",
        "", f"Diversity, measured on the same {len(sample)}-body sample:", "",
        f"- PR(support)  = {participation_ratio(desc['support']):.2f}",
        f"- PR(concavity) = {participation_ratio(desc['concavity']):.2f}",
        f"- PR(combined)  = {participation_ratio(desc['combined']):.2f}",
        f"- pairwise Dice: mean {dice.mean():.3f}, std {dice.std():.3f}, "
        f"range [{dice.min():.3f}, {dice.max():.3f}]",
        "", "Base archetype counts over the full library:", "",
    ]
    for k, v in sorted(bases.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v}")
    text = "\n".join(lines) + "\n"
    with open(Path(out_dir) / "report.md", "w") as fh:
        fh.write(text)
    print("\n" + text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dataset/generated/shapes")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--res", type=int, default=64, help="marching-cubes grid resolution")
    ap.add_argument("--extent", type=float, default=1.6)
    ap.add_argument("--convexity-max", type=float, default=0.95)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--thingi10k-dir", default=None)
    ap.add_argument("--thingi10k-limit", type=int, default=None)
    ap.add_argument("--damit-points", default=None,
                    help="npz of point clouds, one array per DAMIT shape")
    ap.add_argument("--report-sample", type=int, default=300)
    a = ap.parse_args()

    spec = LibrarySpec(res=a.res, extent=a.extent, convexity_max=a.convexity_max)

    print(f"[1/3] procedural bodies: n={a.n}, workers={a.workers}, res={a.res}, "
          f"out={a.out}", flush=True)
    entries = build(a.n, a.seed, a.out, a.workers, spec, a.checkpoint_every)

    extra = ingest_extra(a.out, len(entries), spec, a.seed, a.thingi10k_dir,
                         a.thingi10k_limit, a.damit_points)
    entries += extra
    write_manifest(a.out, entries)

    print(f"[2/3] wrote {len(entries)} bodies to {a.out}", flush=True)
    print("[3/3] validity + diversity report", flush=True)
    write_report(a.out, entries, spec, sample_n=a.report_sample, seed=a.seed)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score every reconstruction under results/ with both public-model measures, into one CSV.

    python scripts/score_all.py
    python scripts/score_all.py --groups headline --out results/scores_headline.csv
    python scripts/score_all.py --models 3 --groups genetic_tuning --no-side-view

The two scorers on this branch each take one flat directory of `Asteroid<NN>.stl` files
(`hac26/scoring/voxel.py --stl ...`, `hac26/scoring/side_view.py --recon-dir ...`), which is
how scripts/run_pipeline.sh scores results/lpd. The genetic runs are not laid out that way --
theirs sit at `<run>/model_NN/<config>/stl/generation_NNNN.stl` -- and there are now enough of
them that a table per directory is not a scoreboard. This walks results/, scores every
candidate through those same two modules, and writes one tidy row per approach and model.

Both measures come from the modules themselves, not from a second implementation of them:
`voxel.score` is called as `hac26/scoring/voxel.py`'s own CLI calls it, and the outline
numbers come from `side_view.side_view_measure` on `side_view.surface_points`, as
`side_view.py`'s `main` builds them. The truth mesh is loaded once per model rather than once
per candidate (see `_cache_truth_loads`), which is the only thing done differently, and it
changes no arithmetic.

  voxel_dice          the challenge's voxel measure, 2|A and B|/(|A|+|B|) on a 128^3 parity
                      scan. Higher is better, 1 is perfect.
  assd_mean/_worst    symmetric mean nearest-neighbour distance between the two bodies'
                      side-view outlines, over 36 equatorial directions, in model units
                      (the body spans z = -1 to +1). Lower is better, 0 is perfect.
  hausdorff_mean      the worst single point on those outlines, same units.

Only models 1-3 have a released truth (hac26.conventions.PUBLIC_MODELS), so they are the whole
observable. Two reference rows put the others in scale and are scored the same way:
`truth_vs_truth` is the truth against a second surface sampling of itself -- the floor the
sampling and the pixel grid impose, below which no ASSD can go -- and `hull_of_truth` is the
truth's own convex hull, which is what a perfectly convex answer costs.

Alongside each score the row carries the mesh's topology -- face count, watertightness,
winding, signed volume, body count -- because the voxel measure is only meaningful on a closed
surface: the parity scan that decides "inside" is inverted by a hole all the way up the column
through it, so an open mesh scores as something other than itself rather than failing.
`voxel.score` warns; this records the fact next to the number so a suspect score can be read
as suspect. results/lpd/Asteroid02.stl is the case to know about -- it is open, and its
winding is inverted, so its volume integrates negative.

Everything measured here is CPU work: a numpy parity scan for the voxel measure, and trimesh
sampling plus scipy and skimage for the outlines. torch is pulled in transitively --
`hac26.shapes` imports `hac26.forward.convex_egi`, which imports it -- but no tensor is
allocated and no device is touched, so `torch.cuda.is_initialized()` is still False when a run
ends. There is no GPU in this path and none is wanted; the whole sweep runs on a laptop.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.conventions import PUBLIC_MODELS                       # noqa: E402
from hac26.data_io import public_stl                              # noqa: E402
from hac26.scoring import side_view, voxel                        # noqa: E402
from hac26.shapes import hull_mesh, rescale_touch_z               # noqa: E402
from hac26.stl_io import load_stl                                 # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

FIELDS = ["group", "approach", "model", "generation", "voxel_dice", "assd_mean",
          "assd_worst", "hausdorff_mean", "hausdorff_worst", "voxel_dice_sum",
          "n_models_scored", "n_faces", "watertight", "winding_ok", "volume", "n_bodies",
          "ga_internal_dice", "seconds", "stl"]


def _cache_truth_loads() -> None:
    """Load each STL once instead of once per candidate.

    `voxel.score` loads both meshes on every call, and the released truths are large --
    asteroid1.stl is 219 MB, and re-reading it for each of ~50 candidates is most of the
    runtime. Memoising the loader leaves `voxel.score`'s own arithmetic untouched: it is keyed
    on the path, and no file is written while this runs.
    """
    voxel.load_one_solid = functools.lru_cache(maxsize=16)(voxel.load_one_solid)


# --------------------------------------------------------------------------- candidates
def _run_dir(model_dir: Path) -> Path | None:
    """The single `<config>/` directory a reconstruct_genetic.py run writes under its output."""
    subs = [p for p in sorted(model_dir.iterdir()) if p.is_dir()] if model_dir.is_dir() else []
    return subs[0] if len(subs) == 1 else None


def _generations(run_dir: Path) -> dict[int, Path]:
    """{generation: stl} for one run, from either layout it writes (an stl/ subdir, or flat)."""
    d = run_dir / "stl" if (run_dir / "stl").is_dir() else run_dir
    return {int(p.stem.split("_")[1]): p for p in sorted(d.glob("generation_*.stl"))}


def _internal_dice(run_dir: Path) -> dict[int, float | None]:
    """The GA's own Dice per checkpoint, off its results.json.

    This is the GA's selection signal, not a score: reconstruct_genetic.py computes it on a
    `dice_resolution`^3 grid (16 in the final run, against the 128 used here), and for models
    4-10 it is null because there is no truth to compute it against.
    """
    f = run_dir / "results.json"
    if not f.exists():
        return {}
    cps = json.loads(f.read_text()).get("checkpoints", {})
    return {int(k.split("_")[1]): v.get("dice_score") for k, v in cps.items()}


def _model_of(run_dir: Path) -> int | None:
    f = run_dir / "results.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("config", {}).get("model")


def collect(models: set[int]) -> list[dict]:
    """Every scorable (group, approach, model, stl) under results/, in report order."""
    rows: list[dict] = []

    def add(group, approach, model, stl, ga_dice=None, generation=None):
        if model in models and Path(stl).exists():
            rows.append({"group": group, "approach": approach, "model": model,
                         "generation": generation, "ga_internal_dice": ga_dice,
                         "stl": str(Path(stl).relative_to(REPO))})

    # -- headline: the pipelines an answer would be chosen from
    for M in sorted(models):
        add("headline", "convex", M, RESULTS / "convex" / f"Asteroid{M:02d}.stl")
        add("headline", "lpd", M, RESULTS / "lpd" / f"Asteroid{M:02d}.stl")

    for base, tag in (("final_run", "genetic_final"), ("final_run_exact", "genetic_exact")):
        root = RESULTS / "genetic" / base
        if not root.is_dir():
            continue
        for M in sorted(models):
            # the mesh the GA actually started from: results/lpd decimated to ~2000 faces by
            # run_final_ga_all_models.py. Without it no GA row can be read, because any move
            # away from `lpd` mixes that decimation with the search itself.
            add("headline", f"{tag}_start", M,
                root / "start_meshes" / f"Asteroid{M:02d}_start.stl")
            run = _run_dir(root / f"model_{M:02d}")
            if run is None:
                continue
            gens, dice = _generations(run), _internal_dice(run)
            if not gens:
                continue
            last = max(gens)
            add("headline", tag, M, gens[last], dice.get(last), last)
            for g in sorted(gens)[:-1]:
                add(f"{tag}_trajectory", f"{tag}_gen{g:04d}", M, gens[g], dice.get(g), g)

    # -- the tuning and exploratory runs kept alongside them. toy_spherical_harmonics is
    # excluded on purpose: it carries its own synthetic truth.stl and is not a challenge body,
    # so there is nothing here to score it against.
    for sub in ("surfaces", "testing", "simpfaces", "example_surface_deformations"):
        base = RESULTS / "genetic" / sub
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("results.json")):
            run = f.parent
            M = _model_of(run)
            gens, dice = _generations(run), _internal_dice(run)
            if M is None or not gens:
                continue
            last = max(gens)
            add("genetic_tuning", f"{sub}/{run.name}", M, gens[last], dice.get(last), last)

    return rows


# --------------------------------------------------------------------------- measuring
def topology(stl: Path) -> dict:
    """Face count, closure, winding and signed volume of one candidate.

    `volume` is signed, so an inverted mesh reads negative: trimesh integrates over the faces
    as wound, and a body whose normals all point inward has the volume of its own complement.
    Neither measure here can see that -- the parity scan counts crossings and the silhouette
    rasteriser fills triangles, and both are winding-blind -- but a submission is scored by
    code whose voxeliser is not published, so it is worth recording.

    `n_bodies` is trimesh's `body_count`, not `len(split(...))`. On a mesh with non-manifold
    edges the face-adjacency graph that `split` walks falls apart into thousands of fragments
    that are not separate bodies: results/lpd/Asteroid02.stl splits into 4767 while its
    body_count is 1. The count that means something here is the one that does not.
    """
    import trimesh
    m = trimesh.load(str(stl), process=True)
    m.merge_vertices()
    return {"n_faces": len(m.faces), "watertight": bool(m.is_watertight),
            "winding_ok": bool(m.is_winding_consistent), "volume": float(m.volume),
            "n_bodies": int(m.body_count)}


@functools.lru_cache(maxsize=8)
def truth_points(model: int, data_dir: str):
    """The truth's posed mesh and a surface sampling of it, as side_view.main builds them."""
    tv, tf = load_stl(public_stl(data_dir, model))
    tv = rescale_touch_z(tv, tf, centre_xy=False)
    return tv, tf, side_view.surface_points(tv, tf)


def outline_scores(stl: Path, model: int, data_dir: str, n_dirs: int, res: int) -> dict:
    _, _, tp = truth_points(model, data_dir)
    rv, rf = load_stl(str(stl))
    rv = rescale_touch_z(rv, rf, centre_xy=False)
    r = side_view.side_view_measure(side_view.surface_points(rv, rf), tp, n_dirs, res)
    return {k: r[k] for k in ("assd_mean", "assd_worst", "hausdorff_mean", "hausdorff_worst")}


def reference_rows(models, data_dir, n_dirs, res, n) -> list[dict]:
    """The two rows that put an ASSD in scale: the sampling floor, and the convex ceiling."""
    out = []
    for M in sorted(models):
        tv, tf, tp = truth_points(M, data_dir)
        # the truth against a second sampling of itself: the floor the sampling and the pixel
        # grid impose. Its voxel Dice is 1 by construction and is left blank rather than
        # reported, since nothing was measured.
        r = side_view.side_view_measure(side_view.surface_points(tv, tf, seed=1), tp, n_dirs, res)
        out.append({"group": "reference", "approach": "truth_vs_truth", "model": M,
                    "stl": "", "generation": "", "n_faces": len(tf),
                    "watertight": "", "winding_ok": "",
                    "volume": "", "n_bodies": "", "ga_internal_dice": None, "voxel_dice": None,
                    **{k: r[k] for k in ("assd_mean", "assd_worst",
                                         "hausdorff_mean", "hausdorff_worst")}})
        # the truth's own convex hull: what a perfectly convex answer costs on this body
        hv, hf = hull_mesh(tv)
        import trimesh
        hull = trimesh.Trimesh(vertices=hv, faces=hf, process=True)
        hull.remove_unreferenced_vertices()
        hp = side_view.surface_points(np.asarray(hull.vertices), np.asarray(hull.faces))
        rh = side_view.side_view_measure(hp, tp, n_dirs, res)
        e = max(float(np.abs(tv).max()), float(np.abs(hull.vertices).max())) * 1.05
        from hac26.recon import dice, mesh_occupancy
        d = dice(mesh_occupancy(np.asarray(hull.vertices), np.asarray(hull.faces), n, e),
                 mesh_occupancy(tv, tf, n, e))
        out.append({"group": "reference", "approach": "hull_of_truth", "model": M,
                    "stl": "", "generation": "", "n_faces": len(hull.faces),
                    "watertight": bool(hull.is_watertight),
                    "winding_ok": bool(hull.is_winding_consistent),
                    "volume": float(hull.volume), "n_bodies": int(hull.body_count),
                    "ga_internal_dice": None, "voxel_dice": float(d),
                    **{k: rh[k] for k in ("assd_mean", "assd_worst",
                                          "hausdorff_mean", "hausdorff_worst")}})
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", type=int, default=list(PUBLIC_MODELS),
                    help="public models to score; only these have a released truth")
    ap.add_argument("--groups", nargs="+", default=None,
                    help="restrict to these groups: headline, reference, "
                         "genetic_final_trajectory, genetic_exact_trajectory, genetic_tuning")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--n", type=int, default=128, help="voxel grid side for the Dice measure")
    ap.add_argument("--n-dirs", type=int, default=36,
                    help="side-view directions, as hac26/scoring/side_view.py's own default")
    ap.add_argument("--res", type=int, default=512, help="silhouette image size in pixels")
    ap.add_argument("--no-side-view", action="store_true",
                    help="voxel Dice only -- about half the cost and the measure that "
                         "separates the pipelines")
    ap.add_argument("--out", default="results/scores_all.csv")
    a = ap.parse_args()

    bad = [m for m in a.models if m not in PUBLIC_MODELS]
    if bad:
        raise SystemExit(f"models {bad} have no released truth; public models are "
                         f"{list(PUBLIC_MODELS)}")
    missing = [public_stl(a.data_dir, M) for M in a.models
               if not Path(public_stl(a.data_dir, M)).exists()]
    if missing:
        raise SystemExit(f"missing truth shapes, nothing to score against: {missing}")

    _cache_truth_loads()
    models = set(a.models)
    rows = collect(models)
    if not a.no_side_view and (a.groups is None or "reference" in a.groups):
        rows += reference_rows(models, a.data_dir, a.n_dirs, a.res, a.n)
    if a.groups:
        rows = [r for r in rows if r["group"] in set(a.groups)]
    if not rows:
        raise SystemExit("nothing to score: no reconstruction found under results/")

    print(f"scoring {len(rows)} meshes over models {a.models} "
          f"(voxel {a.n}^3"
          f"{'' if a.no_side_view else f', outlines {a.n_dirs} dirs at {a.res}px'})\n",
          flush=True)
    for r in rows:
        if not r["stl"]:                        # a reference row, already measured
            continue
        t0 = time.time()
        stl = REPO / r["stl"]
        r.update(topology(stl))
        r["voxel_dice"] = voxel.score(str(stl), r["model"], a.data_dir, a.n)
        if not a.no_side_view:
            r.update(outline_scores(stl, r["model"], a.data_dir, a.n_dirs, a.res))
        r["seconds"] = round(time.time() - t0, 1)
        print(f"  {r['approach']:<46} m{r['model']}  dice {r['voxel_dice']:.4f}"
              + (f"  assd {r['assd_mean']:.4f}" if not a.no_side_view else "")
              + f"  ({r['seconds']:.0f}s)", flush=True)

    # per-approach totals, repeated onto every row of that approach so the CSV pivots without
    # a second file. An approach missing a model is reported by n_models_scored, not hidden:
    # a sum over two models is not comparable with a sum over three.
    for approach in {r["approach"] for r in rows}:
        grp = [r for r in rows if r["approach"] == approach]
        have = [r for r in grp if r.get("voxel_dice") is not None]
        for r in grp:
            r["voxel_dice_sum"] = round(sum(x["voxel_dice"] for x in have), 6) if have else None
            r["n_models_scored"] = len(have)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["group"] != "headline",
                                             -(r.get("voxel_dice_sum") or -1),
                                             r["approach"], r["model"])):
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items() if k in FIELDS})
    print(f"\nwrote {out}")

    print(f"\n{'approach':<46} {'models':>6} {'dice sum':>9} {'dice mean':>10}"
          + ("" if a.no_side_view else f" {'assd mean':>10}"))
    seen: dict[str, list] = {}
    for r in rows:
        seen.setdefault(r["approach"], []).append(r)
    ranked = sorted(seen.items(), key=lambda kv: -(kv[1][0].get("voxel_dice_sum") or -1))
    for approach, grp in ranked:
        have = [x for x in grp if x.get("voxel_dice") is not None]
        if not have:
            continue
        s = grp[0]["voxel_dice_sum"]
        line = f"{approach:<46} {len(have):>6} {s:>9.4f} {s / len(have):>10.4f}"
        if not a.no_side_view:
            line += f" {np.mean([x['assd_mean'] for x in grp]):>10.4f}"
        print(line)
    print(f"\nvoxel Dice is 1 per model at best ({len(a.models)} over models {a.models}); "
          f"ASSD is a distance, so lower is better and 0 is perfect")


if __name__ == "__main__":
    main()

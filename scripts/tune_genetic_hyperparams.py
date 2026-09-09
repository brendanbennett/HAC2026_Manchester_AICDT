#!/usr/bin/env python3
"""Tune the surface-deformation genetic algorithm's own hyperparameters, against real shapes,
by Optuna search on the Dice score against ground truth.

Motivation and the two-objective structure
--------------------------------------------
scripts/reconstruct_genetic.py's --mode surface fitness (hac26.genetic_utils.surface_fitness)
only ever sees lightcurve residual: it has no access to the true shape, because in a real
reconstruction there isn't one. But *this* script is run against library bodies whose true
shape IS known, purely so the GA's own hyperparameters (population size, mutation schedule,
control-point count, deformation footprint and amplitude, ...) can be picked well.

So there are two separate objectives here, at two nested levels, and they are not the same:
  - inner (unchanged): the GA's own search inside one run is still guided by lightcurve
    residual only, exactly as scripts/reconstruct_genetic.py already does. This script does
    not touch that.
  - outer (this script's job): across many (shape, hyperparameter-setting) trials, this
    script tunes the *hyperparameters* by the mean Dice score of the GA's final answer
    against the true mesh -- a comparison only possible because these are library bodies
    with a known ground truth, standing in for the real asteroids whose truth is secret.

Test shapes: convex starts, not the LPD's
-------------------------------------------
Each test body comes from a shape library (scripts/build_shape_library.py's output -- the
same bodies scripts/fit_shapes.py and scripts/build_corpus.py fit and render for the flow
stage). The GA needs a convex starting mesh to deform, standing in for what the trained LPD
convex stage would hand it; here that stand-in is simply the true mesh's own convex hull
(hac26.shapes.hull_mesh), not an actual LPD checkpoint's output -- cheaper, needs no trained
model, and every library body already has one by construction. If this tuning is later run
as a refinement stage after a real LPD reconstruction, swap the hull for that reconstruction's
own mesh as the --mode surface starting point; nothing else about the search changes.

Target curves come from the true (non-convex) mesh via the same convex-only forward operator
(hac26.shapes.mesh_curves_convex) the GA's own fitness already uses -- consistent with, not
harder than, what the GA is actually asked to match in scripts/reconstruct_genetic.py.

Cost
----
One trial runs the full GA (population_size x n_generations lightcurve evaluations) once per
test shape, then scores Dice once per shape. So cost scales as roughly

    n_trials x n_shapes x population_size x n_generations

and population_size/n_generations are themselves being searched (SEARCH_SPACE below), so a
trial's cost varies. A coarse Optuna pruner (median, after each shape) cuts off trials that
are already doing badly before they reach every shape, but the search space bounds and
--n-trials/--n-shapes still set the ceiling. Start small (e.g. --n-trials 20 --n-shapes 3) to
see the per-trial wall-clock before committing to a long run; this script has not been run
here, so no empirical timing is given.

Running it
----------
    python scripts/tune_genetic_hyperparams.py --shapes-dir dataset/generated/shapes \\
        --n-shapes 6 --n-trials 100

Writes runs/genetic_tuning/best_params.json (the winning hyperparameters and their mean
Dice) and trials.csv (every trial, for inspection). Uses a SQLite-backed Optuna study
(--storage) by default, so a killed run resumes into the same study on rerun, and real
parallelism is available by running this script again concurrently against the same
--storage rather than via --n-jobs (which threads, of limited use for CPU-bound work).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.genetic_utils import (                                                 # noqa: E402
    ExactForwardModel,
    build_surface_influence_matrix,
    deform_surface,
    render_curves,
    sample_surface_control_points,
    surface_fitness,
)
from hac26.geometry import build_cameras                                          # noqa: E402
from hac26.library_io import load_library_dir                                     # noqa: E402
from hac26.scoring.voxel import prepare_truth, score_mesh                         # noqa: E402
from hac26.shapes import hull_mesh                                                # noqa: E402
from hac26.solvers.genetic import GeneticSolver                                   # noqa: E402

# The search space. Bounds cover the historically-explored surface-mode range
# (results/genetic/surfaces/, results/genetic/example_surface_deformations/ used population
# 100-304, generations 100-403, mutation_scale 0.05/0.5/0.75). population_size and
# n_generations both multiply directly into per-trial cost -- at ~34ms per lightcurve
# evaluation (measured on one ~8k-face body), the upper corner (population=300,
# generations=400) is roughly a shape-hour on its own, so a trial landing there is slow, not
# broken; narrow the bounds instead of widening n_trials if that turns out to dominate wall
# clock in practice. Narrow or widen any of these and rerun rather than editing deeper in the
# file.
SEARCH_SPACE = {
    "mutation_scale": dict(low=0.02, high=0.8, log=True),
    "population_size": dict(low=30, high=300),
    "n_generations": dict(low=20, high=400),
    "mutation_decay": dict(low=0.9, high=1.0),
    "deform_width": dict(low=0.05, high=0.6),
    "max_amp": dict(low=0.1, high=0.8),
    "n_cpts": dict(low=10, high=120),
}


def convex_start(verts: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """The convex-hull stand-in for the LPD's output: hull_mesh keeps every input point, only
    some of which are referenced by the hull's own faces (hac26.shapes.hull_mesh's
    docstring), so the unreferenced interior ones are dropped here. Left in, farthest-point
    sampling in sample_surface_control_points could pick one as a "control point" -- its
    geodesic distance to every real hull-surface vertex is infinite, so it would sit in the
    parameter vector doing nothing."""
    hv, hf = hull_mesh(verts)
    mesh = trimesh.Trimesh(vertices=hv, faces=hf, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def load_test_shapes(shapes_dir: str, n_shapes: int, seed: int, m: int, cameras: list,
                     curve_types: list, dice_resolution: int, simplify_faces: int | None,
                     forward: "ExactForwardModel | None" = None) -> list[dict]:
    """Everything about each test body that does NOT depend on the hyperparameters being
    tuned, computed once: the convex-hull starting mesh, its own target lightcurves (from the
    true mesh, module docstring, rendered with whichever forward model this run uses --
    render_curves), its prepared truth for repeated Dice scoring
    (hac26.scoring.voxel.prepare_truth), and its characteristic length, which sets the
    physical scale of deform_width/max_amp exactly as scripts/reconstruct_genetic.py does."""
    bodies = load_library_dir(shapes_dir, n=n_shapes, seed=seed, with_entries=True)
    if not bodies:
        raise SystemExit(f"no bodies in {shapes_dir} -- run scripts/build_shape_library.py "
                         f"first")
    out = []
    for i, (verts, faces, entry) in enumerate(bodies):
        start_mesh = convex_start(verts, faces)
        target_curves = render_curves(verts, faces, cameras=cameras, m=m,
                                      curve_types=curve_types, forward=forward)
        truth = prepare_truth(verts, faces, n=dice_resolution, simplify_faces=simplify_faces)
        out.append({
            "index": i,
            "family": str(entry.get("base", "unknown")),
            "start_mesh": start_mesh,
            "target_curves": target_curves,
            "truth": truth,
            "characteristic_length": float(np.max(start_mesh.extents)),
        })
        print(f"  shape {i} ({entry.get('base', 'unknown')}): {len(verts)} true verts, "
              f"{len(start_mesh.vertices)} hull verts", flush=True)
    return out


def run_ga_once(shape: dict, params: dict, cameras: list, curve_types: list, m: int,
                cpts_seed: int, ga_seed: int,
                forward: "ExactForwardModel | None" = None) -> tuple[float, trimesh.Trimesh]:
    """One GA reconstruction of one shape at one hyperparameter setting, from its convex-hull
    start to its own target curves (surface_fitness, unchanged -- lightcurve-only). Returns
    (Dice of the final mesh against the true mesh, the final mesh itself).

    cpts_seed is fixed across trials (not varied with ga_seed): sample_surface_control_points
    is prefix-stable in its seed (the same seed's first k points do not depend on how many
    total points are requested), so fixing it means n_cpts trials differ only in how many of
    the same growing sequence of points they use, not in which points too -- one less source
    of trial-to-trial noise for Optuna to search through.
    """
    mesh = shape["start_mesh"]
    n_cpts = params["n_cpts"]
    control_point_indices = sample_surface_control_points(mesh, n_points=n_cpts,
                                                           seed=cpts_seed)

    characteristic_length = shape["characteristic_length"]
    sigma = params["deform_width"] * characteristic_length
    influence = build_surface_influence_matrix(mesh, control_point_indices, sigma=sigma)

    max_amplitude = params["max_amp"] * characteristic_length
    bounds = np.array([[-max_amplitude, max_amplitude]] * n_cpts)
    initial_params = np.zeros(n_cpts)

    def fitness(p):
        return surface_fitness(initial_mesh=mesh, params=p, influence=influence,
                               target_curves=shape["target_curves"], cameras=cameras, m=m,
                               curve_types=curve_types, forward=forward)

    solver = GeneticSolver(
        fitness_fn=fitness,
        initial_params=initial_params,
        mutation_scale=params["mutation_scale"],
        population_size=params["population_size"],
        n_parents=params["n_parents"],
        n_generations=params["n_generations"],
        mutation_decay=params["mutation_decay"],
        bounds=bounds,
        seed=ga_seed,
    )
    result = solver.run()

    final_mesh = deform_surface(mesh, result.best_params, influence)
    dice = score_mesh(final_mesh.vertices, final_mesh.faces, shape["truth"])
    return float(dice), final_mesh


def make_objective(shapes: list[dict], cameras: list, curve_types: list, m: int, seed: int,
                   forward: "ExactForwardModel | None" = None):
    """The Optuna objective: mean Dice across every test shape at one sampled hyperparameter
    setting. Reports the running mean after each shape so the pruner can cut off a trial
    that is already doing badly without paying for the remaining shapes."""

    def objective(trial: optuna.trial.Trial) -> float:
        population_size = trial.suggest_int("population_size", **SEARCH_SPACE["population_size"])
        n_parents = trial.suggest_int("n_parents", 4, max(4, population_size // 2))
        params = {
            "mutation_scale": trial.suggest_float("mutation_scale", **SEARCH_SPACE["mutation_scale"]),
            "population_size": population_size,
            "n_parents": n_parents,
            "n_generations": trial.suggest_int("n_generations", **SEARCH_SPACE["n_generations"]),
            "mutation_decay": trial.suggest_float("mutation_decay", **SEARCH_SPACE["mutation_decay"]),
            "deform_width": trial.suggest_float("deform_width", **SEARCH_SPACE["deform_width"]),
            "max_amp": trial.suggest_float("max_amp", **SEARCH_SPACE["max_amp"]),
            "n_cpts": trial.suggest_int("n_cpts", **SEARCH_SPACE["n_cpts"]),
        }

        dice_scores = []
        for shape in shapes:
            ga_seed = seed + 1000 * (trial.number + 1) + shape["index"]
            dice, _ = run_ga_once(shape, params, cameras, curve_types, m,
                                  cpts_seed=seed, ga_seed=ga_seed, forward=forward)
            dice_scores.append(dice)

            trial.report(float(np.mean(dice_scores)), step=shape["index"])
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(dice_scores))

    return objective


def write_trials_csv(study: optuna.Study, path: Path) -> None:
    """One row per trial: number, state, mean Dice, every hyperparameter, and timing. Written
    by hand over `study.trials` rather than optuna's own trials_dataframe(), which needs
    pandas -- not otherwise a dependency of this project, and not worth adding for a CSV
    dump this small."""
    rows = study.trials
    fields = ["number", "state", "value"] + list(SEARCH_SPACE) + [
        "datetime_start", "duration_seconds"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for t in rows:
            duration = t.duration.total_seconds() if t.duration is not None else None
            writer.writerow({
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                **{k: t.params.get(k) for k in SEARCH_SPACE},
                "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
                "duration_seconds": duration,
            })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes-dir", default="dataset/generated/shapes",
                    help="a library written by scripts/build_shape_library.py")
    ap.add_argument("--n-shapes", type=int, default=6,
                    help="library bodies tested per hyperparameter setting")
    ap.add_argument("--n-trials", type=int, default=100, help="Optuna trials to run")
    ap.add_argument("--m", type=int, default=50, help="lightcurve phase samples")
    ap.add_argument("--dice-resolution", type=int, default=16,
                    help="voxel grid resolution for Dice scoring")
    ap.add_argument("--simplify-faces", type=int, default=None,
                    help="decimate true meshes to this many faces before voxelising, for "
                         "cheaper repeated Dice evaluation on dense library bodies")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--study-name", default="genetic_ga_tuning")
    ap.add_argument("--storage", default="sqlite:///runs/genetic_tuning.db",
                    help="Optuna storage URL; a killed run resumes into the same study on "
                         "rerun, and running this script again concurrently against the same "
                         "URL is how to parallelise across processes")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallel trials inside this one process (Optuna threads these -- "
                         "limited value on CPU-bound work; prefer multiple invocations "
                         "against the same --storage for real parallelism)")
    ap.add_argument("--output-dir", default="runs/genetic_tuning")
    ap.add_argument("--forward-model", choices=["convex", "exact"], default="convex",
                    help="convex (default): the cheap per-facet kernel every candidate AND "
                         "every target curve is rendered with (hac26.forward.convex_egi); "
                         "blind to cast shadows/interreflection, exact only for a truly "
                         "convex candidate. exact: the flow-matching stage's own "
                         "hac26.forward.mesh.exact.ExactForward -- real shadows and a "
                         "radiosity interreflection solve, at real cost: at least an order "
                         "of magnitude slower per candidate, multiplying directly into the "
                         "cost structure in the module docstring above, more still on a "
                         "machine with no CUDA/nvdiffrast (--exact-backend software).")
    ap.add_argument("--calibration", default="models/instrument_calibration.pt",
                    help="Instrument written by scripts/calibrate.py; only used with "
                         "--forward-model exact")
    ap.add_argument("--exact-device", default="cpu", help="torch device for --forward-model "
                    "exact (cpu/cuda/mps)")
    ap.add_argument("--exact-backend", default=None,
                    help="rasteriser backend for --forward-model exact: 'nvdiffrast' (needs "
                         "a CUDA build) or 'software' (the pure-torch stand-in this repo "
                         "documents as being for tests, not real runs -- the only option "
                         "without nvdiffrast). Defaults to nvdiffrast, or software if the "
                         "HAC26_SOFTWARE_RASTER env var is set.")
    ap.add_argument("--exact-radiosity-faces", type=int, default=200,
                    help="patches the interreflection solve uses; RenderConfig's own default "
                         "is 600, expensive per call in a GA's inner loop")
    ap.add_argument("--exact-res", type=int, nargs=2, default=[108, 192],
                    metavar=("HEIGHT", "WIDTH"), help="sensor resolution before supersampling")
    a = ap.parse_args()

    cameras = build_cameras()
    curve_types = ["intensity"] * len(cameras)

    forward = None
    if a.forward_model == "exact":
        print(f"[forward model] exact (hac26.forward.mesh.exact.ExactForward), "
              f"device={a.exact_device}, backend={a.exact_backend or 'nvdiffrast (default)'}",
              flush=True)
        forward = ExactForwardModel(
            a.calibration, m=a.m, device=a.exact_device, backend=a.exact_backend,
            radiosity_faces=a.exact_radiosity_faces,
            height=a.exact_res[0], width=a.exact_res[1],
        )

    print(f"[1/3] preparing {a.n_shapes} test shapes from {a.shapes_dir}", flush=True)
    shapes = load_test_shapes(a.shapes_dir, a.n_shapes, a.seed, a.m, cameras, curve_types,
                              a.dice_resolution, a.simplify_faces, forward=forward)

    if a.storage.startswith("sqlite:///"):
        Path(a.storage.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] running {a.n_trials} Optuna trials (storage={a.storage})", flush=True)
    study = optuna.create_study(
        study_name=a.study_name,
        storage=a.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=a.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(make_objective(shapes, cameras, curve_types, a.m, a.seed, forward=forward),
                   n_trials=a.n_trials, n_jobs=a.n_jobs)

    print(f"[3/3] writing results to {a.output_dir}", flush=True)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "best_params.json", "w") as fh:
        json.dump({"mean_dice": study.best_value, "params": study.best_params}, fh, indent=2)
    write_trials_csv(study, out / "trials.csv")

    print(f"\nbest mean Dice: {study.best_value:.4f}")
    print("best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

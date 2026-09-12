#!/usr/bin/env python3
"""Run the genetic algorithm, at hyperparameters found by scripts/tune_genetic_hyperparams.py,
on all ten challenge models -- the production step after scripts/run_tuning_laptop.sh.

Unlike the tuning script (see its own module docstring), this DOES save STL files and
checkpoints: each model's scripts/reconstruct_genetic.py invocation runs with its normal
checkpointing (hac26.genetic_utils.save_checkpoint_results), since this is a one-off
production run whose entire point is the reconstructed shapes, not thousands of throwaway
tuning trials whose meshes were deliberately discarded after scoring.

Starting point: each model's own results/lpd/AsteroidNN.stl -- the flow-matching stage's own
reconstruction, exactly the "final refinement stage after the LPD" role this GA is meant for.
Those files are dense (tens of thousands of faces, the LPD's own mesh-extraction
resolution); the tuned hyperparameters were calibrated against much coarser convex hulls
(hundreds to a few thousand faces -- scripts/tune_genetic_hyperparams.py's convex_start), so
each is decimated to --target-faces first: measured ~240ms per lightcurve evaluation at full
resolution on Asteroid04.stl (20,052 faces) versus ~40ms decimated to ~2,000, and the tuned
deform_width/max_amp/n_cpts (control-point footprint and count, calibrated at that coarser
scale) apply most consistently at a comparable one.

Only models 1-3 have a released truth shape (hac26.conventions.PUBLIC_MODELS), so only those
get a real Dice score; 4-10 report dice=n/a -- scripts/reconstruct_genetic.py's own
secret-model handling, not a failure.

The ten models are fully independent GA runs (different truth/initial mesh, no shared state),
and each reconstruct_genetic.py subprocess is single-threaded CPU-bound work -- the same
"separate OS processes, not threads" reasoning as scripts/run_tuning_laptop.sh. --workers N
runs up to N of them concurrently via a thread pool whose threads just block on each
subprocess, so real parallelism comes from the OS processes, not from Python threading.
Default is cpu_count() - 2, leaving headroom the way run_tuning_laptop.sh does. Each model's
console output goes to its own log (<output-dir>/model_NN.log) instead of interleaving on
this process's stdout -- monitor progress with `tail -f` on one of those, or watch
<output-dir>/model_NN/*/results.json for the "checkpoints" written after each of the 5
checkpoint generations (0, gen//4, gen//2, 3*gen//4, gen) reconstruct_genetic.py saves.

    python scripts/run_final_ga_all_models.py \\
        --best-params results/genetic/hyperparameter_tuning/best_params.json \\
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import trimesh

from hac26.stl_io import load_stl

REPO_ROOT = Path(__file__).resolve().parents[1]


def decimate_lpd_mesh(model: int, target_faces: int, out_dir: Path) -> Path:
    """results/lpd/AsteroidNN.stl, decimated to about target_faces -- see the module
    docstring for why. Raises if the LPD reconstruction for this model does not exist."""
    path = REPO_ROOT / f"results/lpd/Asteroid{model:02d}.stl"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the LPD reconstruction for model {model} "
                         f"first (scripts/reconstruct_lpd.py)")
    # load_stl merges duplicate vertices, unlike trimesh.load(path, process=False); quadric
    # decimation needs real edge adjacency to collapse anything, and the mesh this produces
    # feeds straight into the GA's geodesic control-point influence graph (build_surface_
    # influence_matrix), which is broken on unmerged triangle-soup -- see hac26.genetic_utils.
    verts, faces = load_stl(str(path))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    n_before = len(mesh.faces)
    if n_before > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    out = out_dir / f"Asteroid{model:02d}_start.stl"
    mesh.export(out)
    print(f"  model {model}: {n_before} faces -> {len(mesh.faces)} faces, start mesh: {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--best-params", default="results/genetic/hyperparameter_tuning/best_params.json",
                    help="written by scripts/tune_genetic_hyperparams.py")
    ap.add_argument("--models", type=int, nargs="+", default=list(range(1, 11)),
                    help="challenge model numbers to run; default is all ten")
    ap.add_argument("--target-faces", type=int, default=2000,
                    help="decimate each model's results/lpd/AsteroidNN.stl to about this "
                         "many faces before running the GA on it -- see the module docstring")
    ap.add_argument("--forward-model", choices=["convex", "exact"], default="convex")
    ap.add_argument("--calibration", default="models/instrument_calibration.pt",
                    help="only used with --forward-model exact")
    ap.add_argument("--exact-device", default="cpu",
                    help="torch device for --forward-model exact (cpu/cuda/mps)")
    ap.add_argument("--exact-backend", default=None,
                    help="rasteriser backend for --forward-model exact: nvdiffrast (needs a "
                         "CUDA build, see scripts/setup_toolchain.sh) or software; default "
                         "picks nvdiffrast unless HAC26_SOFTWARE_RASTER is set")
    ap.add_argument("--exact-radiosity-faces", type=int, default=600,
                    help="patches the interreflection solve uses, for --forward-model exact. "
                         "reconstruct_genetic.py's own default (200) trades fidelity for "
                         "speed in a tuning search's inner loop; measured on a real GPU, "
                         "_prepare()'s cost (where this matters) is under 1%% of a call's "
                         "total time at production frame counts, so this defaults to "
                         "RenderConfig's own full 600 here instead -- this run's whole point "
                         "is the reconstructed shapes, so there is no real reason to trade "
                         "fidelity away for a saving that small")
    ap.add_argument("--output-dir", default="results/genetic/final_run")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--m", type=int, default=50, help="lightcurve phase samples")
    ap.add_argument("--dice-resolution", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None,
                    help="run up to this many models' GA concurrently, as separate "
                         "subprocesses -- see the module docstring. Default: "
                         "cpu_count()-2 for --forward-model convex; 1 for exact, since "
                         "several processes sharing one GPU (each opening its own CUDA "
                         "context, no batching across models -- see ExactForward.raw_curves) "
                         "adds VRAM pressure and contention rather than real throughput.")
    a = ap.parse_args()
    if a.workers is None:
        a.workers = 1 if a.forward_model == "exact" else max(1, (os.cpu_count() or 4) - 2)

    with open(a.best_params) as fh:
        best = json.load(fh)
    params = best["params"]
    print(f"Using hyperparameters tuned to mean Dice {best.get('mean_dice', 'n/a')} "
         f"from {a.best_params}:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    out_dir = Path(a.output_dir)
    start_dir = out_dir / "start_meshes"
    start_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDecimating each model's LPD reconstruction to ~{a.target_faces} faces "
         f"(module docstring explains why):")
    start_stls = {}
    failures = []
    for model in a.models:
        try:
            start_stls[model] = decimate_lpd_mesh(model, a.target_faces, start_dir)
        except SystemExit as exc:
            print(f"  model {model}: skipping: {exc}")
            failures.append(model)

    runnable = [m for m in a.models if m in start_stls]
    print(f"\nRunning GA for {len(runnable)} models, up to {a.workers} concurrently "
         f"(one subprocess per model; see module docstring for how to monitor):")

    def run_one(model: int) -> tuple[int, int, Path]:
        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "reconstruct_genetic.py"),
            "--mode", "surface",
            "--model", str(model),
            "--initial-stl", str(start_stls[model]),
            "--data-dir", a.data_dir,
            "--forward-model", a.forward_model,
            "--calibration", a.calibration,
            "--exact-device", a.exact_device,
            "--exact-radiosity-faces", str(a.exact_radiosity_faces),
            "--output-dir", str(out_dir / f"model_{model:02d}"),
            "--m", str(a.m),
            "--dice-resolution", str(a.dice_resolution),
            "--seed", str(a.seed),
            "--population-size", str(params["population_size"]),
            "--parents", str(params["n_parents"]),
            "--mutation-scale", str(params["mutation_scale"]),
            "--generations", str(params["n_generations"]),
            "--mutation-decay", str(params["mutation_decay"]),
            "--deform-width", str(params["deform_width"]),
            "--max-amp", str(params["max_amp"]),
            "--n-cpts", str(params["n_cpts"]),
        ]
        if a.exact_backend is not None:
            cmd += ["--exact-backend", a.exact_backend]
        log_path = out_dir / f"model_{model:02d}.log"
        with open(log_path, "w") as log_fh:
            result = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        return model, result.returncode, log_path

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futures = {pool.submit(run_one, model): model for model in runnable}
        for future in as_completed(futures):
            model, returncode, log_path = future.result()
            if returncode != 0:
                failures.append(model)
                print(f"  model {model} FAILED (exit {returncode}) -- see {log_path}")
            else:
                print(f"  model {model} done -- log: {log_path}")

    print(f"\nDone. {len(a.models) - len(failures)}/{len(a.models)} models completed.")
    if failures:
        print(f"Failed/skipped: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import time
import cloudpickle

from hac26.shapes import (
    icosphere,
    sh_mesh_from_coefficients,
)
from hac26.solvers.genetic import GeneticSolver
from hac26.geometry import build_cameras
from hac26.recon import dice
from hac26.scoring.voxel import score_mesh, prepare_truth
from hac26.data_io import load_model_curves
from hac26.genetic_utils import make_target_coefficients, sh_fitness, surface_fitness, \
                                save_shape_stl,  load_truth_mesh, \
                                plot_lightcurve_comparison, plot_genetic_convergence, \
                                save_checkpoint_results, load_initialisation_mesh, \
                                sample_surface_control_points, build_surface_influence_matrix, \
                                deform_surface, render_curves, ExactForwardModel
                                




def main():

    parser = argparse.ArgumentParser(
        description="Prototype genetic optimisation of an SH shape."
    )

    parser.add_argument(
        "--L",
        type=int,
        default=3,
        help="Maximum spherical-harmonic degree.",
    )

    parser.add_argument(
        "--subdiv",
        type=int,
        default=3,
        help="Icosphere subdivision level.",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--parents",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--m",
        type=int,
        default=50,
    )

    parser.add_argument(
    "--dice-resolution",
    type=int,
    default=16,
    help="Voxel resolution used for Dice evaluation.",
    )

    parser.add_argument(
    "--simplify-faces",
    type=int,
    default=None,
    help="Reduces size of input model for cheaper dice evaluation",
    )

    parser.add_argument(
    "--mutation-scale",
    type=float,
    default=0.05,
    help="Scale for mutations.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default='results/genetic',
    )

    parser.add_argument(
        "--model",
        type=int,
        default=None,
        help=(
            "Challenge asteroid model number. If omitted, use a synthetic SH target."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/raw",
        help="Challenge dataset directory.",
    )

    parser.add_argument(
        "--initial-stl",
        type=str,
        default=None,
        help="STL file to use as the initial shape.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='surface',
        help="Whether to use spherical harmonics (sh) or deform the surface (surface)",
    )

    parser.add_argument(
        "--deform-width",
        type=float,
        default=0.5,
        help="Sigma for dents/bulges in units of characteristic length",
    )

    parser.add_argument(
        "--n-cpts",
        type=int,
        default=100,
        help="Number of control points for surface deformation",
    )

    parser.add_argument(
        "--max-amp",
        type=float,
        default=0.5,
        help="Max amplitude for dents/bulges in units of characteristic length",
    )

    parser.add_argument(
        "--mutation-decay",
        type=float,
        default=0.98,
        help="Decay rate of mutation noise",
    )

    parser.add_argument(
        "--forward-model",
        choices=["convex", "exact"],
        default="convex",
        help=(
            "convex (default): the cheap per-facet Lommel-Seeliger+Lambert kernel "
            "(hac26.forward.convex_egi.kernel) -- exact for a convex body, but blind to cast "
            "shadows and interreflection, so increasingly approximate as a candidate becomes "
            "concave. exact: the flow-matching (LPD) stage's own forward model "
            "(hac26.forward.mesh.exact.ExactForward) -- real shadows and a radiosity "
            "interreflection solve, at real cost: at least an order of magnitude slower per "
            "candidate, much more on a machine with no CUDA/nvdiffrast (see --exact-backend)."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="models/instrument_calibration.pt",
        help="Instrument written by scripts/calibrate.py; only used with --forward-model exact",
    )
    parser.add_argument(
        "--exact-device",
        type=str,
        default="cpu",
        help="torch device for --forward-model exact (cpu/cuda/mps)",
    )
    parser.add_argument(
        "--exact-backend",
        type=str,
        default=None,
        help="rasteriser backend for --forward-model exact: 'nvdiffrast' (needs a CUDA "
             "build) or 'software' (the pure-torch stand-in this repo documents as being for "
             "tests, not real runs -- but the only option without nvdiffrast). Defaults to "
             "nvdiffrast, or software if the HAC26_SOFTWARE_RASTER env var is set.",
    )
    parser.add_argument(
        "--exact-radiosity-faces",
        type=int,
        default=200,
        help="patches the interreflection solve uses, for --forward-model exact; "
             "hac26.forward.mesh.exact.RenderConfig's own default is 600, expensive per call "
             "in a GA's inner loop, so this defaults lower",
    )
    parser.add_argument(
        "--exact-res",
        type=int,
        nargs=2,
        default=[108, 192],
        metavar=("HEIGHT", "WIDTH"),
        help="sensor resolution for --forward-model exact, before supersampling",
    )

    # load args
    args = parser.parse_args()

    # generate seed
    rng = np.random.default_rng(args.seed)

    # time computation
    start_time = time.perf_counter()


    # configure directory
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / (
            f"L{args.L}"
            f"_subdiv{args.subdiv}"
            f"_m{args.m}"
            f"_gen{args.generations}"
            f"_pop{args.population_size}"
            f"_parents{args.parents}"
            f"_mutscale{args.mutation_scale}"
            f"_seed{args.seed}"
            f"_diceres{args.dice_resolution}"
            f"_simpface{args.simplify_faces}"
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # make a output path for lightcurve files
    lc_dir = output_dir / Path('curves')
    lc_dir.mkdir(parents=True, exist_ok=True)

    # make an output path for the stl files
    stl_dir = output_dir / Path('stl')
    stl_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "surface" and args.initial_stl is None:
        raise ValueError(
            "--initial-stl is required when mode='surface'"
        )


    # --------------------------------------------------------
    # Load source of truth (if exists)
    # --------------------------------------------------------
    
    # camera angles
    cameras = build_cameras()

    # lightcurve types
    curve_types = ["intensity"] * len(cameras) #+ ["binary"] * len(cameras)

    # --------------------------------------------------------
    # Forward model: the cheap convex kernel (default), or the flow-matching stage's own
    # exact one (--forward-model exact). Built once and reused for every render this run
    # makes -- construction loads the calibrated Instrument and builds Rasterisers, both
    # too expensive to redo per candidate.
    # --------------------------------------------------------

    forward = None
    if args.forward_model == "exact":
        print(f"[forward model] exact (hac26.forward.mesh.exact.ExactForward), "
              f"device={args.exact_device}, backend={args.exact_backend or 'nvdiffrast (default)'}",
              flush=True)
        forward = ExactForwardModel(
            args.calibration, m=args.m, device=args.exact_device, backend=args.exact_backend,
            radiosity_faces=args.exact_radiosity_faces,
            height=args.exact_res[0], width=args.exact_res[1],
        )

    if args.model is None:
        
        # Synthetic problem
        print("Running synthetic SH reconstruction")

        target_coefficients = make_target_coefficients(
            rng,
            L=args.L,
        )

        target_vertices, target_faces = sh_mesh_from_coefficients(
            target_coefficients,
            L=args.L,
            subdiv=args.subdiv,
        )

        target_curves = render_curves(
            target_vertices,
            target_faces,
            cameras=cameras,
            m=args.m,
            curve_types=curve_types,
            forward=forward,
        )

        # TODO: use this ?
        target_mask = np.ones(
            target_curves.shape[0],
            dtype=np.float32,
            )

        np.save(
        lc_dir / "truth_curves.npy",
        target_curves,
        )

        # TODO: looks like some overlap of functionality here
        truth_mesh = save_shape_stl(
            target_coefficients,
            L=args.L,
            subdiv=args.subdiv,
            path=stl_dir / "truth.stl",
        ) 

        truth_mesh_voxelised = prepare_truth(
            truth_mesh.vertices,
            truth_mesh.faces,
            n=args.dice_resolution,
        )

    else:

        # Real asteroid
        print(f"Running real asteroid model {args.model}")

        lc_dict = load_model_curves(data_dir=data_dir,
                                          model_idx=args.model,
                                          m=args.m,
                                          renormalize=True)
        
        # TODO: When using the binary, use all 56 curves here
        target_curves = lc_dict['curves'][:28]

        np.save(
        lc_dir / "truth_curves.npy",
        target_curves,
        )

        # TODO: use the mask?
        target_mask = lc_dict['mask']

        truth_mesh = load_truth_mesh(
            args.model,
            data_dir,
        )

        truth_mesh_voxelised = prepare_truth(
            truth_mesh.vertices,
            truth_mesh.faces,
            n=args.dice_resolution,
            simplify_faces=args.simplify_faces
        )

    # ------------------------------------------------------------
    # Load initialisation
    # ------------------------------------------------------------

    # select which optimisation mode to use
    
    if args.mode == "sh":

        # initialise parameters and mutation bounds
        n_coefficients = args.L * (args.L + 2)
        initial_params = np.zeros(n_coefficients)
        bounds = np.array(
            [
                [-0.5, 0.5]
                for _ in range(n_coefficients)
            ]
        )

        print(f"Using L={args.L}, ({n_coefficients} SH coefficients)")

        # quantify fitness from lightcurve
        def fitness(coefficients):
            return sh_fitness(
                coefficients=coefficients,
                target_curves=target_curves,
                L=args.L,
                subdiv=args.subdiv,
                cameras=cameras,
                m=args.m,
                curve_types=curve_types,
                forward=forward,
            )


    elif args.mode == "surface":


        # load initial mesh
        initial_mesh = load_initialisation_mesh(args)

        # weld vertices
        initial_mesh.merge_vertices()

        # select number of points to deform on surface 
        n_control_points = args.n_cpts
        control_point_indices = sample_surface_control_points(
            initial_mesh,
            n_points=n_control_points,
            seed=args.seed,
        )

        # set bounds on mutations
        characteristic_length = np.max(initial_mesh.extents)

        # Width of each deformation
        sigma = args.deform_width* characteristic_length

        influence = build_surface_influence_matrix(
            initial_mesh,
            control_point_indices,
            sigma=sigma,
        )


        params = np.zeros(n_control_points)
        params[0] = 0.05 * characteristic_length

        ##### to test a single bump ######
        test_mesh = deform_surface(
            initial_mesh,
            params,
            influence,
        )

        test_mesh.export(output_dir / "test_single_bump.stl")
        #############################################

        initial_params = np.zeros(
            n_control_points
        )

        # set mutation bounds
        max_amplitude = args.max_amp * characteristic_length
        bounds = np.array(
            [
                [-max_amplitude, max_amplitude]
                for _ in range(n_control_points)
            ]
        )

        # define lightcurve fitness
        def fitness(params):
            return surface_fitness(
                initial_mesh=initial_mesh,
                params=params,
                influence=influence,
                target_curves=target_curves,
                cameras=cameras,
                m=args.m,
                curve_types=curve_types,
                forward=forward,
                )


        initial_curves = render_curves(
            initial_mesh.vertices,
            initial_mesh.faces,
            cameras=cameras,
            m=args.m,
            curve_types=curve_types,
            forward=forward,
        )

        initial_dice = score_mesh(
            initial_mesh.vertices,
            initial_mesh.faces,
            truth_mesh_voxelised,
        )

        print(f"Initial STL Dice: {initial_dice:.4f}")


    else:
        raise ValueError(
            f"Unknown genetic mode: {args.mode}"
        )




    # ------------------------------------------------------------
    # Define how to save checkpoints in solver
    # ------------------------------------------------------------

    # [0] = initial pop, [n] = nth evolution    
    checkpoint_generations = [
        0,
        args.generations // 4,
        args.generations // 2,
        3 * args.generations // 4,
        args.generations,
    ]

    def checkpoint(
        generation,
        best_params,
        best_fitness,
    ):
        # Only save at requested checkpoints
        if generation not in checkpoint_generations:
            return

        checkpoint_name = f"generation_{generation:04d}"

        # --------------------------------------------------------
        # Create current best mesh
        # --------------------------------------------------------

        if args.mode == "sh":

            mesh = save_shape_stl(
                best_params,
                L=args.L,
                subdiv=args.subdiv,
                path=stl_dir / f"{checkpoint_name}.stl",
            )

        elif args.mode == "surface":

            mesh = deform_surface(
                initial_mesh,
                best_params,
                influence,
            )

            mesh.export(
                stl_dir / f"{checkpoint_name}.stl"
            )

        # --------------------------------------------------------
        # Generate and save lightcurves
        # --------------------------------------------------------

        curves = render_curves(
            mesh.vertices,
            mesh.faces,
            cameras=cameras,
            m=args.m,
            curve_types=curve_types,
            forward=forward,
        )

        np.save(
            lc_dir / f"{checkpoint_name}_curves.npy",
            curves,
        )

        # --------------------------------------------------------
        # Dice score
        # --------------------------------------------------------

        dice_score = score_mesh(
            mesh.vertices,
            mesh.faces,
            truth_mesh_voxelised,
        )

        # --------------------------------------------------------
        # Time taken
        # --------------------------------------------------------

        elapsed_time = time.perf_counter() - start_time

        print(
            f"Checkpoint generation {generation}: "
            f"fitness={best_fitness:.6g}, "
            f"dice={dice_score:.4f}, "
            f"time={elapsed_time:.1f}s"
        )

        # --------------------------------------------------------
        # Update results.json
        # --------------------------------------------------------

        save_checkpoint_results(
            output_dir=output_dir,
            args=args,
            generation=generation,
            best_params=best_params,
            best_fitness=best_fitness,
            dice_score=dice_score,
            time_taken=elapsed_time,
        )

        # --------------------------------------------------------
        # Save solver (a resume-from-checkpoint convenience only -- everything else this
        # checkpoint writes above, STL/curves/Dice/results.json, has already been saved by
        # this point, so a failure here should not cost any of that)
        # --------------------------------------------------------

        try:
            with open(output_dir / "solver.pkl", "wb") as f:
                cloudpickle.dump(solver, f)
        except TypeError as exc:
            # solver.fitness_fn closes over `forward`, which for --forward-model exact holds
            # nvdiffrast's live CUDA context (RasterizeCRStateWrapper, a C extension object
            # with no __reduce__) -- unpicklable by construction, not a bug in this specific
            # run. Uncaught, this used to kill the whole GA at generation 0's checkpoint,
            # every time, for every exact-model run: confirmed on 8/10 models of a real
            # 10-model production run, all stopped dead here.
            print(f"warning: could not save solver.pkl (the resume-from-checkpoint file) at "
                 f"generation {generation}: {exc}. Continuing without it -- this run just "
                 f"cannot be resumed from this checkpoint if interrupted; the STL, curves "
                 f"and results.json already written above are unaffected.", flush=True)

    # ------------------------------------------------------------
    # Genetic optimiser
    # ------------------------------------------------------------

    solver = GeneticSolver(
        fitness_fn=fitness,
        initial_params=initial_params,
        mutation_scale=args.mutation_scale,
        population_size=args.population_size,
        n_parents=args.parents,
        n_generations=args.generations,
        mutation_decay=args.mutation_decay,
        bounds=bounds,
        seed=args.seed,
    )

    result = solver.run(checkpoint_fn=checkpoint)


    # ------------------------------------------------------------
    # Print recovered coefficients
    # ------------------------------------------------------------

    print("\nOptimisation complete")
    print("---------------------")

    if args.model is None and args.mode == 'sh':
        print("\nTarget coefficients:")
        print(target_coefficients)

    print("\nRecovered coefficients:")
    print(result.best_params)

    print(
        f"\nBest fitness: "
        f"{result.best_fitness:.6g}"
    )



    # ------------------------------------------------------------
    # Lightcurve visualisations
    # ------------------------------------------------------------

    plot_lightcurve_comparison(
        target_curves=target_curves,
        curves_dir=lc_dir,
        m=args.m,
        output_path=output_dir / "lightcurve_comparison.png",
        plot_all=False
    )



    # ------------------------------------------------------------
    # Convergence Plot
    # ------------------------------------------------------------

    plot_genetic_convergence(
        result.best_fitness_history,
        output_dir / "genetic_convergence.png",
    )


    # ------------------------------------------------------------
    # Save final results
    # ------------------------------------------------------------

    with open(output_dir / "solver.pkl", "rb") as f:
        solver = cloudpickle.load(f)


if __name__ == "__main__":
    main()
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
    mesh_curves_convex,
)
from hac26.solvers.genetic import GeneticSolver
from hac26.geometry import build_cameras
from hac26.recon import dice
from hac26.scoring.voxel import score_mesh, prepare_truth
from hac26.data_io import load_model_curves
from hac26.genetic_utils import make_target_coefficients, lightcurve_fitness, \
                                save_shape_stl,  load_truth_mesh, \
                                plot_lightcurve_comparison, plot_genetic_convergence, \
                                save_checkpoint_results




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

    # --------------------------------------------------------
    # Load source of truth (if exists)
    # --------------------------------------------------------
    
    # camera angles 
    cameras = build_cameras()

    # lightcurve types
    curve_types = ["intensity"] * len(cameras) #+ ["binary"] * len(cameras) 


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

        target_curves = mesh_curves_convex(
            target_vertices,
            target_faces,
            cameras=cameras,
            m=args.m,
            curve_types=curve_types,
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
        output_dir / "truth_curves.npy",
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
    # SH dimensionality
    # ------------------------------------------------------------

    n_coefficients = args.L * (args.L + 2)

    print(
        f"Using L={args.L} "
        f"({n_coefficients} SH coefficients)"
    )


    # ------------------------------------------------------------
    # Initial genome
    #
    # All zeros corresponds to:
    #
    # r = exp(0) = 1
    #
    # i.e. a unit sphere.
    # ------------------------------------------------------------

    initial_coefficients = np.zeros(
        n_coefficients,
    )


    # ------------------------------------------------------------
    # Bounds
    #
    # Keep the mutations within a sensible range.
    # ------------------------------------------------------------

    bounds = np.array(
        [
            [-0.5, 0.5]
            for _ in range(n_coefficients)
        ]
    )


    # ------------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------------

    def fitness(coefficients):
        return lightcurve_fitness(
            coefficients=coefficients,
            target_curves=target_curves,
            L=args.L,
            subdiv=args.subdiv,
            cameras=cameras,
            m=args.m,
            curve_types=curve_types,
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

    dice_scores = {}

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
        # Save current best parameters
        # --------------------------------------------------------

        # np.save(
        #     output_dir / f"{checkpoint_name}_params.npy",
        #     best_params,
        # )

        # --------------------------------------------------------
        # Save current best STL
        # --------------------------------------------------------

        mesh = save_shape_stl(
            best_params,
            L=args.L,
            subdiv=args.subdiv,
            path=stl_dir / f"{checkpoint_name}.stl",
        )

        # --------------------------------------------------------
        # Generate and save current best lightcurves
        # --------------------------------------------------------

        curves = mesh_curves_convex(
            mesh.vertices,
            mesh.faces,
            cameras=cameras,
            m=args.m,
            curve_types=curve_types,
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

        dice_scores[checkpoint_name] = float(dice_score)

        print(
            f"Checkpoint generation {generation}: "
            f"fitness={best_fitness:.6g}, "
            f"dice={dice_score:.4f}"
        )

        # --------------------------------------------------------
        # Update results.json
        # --------------------------------------------------------

        elapsed_time = time.perf_counter() - start_time

        save_checkpoint_results(
            output_dir=output_dir,
            args=args,
            best_params=best_params,
            best_fitness=best_fitness,
            dice_scores=dice_scores,
            comp_t = elapsed_time
        )

        with open(output_dir / "solver.pkl", "wb") as f:
            cloudpickle.dump(solver, f)

    # ------------------------------------------------------------
    # Genetic optimiser
    # ------------------------------------------------------------

    solver = GeneticSolver(
        fitness_fn=fitness,
        initial_params=initial_coefficients,
        mutation_scale=args.mutation_scale,
        population_size=args.population_size,
        n_parents=args.parents,
        n_generations=args.generations,
        mutation_decay=0.98,
        bounds=bounds,
        seed=args.seed,
    )

    result = solver.run(checkpoint_fn=checkpoint)


    # ------------------------------------------------------------
    # Print recovered coefficients
    # ------------------------------------------------------------

    print("\nOptimisation complete")
    print("---------------------")

    if args.model is None:
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


    final_vertices, final_faces = sh_mesh_from_coefficients(
        result.best_params,
        L=args.L,
        subdiv=args.subdiv,
    )

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

    total_time = time.perf_counter() - start_time

    with open(output_dir / "solver.pkl", "rb") as f:
        solver = cloudpickle.load(f)


if __name__ == "__main__":
    main()
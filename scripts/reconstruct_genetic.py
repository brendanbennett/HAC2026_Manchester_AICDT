#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import time

from hac26.shapes import (
    icosphere,
    sh_mesh_from_coefficients,
    mesh_curves_convex,
)
from hac26.solvers.genetic import GeneticSolver
from hac26.geometry import build_cameras
from hac26.recon import dice
from hac26.scoring.voxel import score_mesh, prepare_truth


def make_target_coefficients(
    rng: np.random.Generator,
    L: int,
    amp: float = 0.35,
    decay: float = 1.5,
) -> np.ndarray:
    """Generate the hidden SH coefficients for the toy target."""

    ls = np.concatenate(
        [
            [l] * (2 * l + 1)
            for l in range(1, L + 1)
        ]
    )

    return rng.normal(
        0.0,
        amp / (1.0 + ls) ** decay,
    )


def shape_fitness(
    coefficients: np.ndarray,
    target_coefficients: np.ndarray,
) -> float:
    """Toy fitness based on coefficient recovery.

    This is deliberately simple for the first test. The target coefficients
    are hidden from the GA, but the fitness currently measures distance in
    coefficient space.

    This will later be replaced by a lightcurve-based fitness.
    """

    error = np.sum(
        (coefficients - target_coefficients) ** 2
    )

    return -error


def lightcurve_fitness(
    coefficients,
    target_curves,
    L,
    subdiv,
    cameras,
    m,
    curve_types,
    c_lambert=0.1,
    sigma=1.0,
    delta=1.0,
    psi0=0.0,
    ls_weight=1.0,
):
    """Calculate fitness by comparing model and target lightcurves.

    Higher fitness is better, so this returns the negative mean squared
    lightcurve residual.
    """

    # ------------------------------------------------------------
    # Candidate shape
    # ------------------------------------------------------------

    vertices, faces = sh_mesh_from_coefficients(
        coefficients,
        L=L,
        subdiv=subdiv,
    )

    # ------------------------------------------------------------
    # Candidate lightcurves
    # ------------------------------------------------------------

    curves = mesh_curves_convex(
        vertices,
        faces,
        cameras=cameras,
        m=m,
        curve_types=curve_types,
        c_lambert=c_lambert,
        sigma=sigma,
        delta=delta,
        psi0=psi0,
        ls_weight=ls_weight,
    )

    # ------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------

    residual = curves - target_curves

    # Mean squared error over all cameras and phases.
    mse = np.mean(residual**2)

    return -mse


def plot_mesh(
    ax,
    vertices,
    faces,
    title,
):
    """Plot a triangular mesh using matplotlib."""

    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        linewidth=0.1,
        alpha=0.9,
    )

    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_mesh_views(
    vertices,
    faces,
    title,
    axes,
):
    """Plot six views of a triangular mesh."""

    views = [
        (90, -90, "XY"),
        (0, -90, "XZ"),
        (0, 0, "YZ"),
        (30, 45, "Iso +++"),
        (30, 135, "Iso -++"),
        (30, -45, "Iso +-+"),
    ]

    for ax, (elev, azim, view_title) in zip(
        axes,
        views,
    ):
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            linewidth=0.1,
            alpha=0.9,
        )

        ax.view_init(
            elev=elev,
            azim=azim,
        )

        ax.set_title(view_title)

        ax.set_box_aspect(
            (1, 1, 1)
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


def save_shape_stl(
    coefficients,
    L,
    subdiv,
    path,
):
    """Generate an SH mesh and save it as an STL file."""

    vertices, faces = sh_mesh_from_coefficients(
        coefficients,
        L=L,
        subdiv=subdiv,
    )

    
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh.export(path)

    print(f"Saved {path}")

    return mesh


def save_results_json(
    output_dir,
    args,
    result,
    dice_scores,
    comp_t
):
    """Save run configuration and optimisation results to JSON."""


    config = vars(args).copy()
    config["n_coefficients"] = args.L * (args.L + 2)

    results = {
        "config": config,
        "result": {
            "dice_scores": dice_scores,
            "best_fitness": float(result.best_fitness),
            "best_params": result.best_params.tolist(),
            "fitness_history": [
                float(x)
                for x in result.best_fitness_history
            ],
        },
        "timing": {
            "total_seconds": comp_t,
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

###############################################################
# MAIN #
###############################################################

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
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
    "--dice-resolution",
    type=int,
    default=16,
    help="Voxel resolution used for Dice evaluation.",
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # time computation
    start_time = time.perf_counter()

    output_dir = Path("results/genetic"
            ) / (
            f"L{args.L}"
            f"_subdiv{args.subdiv}"
            f"_m{args.m}"
            f"_gen{args.generations}"
            f"_pop{args.population_size}"
            f"_parents{args.parents}"
            f"_seed{args.seed}"
            f"_diceres{args.dice_resolution}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Camera and lightcurve configuration
    # ------------------------------------------------------------

    # camera angles 
    cameras = build_cameras()

    # lightcurve types
    curve_types = ["intensity"] * len(cameras) #+ ["binary"] * N_CAMS


    # ------------------------------------------------------------
    # SH dimensionality
    # ------------------------------------------------------------

    n_coefficients = args.L * (args.L + 2)

    print(
        f"Using L={args.L} "
        f"({n_coefficients} SH coefficients)"
    )

    # ------------------------------------------------------------
    # Hidden target shape and lightcurves
    # ------------------------------------------------------------

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

    np.save(
    output_dir / "truth_curves.npy",
    target_curves,
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
    # Mutation scale
    #
    # Use the same scale structure as the SH shape prior.
    # ------------------------------------------------------------

    mutation_scale = 0.05

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
    # Genetic optimiser
    # ------------------------------------------------------------

    solver = GeneticSolver(
        fitness_fn=fitness,
        initial_params=initial_coefficients,
        mutation_scale=mutation_scale,
        population_size=args.population_size,
        n_parents=args.parents,
        n_generations=args.generations,
        mutation_decay=0.98,
        bounds=bounds,
        seed=args.seed,
    )

    result = solver.run()


    # ------------------------------------------------------------
    # Save shape checkpoints
    # ------------------------------------------------------------

    # use this in case it stops early
    # [0] = initial pop, [n] = nth evolution
    n_generations = len(result.best_fitness_history) - 1

    checkpoint_generations = [
        0,
        n_generations // 4,
        n_generations // 2,
        3 * n_generations // 4,
        n_generations,
    ]

    # Truth
    truth_mesh = save_shape_stl(
        target_coefficients,
        L=args.L,
        subdiv=args.subdiv,
        path=output_dir / "truth.stl",
    ) 

    truth_mesh_voxelised = prepare_truth(
        truth_mesh.vertices,
        truth_mesh.faces,
        n=args.dice_resolution,
    )


    dice_scores = {}

    for generation in checkpoint_generations:

        coefficients = result.best_params_history[generation]

        vertices, faces = sh_mesh_from_coefficients(
            coefficients,
            L=args.L,
            subdiv=args.subdiv,
        )

        # track updates
        mesh = save_shape_stl(
            coefficients,
            L=args.L,
            subdiv=args.subdiv,
            path=output_dir / f"generation_{generation:04d}.stl",
        )


        # Dice
        dice_scores[f"generation_{generation:04d}"] = score_mesh(
            mesh.vertices,
            mesh.faces,
            truth_mesh_voxelised,
        )

    # ------------------------------------------------------------
    # Print recovered coefficients
    # ------------------------------------------------------------

    print("\nOptimisation complete")
    print("---------------------")

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

    final_curves = mesh_curves_convex(
        final_vertices,
        final_faces,
        cameras=cameras,
        m=args.m,
        curve_types=curve_types,
    )


    fig, axes = plt.subplots(
        len(cameras),
        1,
        figsize=(8, 2 * len(cameras)),
        sharex=True,
    )

    if len(cameras) == 1:
        axes = [axes]

    for i, ax in enumerate(axes):

        ax.plot(
            target_curves[i],
            label="truth",
        )

        ax.plot(
            final_curves[i],
            "--",
            label="GA",
        )

        ax.set_ylabel(f"Camera {i}")

        ax.legend()

    axes[-1].set_xlabel("Phase")

    fig.tight_layout()

    fig.savefig(
        output_dir / "lightcurve_comparison.png",
        dpi=200,
    )

    plt.close(fig)



    # ------------------------------------------------------------
    # Convergence Plot
    # ------------------------------------------------------------

    plt.figure()

    plt.plot(
        np.arange(
            0,
            len(result.best_fitness_history)
        ),
        result.best_fitness_history
    )

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Genetic algorithm convergence")

    plt.tight_layout()

    plt.savefig(output_dir / "genetic_convergence.png")


    # ------------------------------------------------------------
    # Save final results
    # ------------------------------------------------------------

    total_time = time.perf_counter() - start_time

    save_results_json(
        output_dir=output_dir,
        args=args,
        result=result,
        dice_scores=dice_scores,
        comp_t = total_time
        )


if __name__ == "__main__":
    main()
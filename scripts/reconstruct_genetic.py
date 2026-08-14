#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from hac26.shapes import (
    icosphere,
    sh_mesh_from_coefficients,
)
from hac26.solvers.genetic import GeneticSolver


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
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------
    # SH dimensionality
    # ------------------------------------------------------------

    n_coefficients = args.L * (args.L + 2)

    print(
        f"Using L={args.L} "
        f"({n_coefficients} SH coefficients)"
    )

    # ------------------------------------------------------------
    # Hidden target shape
    # ------------------------------------------------------------

    target_coefficients = make_target_coefficients(
        rng,
        L=args.L,
    )

    target_vertices, target_faces = (
        sh_mesh_from_coefficients(
            target_coefficients,
            L=args.L,
            subdiv=args.subdiv,
        )
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
    # Checkpoints
    # ------------------------------------------------------------

    checkpoints = sorted(
        set(
            [
                1,
                10,
                25,
                args.generations,
            ]
        )
    )

    checkpoint_shapes = {}

    def save_checkpoint(
        generation,
        coefficients,
        fitness,
    ):
        checkpoint_shapes[generation] = (
            coefficients.copy(),
            fitness,
        )

    # ------------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------------

    def fitness(coefficients):
        return shape_fitness(
            coefficients,
            target_coefficients,
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
        # checkpoint_fn=save_checkpoint,
        # checkpoints=checkpoints,
    )

    result = solver.run()

    # ------------------------------------------------------------
    # Save shape checkpoints
    # ------------------------------------------------------------

    output_dir = Path("results/genetic")

    n_generations = len(result.history)

    checkpoint_generations = {
        "start": 0,
        "generation_25": max(1, n_generations // 4),
        "generation_50": max(1, n_generations // 2),
        "generation_75": max(1, 3 * n_generations // 4),
        "final": n_generations,
    }

    # Truth
    save_shape_stl(
        target_coefficients,
        L=args.L,
        subdiv=args.subdiv,
        path=output_dir / "truth.stl",
    )

    # Start
    save_shape_stl(
        initial_coefficients,
        L=args.L,
        subdiv=args.subdiv,
        path=output_dir / "start.stl",
    )

    # Checkpoint / final shapes
    for name, generation in checkpoint_generations.items():

        if name == "start":
            continue

        # History is zero-indexed.
        index = generation - 1

        coefficients = result.best_params_history[index]

        save_shape_stl(
            coefficients,
            L=args.L,
            subdiv=args.subdiv,
            path=output_dir / f"{name}.stl",
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
    # Save checkpoint visualisations
    # ------------------------------------------------------------

    n_generations = len(
        result.history
    )

    checkpoint_generations = sorted(
        set(
            [
                1,
                max(1, n_generations // 4),
                max(1, n_generations // 2),
                max(1, 3 * n_generations // 4),
                n_generations,
            ]
        )
    )

    for generation in checkpoint_generations:

        # history is zero-indexed
        index = generation - 1

        coefficients = (
            result.best_params_history[index]
        )

        fitness = (
            result.best_fitness_history[index]
        )




    # ------------------------------------------------------------
    # Visualise checkpoint shapes
    # ------------------------------------------------------------



    # ------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------

    plt.figure()

    plt.plot(
        np.arange(
            1,
            args.generations + 1,
        ),
        result.history,
    )

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Genetic algorithm convergence")

    plt.tight_layout()

    plt.savefig(output_dir / "genetic_convergence.png")


if __name__ == "__main__":
    main()
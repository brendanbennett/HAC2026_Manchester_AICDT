#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np

from hac26.solvers.genetic import GeneticSolver


def toy_fitness(params):
    """Toy objective for testing the genetic algorithm.

    The optimum is deliberately known:

        params = [1.5, -0.5, 2.0]

    The genetic algorithm should converge towards this point.
    """
    target = np.array([1.5, -0.5, 2.0])

    error = np.sum((params - target) ** 2)

    # Convert minimisation problem into maximisation.
    return -error


def main():
    parser = argparse.ArgumentParser(
        description="Run the prototype genetic shape optimiser."
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations.",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Number of individuals per generation.",
    )

    parser.add_argument(
        "--parents",
        type=int,
        default=10,
        help="Number of parents retained each generation.",
    )

    parser.add_argument(
        "--mutation-scale",
        type=float,
        default=0.2,
        help="Initial mutation scale.",
    )

    parser.add_argument(
        "--mutation-decay",
        type=float,
        default=0.95,
        help="Mutation scale decay per generation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    # Starting shape parameters.
    initial_params = np.array([50.0, 1.0, -10.0])

    # Bounds on the shape parameters.
    bounds = np.array(
        [
            [-100.0, 100.0],
            [-100.0, 100.0],
            [-100.0, 100.0],
        ]
    )

    solver = GeneticSolver(
        fitness_fn=toy_fitness,
        initial_params=initial_params,
        mutation_scale=args.mutation_scale,
        population_size=args.population_size,
        n_parents=args.parents,
        n_generations=args.generations,
        mutation_decay=args.mutation_decay,
        bounds=bounds,
        seed=args.seed,
    )

    result = solver.run()

    print("\nOptimisation complete")
    print("---------------------")
    print(f"Best parameters: {result.best_params}")
    print(f"Best fitness:    {result.best_fitness}")

    # Plot convergence.
    plt.figure()
    plt.plot(result.history)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Genetic algorithm convergence")
    plt.tight_layout()
    plt.savefig('./results/genetic/genetic_algo_convergence.png', dpi=250)



if __name__ == "__main__":
    main()
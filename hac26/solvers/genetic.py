"""
Class to perform genetic algorithm search for asteroid concavities. 

"""

# import modules
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class GeneticResult:
    """Result returned by the genetic solver."""

    best_params: np.ndarray
    best_fitness: float
    best_params_history: list
    best_fitness_history: list



class GeneticSolver:
    """Simple mutation-based genetic algorithm.

    The population consists of vectors of shape parameters. At each
    generation the best individuals are selected and mutated to create
    the next generation.

    Parameters
    ----------
    fitness_fn
        Function which takes a parameter vector and returns a scalar
        fitness. Higher fitness is considered better.

    initial_params
        Parameter vector around which the initial population is generated.

    mutation_scale
        Standard deviation of the initial mutation applied to parameters.

    population_size
        Number of individuals in each generation.

    n_parents
        Number of best individuals retained as parents.

    n_generations
        Number of generations to run.

    mutation_decay
        Multiplicative factor applied to the mutation scale after each
        generation.

    bounds
        Optional array of shape ``(n_parameters, 2)`` containing lower
        and upper bounds for each parameter.

    seed
        Random seed for reproducibility.
    """

    def __init__(
        self,
        fitness_fn,
        initial_params,
        mutation_scale=0.1,
        population_size=50,
        n_parents=10,
        n_generations=50,
        mutation_decay=0.95,
        bounds=None,
        seed=None,
    ):
        self.fitness_fn = fitness_fn
        self.initial_params = np.asarray(initial_params, dtype=float)

        self.mutation_scale = float(mutation_scale)
        self.population_size = int(population_size)
        self.n_parents = int(n_parents)
        self.n_generations = int(n_generations)
        self.mutation_decay = float(mutation_decay)

        self.best_params_history = []
        self.best_fitness_history = []


        self.rng = np.random.default_rng(seed)

        if bounds is not None:
            self.bounds = np.asarray(bounds, dtype=float)

            if self.bounds.shape != (len(self.initial_params), 2):
                raise ValueError(
                    "bounds must have shape "
                    f"({len(self.initial_params)}, 2)"
                )
        else:
            self.bounds = None

        if self.n_parents > self.population_size:
            raise ValueError("n_parents cannot exceed population_size")

    def _apply_bounds(self, params):
        """Apply parameter bounds."""
        if self.bounds is None:
            return params

        return np.clip(
            params,
            self.bounds[:, 0],
            self.bounds[:, 1],
        )

    def _initialise_population(self):
        """Generate the initial population."""
        population = (
            self.initial_params
            + self.rng.normal(
                loc=0.0,
                scale=self.mutation_scale,
                size=(
                    self.population_size,
                    len(self.initial_params),
                ),
            )
        )

        # Always include the unmodified initial shape.
        population[0] = self.initial_params

        return self._apply_bounds(population)

    def _evaluate(self, population):
        """Evaluate the fitness of every individual."""
        return np.asarray(
            [self.fitness_fn(params) for params in population]
        )

    def _select(self, population, fitness):
        """Select the best individuals."""
        indices = np.argsort(fitness)[::-1]
        indices = indices[: self.n_parents]

        return population[indices], fitness[indices]


    def _mutate(self, parents, mutation_scale):
        """Generate a new population by mutating selected parents.

        The best parent is carried over unchanged as population[0].
        The remaining population consists of mutated children drawn
        from the selected parents.
        """

        # ------------------------------------------------------------
        # Keep the best parent unchanged
        # ------------------------------------------------------------

        population = np.empty(
            (
                self.population_size,
                len(parents[0]),
            )
        )

        population[0] = parents[0]

        # ------------------------------------------------------------
        # Generate mutated children
        # ------------------------------------------------------------

        n_children = self.population_size - 1

        parent_indices = self.rng.integers(
            0,
            len(parents),
            size=n_children,
        )

        children = parents[parent_indices].copy()

        mutations = self.rng.normal(
            loc=0.0,
            scale=mutation_scale,
            size=children.shape,
        )

        children += mutations

        children = self._apply_bounds(children)

        population[1:] = children

        return population



    def run(self, checkpoint_fn):
        """Run the genetic optimisation.

        Parameters
        ----------
        checkpoint_fn : callable, optional
            Called after each generation with:
                checkpoint_fn(generation, best_params, best_fitness)
        

        Returns
        -------
        GeneticResult
            Best parameters, best fitness and fitness history.
        """
        population = self._initialise_population()

        best_params = None
        best_fitness = -np.inf

        mutation_scale = self.mutation_scale

        # ------------------------------------------------------------
        # Generation 0: evaluate the initial population
        # ------------------------------------------------------------

        fitness = self._evaluate(population)

        best_idx = np.argmax(fitness)

        best_fitness = fitness[best_idx]
        best_params = population[best_idx].copy()

        self.best_params_history.append(best_params.copy())
        self.best_fitness_history.append(best_fitness)

        print(
            f"Generation {0:3d}/{self.n_generations} "
            f"| fitness = {best_fitness:.6g} "
            f"| mutation = {mutation_scale:.4g}"
        )

        # Checkpoint generation 0
        if checkpoint_fn is not None:
            checkpoint_fn(
                0,
                best_params.copy(),
                best_fitness,
            )

        # ------------------------------------------------------------
        # Evolution
        # ------------------------------------------------------------

        for generation in range(1, self.n_generations + 1):

            # Select parents from the current population.
            parents, parent_fitness = self._select(
                population,
                fitness,
            )

            # Mutate parents to create the next population.
            population = self._mutate(
                parents,
                mutation_scale,
            )

            mutation_scale *= self.mutation_decay

            # Evaluate the new population.
            fitness = self._evaluate(population)

            generation_best_idx = np.argmax(fitness)
            generation_best_fitness = fitness[generation_best_idx]

            # Update global best.
            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_params = population[generation_best_idx].copy()

            self.best_params_history.append(best_params.copy())
            self.best_fitness_history.append(best_fitness)

            print(
                f"Generation {generation:3d}/{self.n_generations} "
                f"| fitness = {best_fitness:.6g} "
                f"| mutation = {mutation_scale:.4g}"
            )


            # Checkpoint
            if checkpoint_fn is not None:
                checkpoint_fn(
                    generation,
                    best_params.copy(),
                    best_fitness,
                )

        return GeneticResult(
            best_params=best_params,
            best_fitness=best_fitness,
            best_params_history = self.best_params_history,
            best_fitness_history = self.best_fitness_history

        )
#!/usr/bin/env python
# Created by "Thieu" at 12:24, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.legacy.bio_based.BBO.OriginalBBO cimport OriginalBBO


cdef class DevBBO(OriginalBBO):
    """
    The developed version: Biogeography-Based Optimization (BBO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_m (float): (0, 1) -> better [0.01, 0.2], Mutation probability
        + n_elites (int): (2, pop_size/2) -> better [2, 5], Number of elites will be keep for next generation

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.bio_based import BBO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = BBO.DevBBO(epoch=1000, pop_size=50, p_m=0.01, n_elites=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            p_m: float = 0.01,
            n_elites: int = 2,
            **kwargs: object
    ) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 100
            p_m: Mutation probability, default=0.01
            n_elites: Number of elites will be keep for next generation, default=2
        """
        super().__init__(epoch, pop_size, p_m, n_elites, **kwargs)

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ranked = self.population.sort()
        pop_elites = [agent.copy() for agent in ranked[:self.n_elites]]
        list_fitness = [agent.fitness for agent in self.population]
        pop_new = []
        for idx in range(0, pop_size):
            # Probabilistic migration to the i-th position
            # Pick a position from which to emigrate (roulette wheel selection)
            idx_selected = cy.roulette_wheel(self.generator, self.problem.sense, list_fitness)
            # this is the migration step
            condition = self.generator.random(self.problem.n_dims) < self.mr[idx]
            pos_new = np.where(
                condition, self.population[idx_selected].solution, self.population[idx].solution
            )
            # Mutation
            mutated = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            pos_new = np.where(
                self.generator.random(self.problem.n_dims) < self.p_m, mutated, pos_new
            )
            pos_new = self.population.correct_solution(pos_new)
            agent_new = self.population.create_agent(pos_new)
            pop_new.append(agent_new)
            if self.mode not in self.AVAILABLE_MODES:
                agent_new.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent_new, sense=self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        # replace the solutions with their new migrated and mutated versions then Merge Populations
        self.population = cy.sort_agents(self.population + pop_elites, self.problem.sense)[:pop_size]

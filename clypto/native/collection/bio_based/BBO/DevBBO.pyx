#!/usr/bin/env python
# Created by "Thieu" at 12:24, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.bio_based.BBO.OriginalBBO cimport OriginalBBO


cdef class DevBBO(OriginalBBO):
    """
    The developed version: Biogeography-Based Optimization (BBO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_m (float): (0, 1) -> better [0.01, 0.2], Mutation probability
        + n_elites (int): (2, pop_size/2) -> better [2, 5], Number of elites will be keep for next generation

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.bio_based import BBO    >>> import numpy as np
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

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ranked = self.population.sort()
        pop_elites = [cy.duplicate_agent(agent) for agent in ranked[:self.n_elites]]
        list_fitness = [agent.fitness for agent in self.population]
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            # Probabilistic migration to the i-th position
            # Pick a position from which to emigrate (roulette wheel selection)
            idx_selected = cy.roulette_wheel(self.generator, self.problem.sense, list_fitness)
            # this is the migration step
            condition = self.generator.random(self.problem.n_dims) < self.mr[idx]
            x = np.where(
                condition, self.population[idx_selected].solution, agent.solution
            )
            # Mutation
            mutated = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            x = np.where(
                self.generator.random(self.problem.n_dims) < self.p_m, mutated, x
            )
            x = cy.correct_solution(self.problem, x)
            agent_new = self.population.create_agent(x)
            n_population.append(agent_new)
            if self.mode == "sequential":
                agent_new.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent_new, sense=self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)
        # replace the solutions with their new migrated and mutated versions then Merge Populations
        self.population = self.population.spawn(cy.sort_agents(self.population + pop_elites, self.problem.sense)[:pop_size])

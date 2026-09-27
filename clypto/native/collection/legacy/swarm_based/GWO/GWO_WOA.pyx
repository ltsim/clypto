#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.legacy.swarm_based.GWO.OriginalGWO cimport OriginalGWO


cdef class GWO_WOA(OriginalGWO):
    """
    The original version of: Hybrid Grey Wolf - Whale Optimization Algorithm (GWO-WOA)

    Links:
        1. https://sci-hub.se/https://doi.org/10.1177/10775463211003402

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.GWO_WOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Obadina, O. O., Thaha, M. A., Althoefer, K., & Shaheed, M. H. (2022). Dynamic characterization of a master–slave
    robotic manipulator using a hybrid grey wolf–whale optimization algorithm. Journal of Vibration and Control, 28(15-16), 1992-2003.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.bb = 1.0
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # linearly decreased from 2 to 0
        a = 2.0 - epoch / self.epoch
        ranked = self.population.sort()
        list_best = [agent.copy() for agent in ranked[:3]]
        pop_new = []
        for idx in range(0, pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            if self.generator.random() < 0.5:
                da = self.generator.random() * np.abs(
                    C1 * list_best[0].solution - self.population[idx].solution
                )
            else:
                P, L = self.generator.random(), self.generator.uniform(-1, 1)
                da = (
                    P
                    * np.exp(self.bb * L)
                    * np.cos(2 * np.pi * L)
                    * np.abs(C1 * list_best[0].solution - self.population[idx].solution)
                )
            X1 = list_best[0].solution - A1 * da
            X2 = list_best[1].solution - A2 * np.abs(
                C2 * list_best[1].solution - self.population[idx].solution
            )
            X3 = list_best[2].solution - A3 * np.abs(
                C3 * list_best[2].solution - self.population[idx].solution
            )
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

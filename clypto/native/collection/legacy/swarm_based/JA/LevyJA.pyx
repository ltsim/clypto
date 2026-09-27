#!/usr/bin/env python
# Created by "Thieu" at 16:30, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.legacy.swarm_based.JA.DevJA cimport DevJA


cdef class LevyJA(DevJA):
    """
    The original version of: Levy-flight Jaya Algorithm (LJA)

    Notes
        + All third loops in this version also are removed
        + The beta value of Levy-flight equal to 1.8 as the best value in the paper.
        + https://doi.org/10.1016/j.eswa.2020.113902

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import JA    >>> import numpy as np
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
    >>> model = JA.LevyJA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Iacca, G., dos Santos Junior, V.C. and de Melo, V.V., 2021. An improved Jaya optimization
    algorithm with Lévy flight. Expert Systems with Applications, 165, p.113902.
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
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ranked = self.population.sort()
        (g_best,) = [agent.copy() for agent in ranked[:1]]
        (g_worst,) = [agent.copy() for agent in ranked[::-1][:1]]
        pop_new = []
        for idx in range(0, pop_size):
            L1 = cy.levy_flight(self.generator, beta=1.8, multiplier=1.0, size=None, case=-1)
            L2 = cy.levy_flight(self.generator, beta=1.8, multiplier=1.0, size=None, case=-1)
            pos_new = (
                    self.population[idx].solution
                    + np.abs(L1) * (g_best.solution - np.abs(self.population[idx].solution))
                    - np.abs(L2) * (g_worst.solution - np.abs(self.population[idx].solution))
            )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent, self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

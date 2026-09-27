#!/usr/bin/env python
# Created by "Thieu" at 17:19, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.legacy.human_based.SPBO.OriginalSPBO cimport OriginalSPBO


cdef class DevSPBO(OriginalSPBO):
    """
    The developed version of: Student Psychology Based Optimization (SPBO)

    Notes:
        1. Replace uniform random number by normal random number
        2. Sort the population and select 1/3 pop size for each category

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import SPBO    >>> import numpy as np
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
    >>> model = SPBO.DevSPBO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        good = int(pop_size / 3)
        average = 2 * int(pop_size / 3)
        x_mean = np.mean([agent.solution for agent in self.population], axis=0)
        pop_new = []
        for idx in range(0, pop_size):
            if idx == 0:
                j = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
                new_pos = self.g_best.solution + self.generator.normal(
                    0, 1, self.problem.n_dims
                ) * (self.g_best.solution - self.population[j].solution)
            elif idx < good:  ## Good Student
                if self.generator.random() > self.generator.random():
                    new_pos = self.g_best.solution + self.generator.normal(
                        0, 1, self.problem.n_dims
                    ) * (self.g_best.solution - self.population[idx].solution)
                else:
                    ra = self.generator.random(self.problem.n_dims)
                    new_pos = (
                            self.population[idx].solution
                            + ra * (self.g_best.solution - self.population[idx].solution)
                            + (1 - ra) * (self.population[idx].solution - x_mean)
                    )
            elif idx < average:  ## Average Student
                new_pos = self.population[idx].solution + self.generator.normal(
                    0, 1, self.problem.n_dims
                ) * (x_mean - self.population[idx].solution)
            else:
                new_pos = self.problem.generate_solution()
            new_pos = self.population.correct_solution(new_pos)
            agent = self.population.create_agent(new_pos)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

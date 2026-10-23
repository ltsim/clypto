#!/usr/bin/env python
# Created by "Thieu" at 17:19, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.human_based.SPBO.OriginalSPBO cimport OriginalSPBO


cdef class DevSPBO(OriginalSPBO):
    """
    The developed version of: Student Psychology Based Optimization (SPBO)

    Notes:
        1. Replace uniform random number by normal random number
        2. Sort the population and select 1/3 pop size for each category

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import SPBO    >>> import numpy as np
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

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        good = int(pop_size / 3)
        average = 2 * int(pop_size / 3)
        x_mean = np.mean([agent.solution for agent in self.population], axis=0)
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            if idx == 0:
                j = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
                new_pos = self.g_best.solution + self.generator.normal(
                    0, 1, self.problem.n_dims
                ) * (self.g_best.solution - self.population[j].solution)
            elif idx < good:  ## Good Student
                if self.generator.random() > self.generator.random():
                    new_pos = self.g_best.solution + self.generator.normal(
                        0, 1, self.problem.n_dims
                    ) * (self.g_best.solution - agent.solution)
                else:
                    ra = self.generator.random(self.problem.n_dims)
                    new_pos = (
                            agent.solution
                            + ra * (self.g_best.solution - agent.solution)
                            + (1 - ra) * (agent.solution - x_mean)
                    )
            elif idx < average:  ## Average Student
                new_pos = agent.solution + self.generator.normal(
                    0, 1, self.problem.n_dims
                ) * (x_mean - agent.solution)
            else:
                new_pos = self.problem.generate_solution()
            new_pos = cy.correct_solution(self.problem, new_pos)
            child = self.population.create_agent(new_pos)
            n_population.append(child)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)

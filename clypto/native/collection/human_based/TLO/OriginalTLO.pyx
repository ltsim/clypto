#!/usr/bin/env python
# Created by "Thieu" at 10:14, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.human_based.TLO.DevTLO cimport DevTLO


cdef class OriginalTLO(DevTLO):
    """
    The original version of: Teaching Learning-based Optimization (TLO)

    Notes:
        + Third loops are removed
        + This version is inspired from above link
        + https://github.com/andaviaco/tblo

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import TLO    >>> import numpy as np
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
    >>> model = TLO.OriginalTLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R.V., Savsani, V.J. and Vakharia, D.P., 2011. Teaching–learning-based optimization: a novel method
    for constrained mechanical design optimization problems. Computer-aided design, 43(3), pp.303-315.
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

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        for idx, agent in enumerate(self.population.toarray()):
            ## Teaching Phrase
            TF = self.generator.integers(1, 3)  # 1 or 2 (never 3)
            #### Remove third loop here
            list_pos = np.array([child.solution for child in self.population])
            x = agent.solution + self.generator.uniform(
                0, 1, self.problem.n_dims
            ) * (self.g_best.solution - TF * np.mean(list_pos, axis=0))
            x = cy.correct_solution(self.problem, x)
            child = self.population.generate_agent(x)
            if cy.is_better(child, agent, self.problem.sense):
                self.population[idx] = child
            ## Learning Phrase
            id_partner = self.generator.choice(
                np.setxor1d(np.array(range(pop_size)), np.array([idx]))
            )
            #### Remove third loop here
            if cy.is_better(self.population[idx], self.population[id_partner], self.problem.sense):
                diff = self.population[idx].solution - self.population[id_partner].solution
            else:
                diff = self.population[id_partner].solution - self.population[idx].solution
            x = (
                    self.population[idx].solution
                    + self.generator.uniform(0, 1, self.problem.n_dims) * diff
            )
            x = cy.correct_solution(self.problem, x)
            child = self.population.generate_agent(x)
            if cy.is_better(child, self.population[idx], self.problem.sense):
                self.population[idx] = child

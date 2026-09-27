#!/usr/bin/env python
# Created by "Thieu" at 17:41, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalWarSO(cy.Optimizer):
    """
    The original version of: War Strategy Optimization (WarSO) algorithm

    Links:
       1. https://www.researchgate.net/publication/358806739_War_Strategy_Optimization_Algorithm_A_New_Effective_Metaheuristic_Algorithm_for_Global_Optimization

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + rr (float): [0.1, 0.9], the probability of switching position updating, default=0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import WarSO    >>> import numpy as np
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
    >>> model = WarSO.OriginalWarSO(epoch=1000, pop_size=50, rr=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Ayyarao, Tummala SLV, and Polamarasetty P. Kumar. "Parameter estimation of solar PV models with a new proposed
    war strategy optimization algorithm." International Journal of Energy Research (2022).
    """

    cdef public double rr

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, rr: float = 0.1, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            rr (float): the probability of switching position updating, default=0.1
        """
        super().__init__(parameters=["epoch", "pop_size", "rr"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.rr = cy.validator(float, rr, (0.0, 1.0), "rr")

    def initialize_variables(self):
        pop_size = self.population.size()
        self.wl = 2 * np.ones(pop_size)
        self.wg = np.zeros(pop_size)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_sorted, indices = (cy.sort_agents(self.population, self.problem.sense), cy.argsort_agents(self.population, self.problem.sense))
        self.wl = self.wl[indices]
        self.wg = self.wg[indices]
        com = self.generator.permutation(pop_size)
        for idx, agent in enumerate(self.population.toarray()):
            r1 = self.generator.random()
            if r1 < self.rr:
                x = 2 * r1 * (
                    self.g_best.solution - self.population[com[idx]].solution
                ) + self.wl[idx] * self.generator.random() * (
                    pop_sorted[idx].solution - agent.solution
                )
            else:
                x = 2 * r1 * (
                    pop_sorted[idx].solution - self.g_best.solution
                ) + self.generator.random() * (
                    self.wl[idx] * self.g_best.solution - agent.solution
                )
            x = cy.correct_solution(self.problem, x)
            child = self.population.generate_agent(x)
            if cy.is_better(child, agent, self.problem.sense):
                self.population[idx] = child
                self.wg[idx] += 1
                self.wl[idx] = 1 * self.wl[idx] * (1 - self.wg[idx] / self.epoch) ** 2

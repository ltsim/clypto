#!/usr/bin/env python
# Created by "Thieu" at 10:21, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.native.collection.human_based.QSA.DevQSA cimport DevQSA
cimport clypto.core as cy


def opposition_based(DevQSA opt, pop, g_best):
    """Opposition-based step of the QSA variants (OppoQSA, ImprovedQSA): each agent against g_best, greedy."""
    pop_size = opt.population.size()
    pop = cy.sort_agents(pop, opt.problem.sense)
    pop_new = []
    for idx in range(0, pop_size):
        pos_new = cy.opposite_solution(opt.problem, opt.generator, pop[idx], g_best)
        pos_new = cy.correct_solution(opt.problem, pos_new)
        agent = opt.population.create_agent(pos_new)
        pop_new.append(agent)
        if opt.mode == "sequential":
            agent.evaluate(opt.problem)
            pop_new[-1] = cy.get_better_agent(agent, pop[idx], opt.problem.sense)
    if opt.mode != "sequential":
        pop_new = opt.population.evaluate(pop_new, opt.mode)
        pop_new = cy.greedy_agents(pop, pop_new, opt.problem.sense)
    return pop_new


cdef class OppoQSA(DevQSA):
    """
    The opposition-based learning version: Queuing Search Algorithm (OQSA)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import QSA    >>> import numpy as np
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
    >>> model = QSA.OppoQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
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
        self.sort_flag = True

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.population, epoch)
        pop = self.update_business_2__(pop)
        pop = self.update_business_3__(pop, self.g_best)
        self.population = self.population.spawn(opposition_based(self, pop, self.g_best))

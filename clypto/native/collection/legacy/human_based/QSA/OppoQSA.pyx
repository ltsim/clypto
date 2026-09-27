#!/usr/bin/env python
# Created by "Thieu" at 10:21, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.native.collection.legacy.human_based.QSA.DevQSA cimport DevQSA
cimport clypto.core as cy


cdef class OppoQSA(DevQSA):
    """
    The opposition-based learning version: Queuing Search Algorithm (OQSA)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import QSA    >>> import numpy as np
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

    def opposition_based__(self, pop=None, g_best=None):
        pop_size = self.population.size()
        pop = cy.sort_agents(pop, self.problem.sense)
        pop_new = []
        for idx in range(0, pop_size):
            pos_new = self.population.opposite_solution(pop[idx], g_best)
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                pop_new[-1] = cy.get_better_agent(agent, pop[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            pop_new = cy.greedy_agents(pop, pop_new, self.problem.sense)
        return pop_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.population, epoch)
        pop = self.update_business_2__(pop)
        pop = self.update_business_3__(pop, self.g_best)
        self.population = self.opposition_based__(pop, self.g_best)

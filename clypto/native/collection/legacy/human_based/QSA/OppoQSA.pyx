#!/usr/bin/env python
# Created by "Thieu" at 10:21, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.native.collection.legacy.human_based.QSA.DevQSA cimport DevQSA


cdef class OppoQSA(DevQSA):
    """
    The opposition-based learning version: Queuing Search Algorithm (OQSA)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import QSA    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = QSA.OppoQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
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
        pop = self.get_sorted_population(pop, self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.generate_opposition_solution(pop[idx], g_best)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop_new[-1] = self.get_better_agent(
                    agent, pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            pop_new = self.greedy_selection_population(
                pop, pop_new, self.problem.minmax
            )
        return pop_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.pop, epoch)
        pop = self.update_business_2__(pop)
        pop = self.update_business_3__(pop, self.g_best)
        self.pop = self.opposition_based__(pop, self.g_best)

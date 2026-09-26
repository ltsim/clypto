#!/usr/bin/env python
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.legacy.human_based.LCO.OriginalLCO cimport OriginalLCO


cdef class DevLCO(OriginalLCO):
    """
    The developed version: Life Choice-based Optimization (LCO)

    Notes:
        + The flow is changed with if else statement.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r1 (float): [1.5, 4], coefficient factor, default = 2.35

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import LCO    >>> import numpy as np
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
    >>> model = LCO.DevLCO(epoch=1000, pop_size=50, r1 = 2.35)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            r1: float = 2.35,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r1 (float): coefficient factor
        """
        super().__init__(epoch, pop_size, r1, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # epoch: current chance, self.epoch: number of chances
        pop_new = []
        for idx in range(0, self.pop_size):
            prob = self.generator.random()
            if prob > 0.875:  # Update using Eq. 1, update from n best position
                temp = np.array(
                    [
                        self.generator.random() * self.pop[j].solution
                        for j in range(0, self.n_agents)
                    ]
                )
                temp = np.mean(temp, axis=0)
            elif prob < 0.7:  # Update using Eq. 2-6
                f = epoch / self.epoch
                if idx != 0:
                    better_diff = (
                            f
                            * self.r1
                            * (self.pop[idx - 1].solution - self.pop[idx].solution)
                    )
                else:
                    better_diff = (
                            f * self.r1 * (self.g_best.solution - self.pop[idx].solution)
                    )
                best_diff = (
                        (1 - f) * self.r1 * (self.pop[0].solution - self.pop[idx].solution)
                )
                temp = (
                        self.pop[idx].solution
                        + self.generator.random() * better_diff
                        + self.generator.random() * best_diff
                )
            else:
                temp = self.problem.generate_solution()
            pos_new = self.correct_solution(temp)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )

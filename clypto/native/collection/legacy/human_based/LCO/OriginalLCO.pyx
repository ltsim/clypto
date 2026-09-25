#!/usr/bin/env python
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer._native.legacy cimport _LegacyOptimizer


cdef class OriginalLCO(_LegacyOptimizer):
    """
    The original version of: Life Choice-based Optimization (LCO)

    Links:
        1. https://doi.org/10.1007/s00500-019-04443-z

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r1 (float): [1.5, 4], coefficient factor, default = 2.35

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import LCO    >>> import numpy as np
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
    >>> model = LCO.OriginalLCO(epoch=1000, pop_size=50, r1 = 2.35)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Khatri, A., Gaba, A., Rana, K.P.S. and Kumar, V., 2020. A novel life choice-based optimizer. Soft Computing, 24(12), pp.9121-9141.
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
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.r1 = self.validator.check_float("r1", r1, [1.0, 3.0])
        self.set_parameters(["epoch", "pop_size", "r1"])
        self.n_agents = int(np.ceil(np.sqrt(self.pop_size)))
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
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
                f1 = 1 - epoch / self.epoch
                f2 = 1 - f1
                prev_pos = (
                    self.g_best.solution if idx == 0 else self.pop[idx - 1].solution
                )
                best_diff = (
                        f1 * self.r1 * (self.g_best.solution - self.pop[idx].solution)
                )
                better_diff = f2 * self.r1 * (prev_pos - self.pop[idx].solution)
                temp = (
                        self.pop[idx].solution
                        + self.generator.random() * better_diff
                        + self.generator.random() * best_diff
                )
            else:
                temp = (
                        self.problem.ub
                        - (self.pop[idx].solution - self.problem.lb)
                        * self.generator.random()
                )
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

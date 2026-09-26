#!/usr/bin/env python
# Created by "Thieu" at 15:05, 03/06/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class OriginalSLO(_LegacyOptimizer):
    """
    The original version of: Sea Lion Optimization Algorithm (SLO)

    Notes:
        + There are some unclear equations and parameters in the original paper
        + https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
        + https://doi.org/10.14569/IJACSA.2019.0100548

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import SLO    >>> import numpy as np
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
    >>> model = SLO.OriginalSLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Masadeh, R., Mahafzah, B.A. and Sharieh, A., 2019. Sea lion optimization algorithm. Sea, 10(5), p.388.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(
            self.problem.lb <= solution, solution <= self.problem.ub
        )
        pos_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, pos_rand)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        c = 2 - 2 * epoch / self.epoch
        t0 = self.generator.random()
        v1 = np.sin(2 * np.pi * t0)
        v2 = np.sin(2 * np.pi * (1 - t0))
        SP_leader = np.abs(
            v1 * (1 + v2) / v2
        )  # In the paper this is not clear how to calculate
        pop_new = []
        for idx in range(0, self.pop_size):
            if SP_leader < 0.25:
                if c < 1:
                    pos_new = self.g_best.solution - c * np.abs(
                        2 * self.generator.random() * self.g_best.solution
                        - self.pop[idx].solution
                    )
                else:
                    ri = self.generator.choice(
                        list(set(range(0, self.pop_size)) - {idx})
                    )  # random index
                    pos_new = self.pop[ri].solution - c * np.abs(
                        2 * self.generator.random() * self.pop[ri].solution
                        - self.pop[idx].solution
                    )
            else:
                pos_new = (
                    np.abs(self.g_best.solution - self.pop[idx].solution)
                    * np.cos(2 * np.pi * self.generator.uniform(-1, 1))
                    + self.g_best.solution
                )
            # In the paper doesn't check also doesn't update old solution at this point
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    self.pop[idx], agent, self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )

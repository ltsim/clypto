#!/usr/bin/env python
# Created by "Thieu" at 15:05, 03/06/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalSLO(LegacyNativeOptimizer):
    """
    The original version of: Sea Lion Optimization Algorithm (SLO)

    Notes:
        + There are some unclear equations and parameters in the original paper
        + https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
        + https://doi.org/10.14569/IJACSA.2019.0100548

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import SLO    >>> import numpy as np
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
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef object amend_solution(self, object solution):
        condition = np.logical_and(
            self.problem.lb <= solution, solution <= self.problem.ub
        )
        pos_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, pos_rand)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        c = 2 - 2 * epoch / self.epoch
        t0 = self.generator.random()
        v1 = np.sin(2 * np.pi * t0)
        v2 = np.sin(2 * np.pi * (1 - t0))
        SP_leader = np.abs(v1 * (1 + v2) / v2)  # In the paper this is not clear how to calculate
        for idx in range(0, self.pop_size):
            if SP_leader < 0.25:
                if c < 1:
                    pos_new = g_best - c * np.abs(2 * self.generator.random() * g_best - Xp[idx])
                else:
                    ri = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))  # random index
                    pos_new = Xp[ri] - c * np.abs(2 * self.generator.random() * Xp[ri] - Xp[idx])
            else:
                pos_new = np.abs(g_best - Xp[idx]) * np.cos(2 * np.pi * self.generator.uniform(-1, 1)) + g_best
            # In the paper doesn't check also doesn't update old solution at this point
            pos_new = self.correct_solution(pos_new)
            ops.commit(self, pop, cand, idx, pos_new, swarm, True)
        if swarm:
            ops.finish(self, cand, 0, pop.n)

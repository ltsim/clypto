#!/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class DevWOA(LegacyNativeOptimizer):
    """
    The developed version of: Whale Optimization Algorithm (WOA)

    Notes:
        + Hanlding simple vector instead of loop through whole dimensions
        + Using greedy to update position

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import WOA    >>> import numpy as np
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
    >>> model = WOA.DevWOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm. Advances in engineering software, 95, pp.51-67.
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

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        a = 2 - 2 * epoch / self.epoch  # linearly decreased from 2 to 0
        pop_new = []
        for idx in range(0, self.pop_size):
            r = self.generator.random()
            A = 2 * a * r - a
            C = 2 * r
            l = self.generator.uniform(-1, 1)
            p = 0.5
            b = 1

            # Get pos1
            pos1 = g_best - A * np.abs(
                C * g_best - Xp[idx]
            )

            # Get pos2
            id_r2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            pos2 = Xp[id_r2] - A * np.abs(
                C * Xp[id_r2] - Xp[idx]
            )

            # Get pos3
            D1 = np.abs(g_best - Xp[idx])
            pos3 = g_best + np.exp(b * l) * np.cos(2 * np.pi * l) * D1

            # Get final pos_new
            pos_new = pos1 if np.abs(A) < 1 else pos2
            pos_new = np.where(
                self.generator.random(size=self.problem.n_dims) < p, pos_new, pos3
            )

            # Correct solution
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class IOBL_GWO(LegacyNativeOptimizer):
    """
    The original version of: Improved Opposite-based Learning Grey Wolf Optimizer (IOBL-GWO)

    Notes:
        + In the paper, they called it "Improved Grey Wolf Optimizer (IGWO)", but there are many improved versions of GWO.
        + So based on their proposed equations, we called it as "Improved Opposite-based Learning Grey Wolf Optimizer (IOBL-GWO)".
        + This algorithm is heavily (4x - 6X slower than original) because of multiple times of calculating the fitness of agent in each population.

    Links:
        1. https://doi.org/10.1007/s12652-020-02153-1

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.IOBL_GWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Bansal, J. C., & Singh, S. (2021). A better exploration strategy in Grey Wolf Optimizer. Journal of Ambient Intelligence and Humanized Computing, 12(1), 1099-1118.
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
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand, sub, obl
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        lb, ub = self.problem.lb, self.problem.ub
        a = 2 - 2.0 * epoch_c / self.epoch  # linearly decreased from 2 to 0
        order = self.sorted_order(pop)
        best = np.array(X[order[:3]])
        # explorative equation first: around a random wolf or the alpha wolf
        R = rng.random((n, 5, 1))
        x_rand = X[ops.others(self, n)[:, 0]]
        x_avg = np.mean(np.ascontiguousarray(X), axis=0)
        pos_e = np.where(R[:, 4] >= 0.5, x_rand - R[:, 0] * np.abs(x_rand - 2 * R[:, 1] * X),
                         (best[0] - x_avg) - R[:, 2] * (lb + R[:, 3] * (ub - lb)))
        cand = pop.empty_like()
        cand.X[:] = self.correct_solution(pos_e)
        self.evaluate(cand, 0, n)
        # where it is not an improvement: the original GWO update
        fail = np.flatnonzero(~ops.better(self, cand.F, pop.F))
        if len(fail):
            m = len(fail)
            G = rng.random((m, 6, d))
            Xs = best[None] - (a * (2 * G[:, :3] - 1)) * np.abs(2 * G[:, 3:] * best[None] - X[fail][:, None, :])
            sub = pop.take(fail)
            sub.X[:] = self.correct_solution(Xs.sum(axis=1) / 3.0)
            self.evaluate(sub, 0, m)
            cand.buf[fail] = sub.buf
        ops.greedy(self, cand)
        # opposition-based learning of the three leaders replaces the three worst wolves when it is better
        order = self.sorted_order(pop)
        obl = pop.take(order[:3])
        obl.X[:] = self.correct_solution(lb + ub - pop.X[order[:3]])
        self.evaluate(obl, 0, 3)
        worst = order[-3:][::-1]
        win = ops.better(self, obl.F, pop.F[worst])
        pop.buf[worst[win]] = obl.buf[win]

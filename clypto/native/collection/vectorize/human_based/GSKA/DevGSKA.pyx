#!/usr/bin/env python
# Created by "Thieu" at 16:58, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevGSKA(LegacyNativeOptimizer):
    """
    The developed version: Gaining Sharing Knowledge-based Algorithm (GSKA)

    Notes:
        + Third loop is removed, 2 parameters is removed
        + Solution represent junior or senior instead of dimension of solution
        + Equations is based vector, can handle large-scale problem
        + Apply the ideas of levy-flight and global best
        + Keep the better one after updating process

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.1, 0.5], percent of the best (p in the paper), default = 0.1
        + kr (float): [0.5, 0.9], knowledge ratio, default = 0.7

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import GSKA    >>> import numpy as np
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
    >>> model = GSKA.DevGSKA(epoch=1000, pop_size=50, pb = 0.1, kr = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    cdef public object pb
    cdef public object kr

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pb: float = 0.1,
        kr: float = 0.7,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, n: pop_size, m: clusters
            pb (float): percent of the best 0.1%, 0.8%, 0.1% (p in the paper), default = 0.1
            kr (float): knowledge ratio, default = 0.7
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pb", "kr"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pb = cy.validator(float, pb, (0, 1.0), "pb")
        self.kr = cy.validator(float, kr, (0, 1.0), "kr")

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        F = np.asarray(pop.F)
        g = np.array(self.g_best_x())
        me = np.arange(n)
        dd = int(np.ceil(n * (1.0 - epoch_c / self.epoch)))
        prev, nxt = ops.neighbors(n)
        rand_idx = ops.exclude(rng.integers(0, n - 3, size=n), np.stack([prev, me, nxt], axis=1))
        id1 = int(self.pb * n)
        id2 = int(id1 + n * (1 - 2 * self.pb))
        best, worst, mid = (ops.pick_range(self, lo, hi, me) for lo, hi in ((0, id1), (id2, n), (id1, id2)))
        U = rng.uniform(0, 1, (n, d))
        rb = ops.better(self, F[rand_idx], F)[:, None]
        mb = ops.better(self, F[mid], F)[:, None]
        senior = np.where(rb, X + U * (X[prev] - X[nxt] + X[rand_idx] - X), g + U * (X[rand_idx] - X))
        junior = np.where(mb, X + U * (X[best] - X[worst] + X[mid] - X), g + U * (X[mid] - X))
        pos = np.where((me < dd)[:, None], senior, junior)
        pos = np.where((rng.uniform(size=n) <= self.kr)[:, None], pos, rng.uniform(self.problem.lb, self.problem.ub, (n, d)))
        ops.step(self, pos)

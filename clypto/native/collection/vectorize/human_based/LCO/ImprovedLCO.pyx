#!/usr/bin/env python
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class ImprovedLCO(LegacyNativeOptimizer):
    """
    The improved version: Life Choice-based Optimization (ILCO)

    Notes:
        + The flow of the original LCO is kept.
        + Gaussian distribution and mutation mechanism are added
        + R1 parameter is removed

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import LCO    >>> import numpy as np
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
    >>> model = LCO.ImprovedLCO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    cdef public object pop_len

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
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pop_len = int(self.pop_size / 2)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d, pl = self.pop_len
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        f = epoch_c / self.epoch
        prob = rng.random((n, 1))
        na = int(np.ceil(np.sqrt(n)))
        top = (rng.random((n, na, 1)) * X[None, :na]).mean(axis=1)  # Eq. 1
        prev = np.vstack([g[None], X[:-1]])
        eq2 = X + f * rng.random((n, 1)) * (prev - X) + (1 - f) * rng.random((n, 1)) * (X[0] - X)  # Eq. 2-6
        eq3 = ub - (X - lb) * rng.random((n, 1))
        ops.step(self, np.where(prob > 0.875, top, np.where(prob < 0.7, eq2, eq3)))
        # the population is split: the better half explores around itself, the rest around the best
        pop = self.pop = self.pop.take(self.sorted_order(self.pop))
        X = pop.X
        local_best = np.array(X[0])
        ops.step(self, X[:pl] + rng.normal(0, 1, (pl, d)) * X[:pl], stop=pl)
        mean_s1 = np.mean(np.array(pop.X[:pl]), axis=0)
        ops.step(self, local_best + rng.uniform(0, 1, (pl, 1)) * mean_s1 * f, start=pl, stop=2 * pl)

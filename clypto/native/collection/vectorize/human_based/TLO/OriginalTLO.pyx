#!/usr/bin/env python
# Created by "Thieu" at 10:14, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.human_based.TLO.DevTLO cimport DevTLO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalTLO(DevTLO):
    """
    The original version of: Teaching Learning-based Optimization (TLO)

    Notes:
        + Third loops are removed
        + This version is inspired from above link
        + https://github.com/andaviaco/tblo

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import TLO    >>> import numpy as np
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
    >>> model = TLO.OriginalTLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R.V., Savsani, V.J. and Vakharia, D.P., 2011. Teaching–learning-based optimization: a novel method
    for constrained mechanical design optimization problems. Computer-aided design, 43(3), pp.303-315.
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
        super().__init__(epoch, pop_size, name=name, mode=mode)
        self.sort_flag = False
        self.is_parallelizable = False

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        # teacher phase
        TF = rng.integers(1, 3, size=(n, 1))  # 1 or 2 (never 3)
        ops.step(self, X + rng.uniform(0, 1, (n, d)) * (g - TF * np.mean(np.ascontiguousarray(X), axis=0)))
        # learner phase: learn from a random partner
        X = pop.X
        j = ops.others(self, n)[:, 0]
        ahead = ops.better(self, np.asarray(pop.F), np.asarray(pop.F)[j])[:, None]
        r = rng.random((n, d))
        ops.step(self, np.where(ahead, X + r * (X - X[j]), X + r * (X[j] - X)))

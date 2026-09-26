#!/usr/bin/env python
# Created by "Thieu" at 11:17, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport _LegacyAgent


from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalSSDO(LegacyNativeOptimizer):
    """
    The original version of: Social Ski-Driver Optimization (SSDO)

    Links:
       1. https://doi.org/10.1007/s00521-019-04159-z
       2. https://www.mathworks.com/matlabcentral/fileexchange/71210-social-ski-driver-ssd-optimization-algorithm-2019

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import SSDO    >>> import numpy as np
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
    >>> model = SSDO.OriginalSSDO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Tharwat, A. and Gabel, T., 2020. Parameters optimization of support vector machines for imbalanced
    data using social ski driver algorithm. Neural Computing and Applications, 32(11), pp.6925-6938.
    """

    cdef public object local_v
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

    cdef void initialization(self):
        LegacyNativeOptimizer.initialization(self)
        self.local_v = np.array(self.pop.X)  # local solution of every agent (its previous position)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = np.array(pop.X)
        c = 2 - epoch_c * (2.0 / self.epoch)  # a decreases linearly from 2 to 0
        pos_mean = np.mean(X[self.sorted_order(pop)[:3]])  # (the classic code takes the mean of all the elements)
        r1 = rng.uniform()  # r1, r2 is a random number in [0,1]
        r2 = rng.uniform()
        trig = np.sin(r1) if r2 <= 0.5 else np.cos(r1)  # sine or cosine move
        vel = c * trig * (self.local_v - X) + (2 - c) * trig * (pos_mean - X)
        pos = rng.normal(0, 1, (n, d)) * X + rng.random((n, 1)) * vel
        cand = pop.empty_like()
        cand.X[:] = self.correct_solution(pos)
        self.evaluate(cand, 0, n)
        ok = ops.better(self, np.asarray(cand.F), np.asarray(pop.F))
        self.local_v = np.where(ok[:, None], X, self.local_v)
        pop.buf[ok] = cand.buf[ok]

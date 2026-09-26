#!/usr/bin/env python
# Created by "Thieu" at 17:19, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.human_based.SPBO.OriginalSPBO cimport OriginalSPBO
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevSPBO(OriginalSPBO):
    """
    The developed version of: Student Psychology Based Optimization (SPBO)

    Notes:
        1. Replace uniform random number by normal random number
        2. Sort the population and select 1/3 pop size for each category

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import SPBO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = SPBO.DevSPBO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        super().__init__(epoch, pop_size, name=name, mode=mode)
        self.sort_flag = True

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        me = np.arange(n)
        lb, ub = self.problem.bounds.low, self.problem.bounds.up
        good, average = int(n / 3), 2 * int(n / 3)
        x_mean = np.mean(np.ascontiguousarray(X), axis=0)
        j = ops.others(self, n)[:, 0]
        first = g + rng.normal(0, 1, (n, d)) * (g - X[j])
        flip = rng.random(n) > rng.random(n)
        ra = rng.random((n, d))
        good_student = np.where(flip[:, None], g + rng.normal(0, 1, (n, d)) * (g - X), X + ra * (g - X) + (1 - ra) * (X - x_mean))
        average_student = X + rng.normal(0, 1, (n, d)) * (x_mean - X)
        pos = np.where((me < good)[:, None], good_student, np.where((me < average)[:, None], average_student, lb + rng.random((n, d)) * (ub - lb)))
        pos[0] = first[0]
        ops.step(self, pos)

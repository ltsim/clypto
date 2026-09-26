#!/usr/bin/env python
# Created by "Thieu" at 17:19, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalSPBO(VectorizeOptimizer):
    """
    The original version of: Student Psychology Based Optimization (SPBO)

    Notes:
        1. This algorithm is a weak algorithm in solving several problems
        2. It also consumes too much time because of ndim * pop_size updating times.

    Links:
       1. https://www.sciencedirect.com/science/article/abs/pii/S0965997820301484
       2. https://www.mathworks.com/matlabcentral/fileexchange/80991-student-psycology-based-optimization-spbo-algorithm

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
    >>> model = SPBO.OriginalSPBO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Das, B., Mukherjee, V., & Das, D. (2020). Student psychology based optimization algorithm: A new population based
    optimization algorithm for solving optimization problems. Advances in Engineering software, 146, 102804.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d, phase
        cdef object rng = self.generator
        g = np.array(self.g_best_x())
        me = np.arange(n)
        lb, ub = self.problem.bounds.low, self.problem.bounds.up
        for phase in range(d):  # one synchronous phase per dimension of the problem
            X = pop.X
            idx_best = np.asarray(pop.F).argmin() if self.problem.sense == "min" else np.asarray(pop.F).argmax()
            mid = rng.integers(1, n - 1)
            x_mean = np.mean(np.ascontiguousarray(X), axis=0)
            j = ops.others(self, n)[:, 0]
            sign = np.where(rng.integers(1, 3, size=(n, 1)) == 1, -1.0, 1.0)
            best_pos = g + sign * rng.random((n, d)) * (g - X[j])
            r = rng.random((n, d))
            flip = rng.random(n) > rng.random(n)
            good = np.where(flip[:, None], g + r * (g - X), X + r * (g - X) + rng.random((n, 1)) * (X - x_mean))
            bad = np.where(flip[:, None], X + r * (x_mean - X), lb + rng.random((n, d)) * (ub - lb))
            pos = np.where((me < mid)[:, None], good, bad)
            pos[idx_best] = best_pos[idx_best]
            ops.step(self, pos)

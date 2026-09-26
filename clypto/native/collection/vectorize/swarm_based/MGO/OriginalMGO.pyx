#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalMGO(LegacyNativeOptimizer):
    """
    The original version of: Mountain Gazelle Optimizer (MGO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0965997822001831
        2. https://www.mathworks.com/matlabcentral/fileexchange/118680-mountain-gazelle-optimizer

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import MGO    >>> import numpy as np
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
    >>> model = MGO.OriginalMGO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Gharehchopogh, F. S., Khodadadi, N., & Mirjalili, S. (2022). Mountain gazelle optimizer: a new
    nature-inspired metaheuristic algorithm for global optimization problems. Advances in Engineering Software, 174, 103282.
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
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        cdef NativePopulation cand
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        rows = np.arange(n)
        m3 = int(np.ceil(n / 3))
        idxs_rand = rng.random((n, n)).argsort(axis=1)[:, :m3]  # a random third of the herd per agent
        idx_rand = rng.integers(m3, n, size=n)
        M = X[idx_rand] * np.floor(rng.normal(size=(n, 1))) + X[idxs_rand].mean(axis=1) * np.ceil(rng.normal(size=(n, 1)))
        # vector of coefficients (n, 4, d)
        a2 = -1.0 + epoch * (-1.0 / self.epoch)
        u = rng.standard_normal((n, d))
        v = rng.standard_normal((n, d))
        cofi = np.empty((n, 4, d))
        cofi[:, 0] = rng.random((n, d))
        cofi[:, 1] = (a2 + 1) + rng.random((n, 1))
        cofi[:, 2] = a2 * rng.standard_normal((n, d))
        cofi[:, 3] = u * np.power(v, 2) * np.cos((rng.random((n, 1)) * 2) * u)
        A = rng.standard_normal((n, d)) * np.exp(2 - epoch * (2.0 / self.epoch))
        D = (np.abs(X) + np.abs(g)) * (2 * rng.random((n, 1)) - 1)
        pick = rng.integers(0, 4, size=(n, 3))
        c1, c2, c3 = (cofi[rows, pick[:, k]] for k in range(3))
        k = rng.integers(1, 3, size=(n, 6, 1))
        x2 = g - np.abs((k[:, 0] * M - k[:, 1] * X) * A) * c1
        x3 = M + c2 + (k[:, 2] * g - k[:, 3] * X[rng.integers(0, n, size=n)]) * c3
        x4 = X - D + (k[:, 4] * g - k[:, 5] * M) * cofi[rows, rng.integers(0, 4, size=n)]
        x1 = lb + rng.random((n, d)) * (ub - lb)
        cand = pop.take(np.zeros(4 * n, dtype=int))
        cand.X[:] = self.correct_solution(np.concatenate([x1, x2, x3, x4]))
        self.evaluate(cand, 0, 4 * n)
        both = pop.concat(cand)
        self.pop = both.take(self.sorted_order(both)[:n])

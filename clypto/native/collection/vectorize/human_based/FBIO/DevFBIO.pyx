#!/usr/bin/env python
# Created by "Thieu" at 08:57, 14/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevFBIO(LegacyNativeOptimizer):
    """
    The developed : Forensic-Based Investigation Optimization (FBIO)

    Notes:
        + Third loop is removed, the flowand a few equations is improved

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import FBIO    >>> import numpy as np
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
    >>> model = FBIO.DevFBIO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
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

    def probability__(self, list_fitness=None):  # Eq.(3) in FBI Inspired Meta-Optimization
        max1 = np.max(list_fitness)
        min1 = np.min(list_fitness)
        return (max1 - list_fitness) / (max1 - min1 + self.EPSILON)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        me = np.arange(n)
        # phase 1 (investigation): one coordinate moves relative to two random neighbours
        nc = rng.integers(0, d, size=n)
        nb1, nb2 = ops.two_others(self, n, 1)
        pos = np.array(X)
        pos[me, nc] = X[me, nc] + rng.normal(size=n) * (X[me, nc] - (X[nb1[:, 0], nc] + X[nb2[:, 0], nc]) / 2)
        ops.step(self, pos)
        # phase 2 (analysis): agents recombine with the best according to their fitness probability
        X = pop.X
        F = np.asarray(pop.F)
        prob = (F.max() - F) / (F.max() - F.min() + self.EPSILON)
        i = ops.k_others(self, n, 3)
        mix = g + X[i[:, 0]] + rng.uniform(size=(n, 1)) * (X[i[:, 1]] - X[i[:, 2]])
        mask2 = rng.random((n, d)) < 0.5
        pos = np.where((rng.uniform(size=n) > prob)[:, None], np.where(mask2, mix, X), lb + rng.random((n, d)) * (ub - lb))
        ops.step(self, pos)
        # phase 3 (pursuit): move towards the best
        X = pop.X
        ops.step(self, rng.uniform(0, 1, (n, d)) * X + rng.uniform(0, 1, (n, d)) * (g - X))
        # phase 4: compare with a random neighbour
        X = pop.X
        rr = ops.others(self, n)[:, 0]
        ahead = ops.better(self, np.asarray(pop.F), np.asarray(pop.F)[rr])[:, None]
        u = rng.uniform(0, 1, (n, d))
        v = rng.uniform(size=(n, 1))
        ops.step(self, np.where(ahead, X + u * (X[rr] - X) + v * (g - X[rr]), X + u * (X - X[rr]) + v * (g - X)))

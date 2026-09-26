#!/usr/bin/env python
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalPFA(VectorizeOptimizer):
    """
    The original version of: Pathfinder Algorithm (PFA)

    Links:
        1. https://doi.org/10.1016/j.asoc.2019.03.012

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import PFA    >>> import numpy as np
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
    >>> model = PFA.OriginalPFA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yapici, H. and Cetinkaya, N., 2019. A new meta-heuristic optimizer: Pathfinder algorithm.
    Applied soft computing, 78, pp.545-568.
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
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = np.array(pop.X)
        g = np.array(self.g_best_x())
        lb, ub = self.problem.bounds.low, self.problem.bounds.up
        alpha, beta = rng.uniform(1, 2, 2)
        A = rng.uniform(lb, ub) * np.exp(-2 * epoch_c / self.epoch)
        t = 1.0 - epoch_c * 1.0 / self.epoch
        space = ub - lb
        pos = np.empty((n, d))
        pos[0] = X[0] + 2 * rng.uniform() * (g - X[0]) + A  # the pathfinder
        # followers: attracted by all the others (k = 1..n-1), plus a distance-keeping term and a pull to the best
        Xk = X[1:]
        U = rng.uniform(size=(n, n - 1))
        sq = np.sum(X ** 2, axis=1)[:, None] + np.sum(Xk ** 2, axis=1)[None, :] - 2 * X @ Xk.T
        dist = np.sqrt(np.maximum(sq, 0.0)) / d  # (n, n - 1)
        t2 = alpha * (U @ Xk - U.sum(axis=1)[:, None] * X)
        t3 = t * (rng.uniform(size=(n, n - 1)) * dist).sum(axis=1)[:, None] / space
        t1 = beta * rng.uniform(size=(n, 1)) * (g - X)
        pos[1:] = ((X + t2 + t3 + t1) / n)[1:]
        ops.step(self, pos)

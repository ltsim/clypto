#!/usr/bin/env python
# Created by "Thieu" at 08:57, 14/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.human_based.FBIO.DevFBIO cimport DevFBIO
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalFBIO(DevFBIO):
    """
    The original version of: Forensic-Based Investigation Optimization (FBIO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2020.106339
        2. https://ww2.mathworks.cn/matlabcentral/fileexchange/76299-forensic-based-investigation-algorithm-fbi

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import FBIO    >>> import numpy as np
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
    >>> model = FBIO.OriginalFBIO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Chou, J.S. and Nguyen, N.M., 2020. FBI inspired meta-optimization. Applied Soft Computing, 93, p.106339.
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

    cdef object _amend_solution(self, object solution):
        rd = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up, size=np.shape(solution))
        condition = np.logical_and(
            self.problem.bounds.low <= solution, solution <= self.problem.bounds.up
        )
        return np.where(condition, solution, rd)

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.bounds.low, self.problem.bounds.up
        me = np.arange(n)
        # phase 1 (investigation): one coordinate moves relative to two random neighbours
        nc = rng.integers(0, d, size=n)
        nb1, nb2 = ops.two_others(self, n, 1)
        pos = np.array(X)
        pos[me, nc] = X[me, nc] + (rng.uniform(size=n) - 0.5) * 2 * (X[me, nc] - (X[nb1[:, 0], nc] + X[nb2[:, 0], nc]) / 2)
        ops.step(self, pos)
        # phase 2 (analysis): agents recombine with the best according to their fitness probability
        X = pop.X
        F = np.asarray(pop.F)
        prob = (F.max() - F) / (F.max() - F.min() + self.EPSILON)
        i = ops.k_others(self, n, 3)
        mix = g + X[i[:, 0]] + rng.uniform(size=(n, 1)) * (X[i[:, 1]] - X[i[:, 2]])
        mask2 = (rng.uniform(size=(n, d)) < rng.uniform(size=(n, d))) | (np.arange(d)[None, :] == (np.floor(rng.uniform(size=(n, 1)) * d) + 1))
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

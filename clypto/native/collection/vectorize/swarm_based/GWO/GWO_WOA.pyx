#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.swarm_based.GWO.OriginalGWO cimport OriginalGWO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.population cimport NativePopulation


cdef class GWO_WOA(OriginalGWO):
    """
    The original version of: Hybrid Grey Wolf - Whale Optimization Algorithm (GWO-WOA)

    Links:
        1. https://sci-hub.se/https://doi.org/10.1177/10775463211003402

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.GWO_WOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Obadina, O. O., Thaha, M. A., Althoefer, K., & Shaheed, M. H. (2022). Dynamic characterization of a master–slave
    robotic manipulator using a hybrid grey wolf–whale optimization algorithm. Journal of Vibration and Control, 28(15-16), 1992-2003.
    """

    cdef public double bb

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
        self.bb = 1.0
        self.sort_flag = False

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        a = 2.0 - epoch_c / self.epoch  # linearly decreased from 2 to 0
        best = X[self.sorted_order(pop)[:3]][None]
        R = rng.random((n, 6, d))
        A = a * (2 * R[:, :3] - 1)
        C = 2 * R[:, 3:]
        S = rng.random((n, 3, 1))
        da_plain = S[:, 0] * np.abs(C[:, 0] * best[:, 0] - X)
        da_spiral = S[:, 1] * np.exp(self.bb * (2 * S[:, 2] - 1)) * np.cos(2 * np.pi * (2 * S[:, 2] - 1)) * np.abs(C[:, 0] * best[:, 0] - X)
        da = np.where((rng.random(n) < 0.5)[:, None], da_plain, da_spiral)
        X1 = best[:, 0] - A[:, 0] * da
        X2 = best[:, 1] - A[:, 1] * np.abs(C[:, 1] * best[:, 1] - X)
        X3 = best[:, 2] - A[:, 2] * np.abs(C[:, 2] * best[:, 2] - X)
        ops.step(self, (X1 + X2 + X3) / 3.0)

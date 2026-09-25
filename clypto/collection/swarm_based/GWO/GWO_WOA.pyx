#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.swarm_based.GWO.OriginalGWO cimport OriginalGWO
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

    cdef void evolve(self, int epoch):
        # The number of draws per agent depends on the branch taken, so candidates
        # are built agent by agent (same draw order); evaluation and selection are batched.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        # linearly decreased from 2 to 0
        a = 2.0 - epoch / self.epoch
        best = pop.X[self.sorted_order(pop)[:3]]
        X, Xc = pop.X, cand.X
        for idx in range(n):
            A1 = a * (2 * self.generator.random(d) - 1)
            A2 = a * (2 * self.generator.random(d) - 1)
            A3 = a * (2 * self.generator.random(d) - 1)
            C1 = 2 * self.generator.random(d)
            C2 = 2 * self.generator.random(d)
            C3 = 2 * self.generator.random(d)
            if self.generator.random() < 0.5:
                da = self.generator.random() * np.abs(C1 * best[0] - X[idx])
            else:
                P, L = self.generator.random(), self.generator.uniform(-1, 1)
                da = P * np.exp(self.bb * L) * np.cos(2 * np.pi * L) * np.abs(C1 * best[0] - X[idx])
            X1 = best[0] - A1 * da
            X2 = best[1] - A2 * np.abs(C2 * best[1] - X[idx])
            X3 = best[2] - A3 * np.abs(C3 * best[2] - X[idx])
            Xc[idx] = self.correct_solution((X1 + X2 + X3) / 3.0)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

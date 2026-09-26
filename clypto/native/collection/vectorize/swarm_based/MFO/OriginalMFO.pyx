#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalMFO(LegacyNativeOptimizer):
    """
    The developed version: Moth-Flame Optimization (MFO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import MFO    >>> import numpy as np
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
    >>> model = MFO.OriginalMFO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2015. Moth-flame optimization algorithm: A novel nature-inspired
    heuristic paradigm. Knowledge-based systems, 89, pp.228-249.
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

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        # Number of flames Eq.(3.14) in the paper (linearly decreased)
        num_flame = round(self.pop_size - epoch * ((self.pop_size - 1) / self.epoch))
        # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1.0 + epoch * (-1.0 / self.epoch)
        Xs = pop.X[self.sorted_order(pop)]  # the flames
        g_best = Xs[0].copy()
        X = pop.X
        b = 1
        t = (a - 1) * self.generator.uniform(0, 1, n) + 1  # one draw per moth
        # exp/cos of a scalar per moth, as the classic loop did
        w = np.array([np.exp(b * t[idx]) * 1.0 for idx in range(n)])
        cs = np.array([np.cos(t[idx] * 2 * np.pi) for idx in range(n)])
        dist = np.abs(Xs - X)  # D in Eq.(3.13)
        # Eq.(3.12): a moth moves around its own flame, the last ones around the best flame
        temp_1 = dist * w[:, None] * cs[:, None] + Xs
        temp_2 = dist * w[:, None] * cs[:, None] + g_best
        pos_new = np.where((np.arange(n) < num_flame)[:, None], temp_1, temp_2)
        cand.X[:] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand, old_first=True)

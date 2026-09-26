#!/usr/bin/env python
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.human_based.LCO.OriginalLCO cimport OriginalLCO
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevLCO(OriginalLCO):
    """
    The developed version: Life Choice-based Optimization (LCO)

    Notes:
        + The flow is changed with if else statement.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r1 (float): [1.5, 4], coefficient factor, default = 2.35

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import LCO    >>> import numpy as np
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
    >>> model = LCO.DevLCO(epoch=1000, pop_size=50, r1 = 2.35)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        r1: float = 2.35,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r1 (float): coefficient factor
        """
        super().__init__(epoch, pop_size, r1, name=name, mode=mode)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        prob = rng.random((n, 1))
        # Eq. 1: mean of random fractions of the n_agents best positions
        top = (rng.random((n, self.n_agents, 1)) * X[None, :self.n_agents]).mean(axis=1)
        prev = np.vstack([g[None], X[:-1]])  # the better neighbour (the best for the first agent)
        f = epoch_c / self.epoch
        eq2 = X + rng.random((n, 1)) * (f * self.r1 * (prev - X)) + rng.random((n, 1)) * ((1 - f) * self.r1 * (X[0] - X))
        eq3 = lb + rng.random((n, d)) * (ub - lb)
        ops.step(self, np.where(prob > 0.875, top, np.where(prob < 0.7, eq2, eq3)))

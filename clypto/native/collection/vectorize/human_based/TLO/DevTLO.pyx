#!/usr/bin/env python
# Created by "Thieu" at 10:14, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class DevTLO(LegacyNativeOptimizer):
    """
    The developed version: Teaching Learning-based Optimization (TLO)

    Links:
       1. https://doi.org/10.5267/j.ijiec.2012.03.007

    Notes:
        + Use numpy np.array to make operations faster
        + The global best solution is used

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import TLO    >>> import numpy as np
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
    >>> model = TLO.DevTLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R. and Patel, V., 2012. An elitist teaching-learning-based optimization algorithm for solving
    complex constrained optimization problems. international journal of industrial engineering computations, 3(4), pp.535-560.
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
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        # teacher phase
        TF = rng.integers(1, 3, size=(n, 1))  # 1 or 2 (never 3)
        ops.step(self, X + rng.random((n, d)) * (g - TF * np.mean(np.ascontiguousarray(X), axis=0)))
        # learner phase: learn from a random partner
        X = pop.X
        j = ops.others(self, n)[:, 0]
        ahead = ops.better(self, np.asarray(pop.F), np.asarray(pop.F)[j])[:, None]
        r = rng.random((n, d))
        ops.step(self, np.where(ahead, X + r * (X - X[j]), X + r * (X[j] - X)))

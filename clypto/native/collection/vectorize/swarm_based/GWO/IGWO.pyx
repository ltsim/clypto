#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.swarm_based.GWO.OriginalGWO cimport OriginalGWO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class IGWO(OriginalGWO):
    """
    The original version of: Improved Grey Wolf Optimization (IGWO)

    Notes:
        1. Link: https://doi.org/10.1007/s00366-017-0567-1
        2. Inspired by: Mohammadtaher Abbasi (https://github.com/mtabbasi)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + a_min (float): Lower bound of a, default = 0.02
        + a_max (float): Upper bound of a, default = 2.2

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
    >>> model = GWO.IGWO(epoch=1000, pop_size=50, a_min = 0.02, a_max = 2.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kaveh, A. & Zakian, P.. (2018). Improved GWO algorithm for optimal design of truss structures.
    Engineering with Computers. 34. 10.1007/s00366-017-0567-1.
    """

    cdef public double a_min
    cdef public double a_max
    cdef public object growth_alpha
    cdef public object growth_delta

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        a_min: float = 0.02,
        a_max: float = 2.2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a_min (float): Lower bound of a, default = 0.02
            a_max (float): Upper bound of a, default = 2.2
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "a_min", "a_max"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.a_min = cy.validator(float, a_min, (0.0, 1.6), "a_min")
        self.a_max = cy.validator(float, a_max, [1.0, 4.0], "a_max")
        self.growth_alpha = 2
        self.growth_delta = 3

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t n = pop.n, d = pop.d
        best = pop.X[self.sorted_order(pop)[:3]][None]
        a_alpha = self.a_max * np.exp((epoch / self.epoch) ** self.growth_alpha * np.log(self.a_min / self.a_max))
        a_delta = self.a_max * np.exp((epoch / self.epoch) ** self.growth_delta * np.log(self.a_min / self.a_max))
        a_beta = (a_alpha + a_delta) * 0.5
        a = np.array([a_alpha, a_beta, a_delta])[None, :, None]
        R = self.generator.random((n, 6, d))  # per agent: A1..A3 then C1..C3 draws
        A = a * (2 * R[:, :3] - 1)
        C = 2 * R[:, 3:]
        Xs = best - A * np.abs(C * best - pop.X[:, None, :])
        cand.X[:] = self.correct_solution((Xs[:, 0] + Xs[:, 1] + Xs[:, 2]) / 3.0)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class ER_GWO(LegacyNativeOptimizer):
    """
    The original version of: Efficient and Robust Grey Wolf Optimizer (ER-GWO)

    Notes:
        + Slow convergence speed due to the (miu_factor)^(iteration) ==> Big number
        + Three more parameters than original GWO, increase the complexity of the algorithm.

    Links:
        1. https://doi.org/10.1007/s00500-019-03939-y

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
    >>> model = GWO.ER_GWO(epoch=1000, pop_size=50, a_initial=2.0, a_final=0.0, miu_factor=1.0001)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Long, W., Cai, S., Jiao, J. et al. An efficient and robust grey wolf optimizer algorithm for large-scale numerical optimization. Soft Comput 24, 997–1026 (2020).
    """

    cdef public double a_initial
    cdef public double a_final
    cdef public double miu_factor

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        a_initial: float = 2.0,
        a_final: float = 0.0,
        miu_factor: float = 1.0001,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a_initial (float): initial value of coefficient a, default = 2.0
            a_final (float): final value of coefficient a, default = 0.0
            miu_factor (float): nonlinear coefficient for equation (8), default = 1.0001
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "a_initial", "a_final", "miu_factor"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.a_initial = cy.validator(float, a_initial, [0.0, 10.0], "a_initial")
        self.a_final = cy.validator(float, a_final, [0.0, self.a_initial], "a_final")
        self.miu_factor = cy.validator(float, miu_factor, [1.0001, 1.01], "miu_factor")

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        # linearly decreased from 2 to 0
        a = self.a_initial - (self.a_initial - self.a_final) * self.miu_factor**epoch
        best = pop.X[self.sorted_order(pop)[:3]][None]
        R = self.generator.random((n, 6, d))  # per agent: A1..A3 then C1..C3 draws
        A = a * (2 * R[:, :3] - 1)
        C = 2 * R[:, 3:]
        Xs = best - A * np.abs(C * best - pop.X[:, None, :])
        pos = np.empty((n, d))
        for idx in range(n):
            X1, X2, X3 = Xs[idx]
            dist1 = np.linalg.norm(X1)
            dist2 = np.linalg.norm(X2)
            dist3 = np.linalg.norm(X3)
            total = dist1 + dist2 + dist3
            if total == 0:
                # Avoid division by zero
                pos[idx] = (X1 + X2 + X3) / 3.0
            else:
                # Normalize distances to avoid division by zero
                pos[idx] = (X1 * dist1 + X2 * dist2 + X3 * dist3) / total
        cand.X[:] = self.correct_solution(pos)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

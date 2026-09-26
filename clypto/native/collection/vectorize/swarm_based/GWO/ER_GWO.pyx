#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class ER_GWO(VectorizeOptimizer):
    """
    The original version of: Efficient and Robust Grey Wolf Optimizer (ER-GWO)

    Notes:
        + Slow convergence speed due to the (miu_factor)^(iteration) ==> Big number
        + Three more parameters than original GWO, increase the complexity of the algorithm.

    Links:
        1. https://doi.org/10.1007/s00500-019-03939-y

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import GWO    >>> import numpy as np
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
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "a_initial", "a_final", "miu_factor"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.a_initial = cy.validator(float, a_initial, [0.0, 10.0], "a_initial")
        self.a_final = cy.validator(float, a_final, [0.0, self.a_initial], "a_final")
        self.miu_factor = cy.validator(float, miu_factor, [1.0001, 1.01], "miu_factor")

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        X = pop.X
        a = self.a_initial - (self.a_initial - self.a_final) * self.miu_factor**epoch_c
        best = X[self.sorted_order(pop)[:3]][None]
        R = self.generator.random((n, 6, d))  # per agent: A1..A3 then C1..C3 draws
        Xs = best - (a * (2 * R[:, :3] - 1)) * np.abs(2 * R[:, 3:] * best - X[:, None, :])  # (n, 3, d)
        dist = np.linalg.norm(Xs, axis=2)  # (n, 3)
        total = dist.sum(axis=1)
        weighted = (Xs * dist[..., None]).sum(axis=1) / np.where(total == 0, 1.0, total)[:, None]
        ops.step(self, np.where((total == 0)[:, None], Xs.sum(axis=1) / 3.0, weighted))

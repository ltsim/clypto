#!/usr/bin/env python
# Created by "Thieu" at 15:53, 07/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalAO(LegacyNativeOptimizer):
    """
    The original version of: Aquila Optimization (AO)

    Links:
        1. https://doi.org/10.1016/j.cie.2021.107250

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import AO    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = AO.OriginalAO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A.A., Al-Qaness, M.A. and Gandomi, A.H., 2021.
    Aquila optimizer: a novel meta-heuristic optimization algorithm. Computers & Industrial Engineering, 157, p.107250.
    """

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
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
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        alpha = delta = 0.1
        g1 = 2 * rng.random() - 1  # Eq. 16
        g2 = 2 * (1 - epoch / self.epoch)  # Eq. 17
        dim_list = np.arange(1, d + 1)
        r = 10 + 0.00565 * dim_list
        phi = -0.005 * dim_list + 3 * np.pi / 2
        x = r * np.sin(phi)  # Eq.(9)
        y = r * np.cos(phi)  # Eq.(10)
        QF = epoch ** ((2 * rng.random() - 1) / (1 - self.epoch) ** 2)  # Eq.(15) quality function
        x_mean = np.mean(np.array(X), axis=0)
        levy_step = self.get_levy_flight_step(beta=1.5, multiplier=1.0, size=(n, 1), case=-1)
        R = rng.random((n, 5, 1))
        if epoch <= (2 / 3) * self.epoch:  # Eq. 3, 4
            other = X[ops.others(self, n)[:, 0]]
            pos = np.where(R[:, 0] < 0.5,
                           g * (1 - epoch / self.epoch) + R[:, 1] * (x_mean - g),
                           g * levy_step + other + R[:, 1] * (y - x))  # Eq. 5
        else:
            pos = np.where(R[:, 0] < 0.5,
                           alpha * (g - x_mean) - R[:, 1] * (R[:, 2] * (ub - lb) + lb) * delta,  # Eq. 13
                           QF * g - (g2 * X * R[:, 3]) - g2 * levy_step + R[:, 4] * g1)  # Eq. 14
        ops.step(self, pos)

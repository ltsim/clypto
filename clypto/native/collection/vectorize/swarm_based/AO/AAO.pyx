#!/usr/bin/env python
# Created by "Thieu" at 15:53, 07/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class AAO(LegacyNativeOptimizer):
    """
    The original version of: Adaptive Aquila Optimizer (AAO)

    Links:
        1. https://doi.org/10.1016/j.rineng.2024.103261

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import AO    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = AO.AAO(epoch=1000, pop_size=50, sharpness=10.0, sigmoid_midpoint=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Al-Selwi, S. M., Hassan, M. F., Abdulkadir, S. J., Ragab, M. G., Alqushaibi, A., & Sumiea, E. H. (2024).
    Smart grid stability prediction using adaptive aquila optimizer and ensemble stacked bilstm. Results in Engineering, 24, 103261.
    """

    cdef public object sharpness
    cdef public object sigmoid_midpoint

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        sharpness = 10.0,
        sigmoid_midpoint = 0.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            sharpness (float): is a positive variable that controls the sharpness of the transition between exploration and exploitation, default is 10.0, Valid range: [0.1, 10000.0].
            sigmoid_midpoint (float): a variable that controls the midpoint of the sigmoid function as it determines when the transition should be applied, default is 0.5, Valid range: [0.0, 1.0].
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "sharpness", "sigmoid_midpoint"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.sharpness = cy.validator(float, sharpness, [0.1, 10000.0], "sharpness")
        self.sigmoid_midpoint = cy.validator(float, sigmoid_midpoint, [0.0, 1.0], "sigmoid_midpoint")

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
        sigmoid_factor = 1 / (1 + np.exp(-self.sharpness * (epoch / self.epoch - self.sigmoid_midpoint)))
        R = rng.random((n, 6, 1))
        other = X[ops.others(self, n)[:, 0]]
        explore = np.where(R[:, 1] < 0.5,
                           g * (1 - epoch / self.epoch) + R[:, 2] * (x_mean - g),  # Eq. (3) and Eq. (4)
                           g * levy_step + other + R[:, 2] * (y - x))  # Eq. 5
        exploit = np.where(R[:, 1] < 0.5,
                           alpha * (g - x_mean) - R[:, 2] * (R[:, 3] * (ub - lb) + lb) * delta,  # Eq. 13
                           QF * g - (g2 * X * R[:, 4]) - g2 * levy_step + R[:, 5] * g1)  # Eq. 14
        ops.step(self, np.where(R[:, 0] <= (1 - sigmoid_factor), explore, exploit))

#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalBES(LegacyNativeOptimizer):
    """
    The original version of: Bald Eagle Search (BES)

    Links:
        1. https://doi.org/10.1007/s10462-019-09732-5

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + a_factor (int): default: 10, determining the corner between point search in the central point, in [5, 10]
        + R_factor (float): default: 1.5, determining the number of search cycles, in [0.5, 2]
        + alpha (float): default: 2, parameter for controlling the changes in position, in [1.5, 2]
        + c1 (float): default: 2, in [1, 2]
        + c2 (float): c1 and c2 increase the movement intensity of bald eagles towards the best and centre points

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import BES    >>> import numpy as np
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
    >>> model = BES.OriginalBES(epoch=1000, pop_size=50, a_factor = 10, R_factor = 1.5, alpha = 2.0, c1 = 2.0, c2 = 2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Alsattar, H.A., Zaidan, A.A. and Zaidan, B.B., 2020. Novel meta-heuristic bald eagle
    search optimisation algorithm. Artificial Intelligence Review, 53(3), pp.2237-2264.
    """

    cdef public object a_factor
    cdef public object R_factor
    cdef public object alpha
    cdef public object c1
    cdef public object c2

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        a_factor: int = 10,
        R_factor: float = 1.5,
        alpha: float = 2.0,
        c1: float = 2.0,
        c2: float = 2.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a_factor (int): default: 10, determining the corner between point search in the central point, in [5, 10]
            R_factor (float): default: 1.5, determining the number of search cycles, in [0.5, 2]
            alpha (float): default: 2, parameter for controlling the changes in position, in [1.5, 2]
            c1 (float): default: 2, in [1, 2]
            c2 (float): c1 and c2 increase the movement intensity of bald eagles towards the best and centre points
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "a_factor", "R_factor", "alpha", "c1", "c2"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.a_factor = cy.validator(int, a_factor, [2, 20], "a_factor")
        self.R_factor = cy.validator(float, R_factor, [0.1, 3.0], "R_factor")
        self.alpha = cy.validator(float, alpha, [0.5, 3.0], "alpha")
        self.c1 = cy.validator(float, c1, (0, 4.0), "c1")
        self.c2 = cy.validator(float, c2, (0, 4.0), "c2")

    def create_x_y_x1_y1__(self):
        ## Eq. 2
        phi = self.a_factor * np.pi * self.generator.uniform(0, 1, self.pop_size)
        r = phi + self.R_factor * self.generator.uniform(0, 1, self.pop_size)
        xr, yr = r * np.sin(phi), r * np.cos(phi)
        ## Eq. 3
        r1 = phi1 = self.a_factor * np.pi * self.generator.uniform(0, 1, self.pop_size)
        xr1, yr1 = r1 * np.sinh(phi1), r1 * np.cosh(phi1)
        x_list = xr / np.max(xr)
        y_list = yr / np.max(yr)
        x1_list = xr1 / np.max(xr1)
        y1_list = yr1 / np.max(yr1)
        return x_list, y_list, x1_list, y1_list

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        x_list, y_list, x1_list, y1_list = self.create_x_y_x1_y1__()
        # 1. Select space
        pos_mean = np.mean(np.array(X), axis=0)
        ops.step(self, g + self.alpha * rng.uniform(size=(n, 1)) * (pos_mean - X))
        # 2. Search in space
        pos_mean = np.mean(np.array(pop.X), axis=0)
        X = pop.X
        rand_agent = X[ops.others(self, n)[:, 0]]
        ops.step(self, X + y_list[:, None] * (X - rand_agent) + x_list[:, None] * (X - pos_mean))
        # 3. Swoop
        pos_mean = np.mean(np.array(pop.X), axis=0)
        X = pop.X
        ops.step(self, rng.uniform(size=(n, 1)) * g + x1_list[:, None] * (X - self.c1 * pos_mean) + y1_list[:, None] * (X - self.c2 * g))

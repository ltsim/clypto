#!/usr/bin/env python
# Created by "Thieu" at 17:38, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalCircleSA(LegacyNativeOptimizer):
    """
    The original version of: Circle Search Algorithm (CircleSA)

    Links:
        1. https://doi.org/10.3390/math10101626
        2. https://www.mdpi.com/2227-7390/10/10/1626

    Examples
    ~~~~~~~~
    >>> from clypto.collection.math_based import CircleSA    >>> import numpy as np
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
    >>> model = CircleSA.OriginalCircleSA(epoch=1000, pop_size=50, c_factor=0.8)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Qais, M. H., Hasanien, H. M., Turky, R. A., Alghuwainem, S., Tostado-Véliz, M., & Jurado, F. (2022).
    Circle Search Algorithm: A Geometry-Based Metaheuristic Optimization Algorithm. Mathematics, 10(10), 1626.
    """

    cdef public double c_factor

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        c_factor = 0.8,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "c_factor"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.c_factor = cy.validator(float, c_factor, (0, 1.0), "c_factor")

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t idx, n = pop.n
        a = np.pi - np.pi * (<object>(epoch / self.epoch)) ** 2  # Eq. 8
        p = 1 - 0.9 * (<object>(epoch / self.epoch)) ** 0.5
        threshold = self.c_factor * self.epoch
        g = np.array(self.g_best_x())
        X = pop.X
        if epoch > threshold:
            R = self.generator.random((n, 2))  # per agent: w draw, then the tan draw
        else:
            R = self.generator.random((n, 1))
        w = a * R[:, 0] - a
        # np.tan on a scalar, as the per-agent classic loop did
        if epoch > threshold:
            tan = np.array([np.tan(w[idx] * R[idx, 1]) for idx in range(n)])
            x_new = g + (g - X) * tan[:, None]
        else:
            tan = np.array([np.tan(w[idx] * p) for idx in range(n)])
            x_new = g - (g - X) * tan[:, None]
        self.pop = self.new_population(self.correct_solution(x_new))

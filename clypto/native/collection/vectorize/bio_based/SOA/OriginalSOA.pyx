#!/usr/bin/env python
# Created by "Thieu" at 17:21, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalSOA(LegacyNativeOptimizer):
    """
    The original version: Seagull Optimization Algorithm (SOA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0950705118305768

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + fc (float): [1.0, 10.0] -> better [1, 5], freequency of employing variable A (A linear decreased from fc to 0), default = 2

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import SOA    >>> import numpy as np
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
    >>> model = SOA.OriginalSOA(epoch=1000, pop_size=50, fc = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Dhiman, G., & Kumar, V. (2019). Seagull optimization algorithm: Theory and its applications
    for large-scale industrial engineering problems. Knowledge-based systems, 165, 169-196.
    """

    cdef public object fc

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        fc = 2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "fc"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.fc = cy.validator(float, fc, [1.0, 10.0], "fc")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        A = self.fc - epoch * self.fc / self.epoch  # Eq. 6
        uu = vv = 1
        for idx in range(0, self.pop_size):
            B = 2 * A ** 2 * self.generator.random()  # Eq. 8
            M = B * (g_best - Xp[idx])  # Eq. 7
            C = A * Xp[idx]  # Eq. 5
            D = np.abs(C + M)  # Eq. 9
            k = self.generator.uniform(0, 2 * np.pi)
            r = uu * np.exp(k * vv)
            xx = r * np.cos(k)
            yy = r * np.sin(k)
            zz = r * k
            pos_new = xx * yy * zz * D + g_best  # Eq. 14
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        self.pop = cand

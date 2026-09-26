#!/usr/bin/env python
# Created by "Thieu" at 17:21, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalSOA(VectorizeOptimizer):
    """
    The original version: Seagull Optimization Algorithm (SOA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0950705118305768

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + fc (float): [1.0, 10.0] -> better [1, 5], freequency of employing variable A (A linear decreased from fc to 0), default = 2

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.bio_based import SOA    >>> import numpy as np
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
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "fc"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.fc = cy.validator(float, fc, [1.0, 10.0], "fc")

    def _evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        A = self.fc - epoch * self.fc / self.epoch  # Eq. 6
        B = 2 * A ** 2 * rng.random((n, 1))  # Eq. 8
        D = np.abs(A * X + B * (g - X))  # Eqs. 5, 7, 9
        k = rng.uniform(0, 2 * np.pi, (n, 1))
        r = np.exp(k)
        ops.replace(self, r * np.cos(k) * r * np.sin(k) * r * k * D + g)  # Eq. 14

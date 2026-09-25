#!/usr/bin/env python
# Created by "Thieu" at 18:09, 13/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalSHIO(LegacyNativeOptimizer):
    """
    The original version of: Success History Intelligent Optimizer (SHIO)

    Links:
        1. https://link.springer.com/article/10.1007/s11227-021-04093-9
        2. https://www.mathworks.com/matlabcentral/fileexchange/122157-success-history-intelligent-optimizer-shio

    Notes:
        1. The algorithm is designed with simplicity and ease of implementation in mind, utilizing basic operators.
        2. This algorithm has several limitations and weak when dealing with several problems
        3. The algorithm's convergence is slow. The Matlab code has many errors and unnecessary things.

    Examples
    ~~~~~~~~
    >>> from clypto.collection.math_based import SHIO    >>> import numpy as np
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
    >>> model = SHIO.OriginalSHIO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Fakhouri, H. N., Hamad, F., & Alawamrah, A. (2022). Success history intelligent optimizer. The Journal of Supercomputing, 1-42.
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

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        b = pop.X[self.sorted_order(pop)[:3]]  # b1, b2, b3
        a_list, a = np.empty(n), 1.5
        for idx in range(n):  # a decreases by 0.04 per agent
            a = a - 0.04
            a_list[idx] = a
        a = a_list[:, None]
        # per agent and leader: one draw for the coefficient, one inside the abs
        R = self.generator.random((n, 3, 2, d))
        Xs = b[None] + (a[:, None] * 2 * R[:, :, 0] - a[:, None]) * np.abs(R[:, :, 1] * b[None] - pop.X[:, None, :])
        cand.X[:] = self.correct_solution((Xs[:, 0] + Xs[:, 1] + Xs[:, 2]) / 3)
        self.evaluate(cand, 0, n)
        self.pop = cand

#!/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalWOA(LegacyNativeOptimizer):
    """
    The original version of: Whale Optimization Algorithm (WOA)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2016.01.008
        2. https://mathworks.com/matlabcentral/fileexchange/55667-the-whale-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import WOA    >>> import numpy as np
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
    >>> model = WOA.OriginalWOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm. Advances in engineering software, 95, pp.51-67.
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

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        # The classic code only refreshes the *fitness* of the population with the candidates' values in the
        # sequential mode (positions stay) and replaces the population in swarm modes; both are kept.
        g = np.array(self.g_best_x())
        a = 2 - 2 * epoch / self.epoch  # linearly decreased from 2 to 0
        a2 = -1 + epoch * ((-1) / self.epoch)
        R = rng.random((n, 4))
        A = (a * (2 * R[:, 0] - a))[:, None]
        C = (2 * R[:, 1])[:, None]
        l = ((a2 - 1) * R[:, 2] + 1)[:, None]
        others = ops.others(self, n, d)  # one random other whale per dimension
        x_rand = X[others, np.arange(d)[None, :]]
        search = x_rand - A * np.abs(C * x_rand - X)
        encircle = g - A * np.abs(C * g - X)
        spiral = np.abs(g - X) * np.exp(l) * np.cos(l * 2 * np.pi) + g
        pos = np.where((R[:, 3] < 0.5)[:, None], np.where(np.abs(A) >= 1, search, encircle), spiral)
        cdef NativePopulation cand = pop.empty_like()
        cand.X[:] = self.correct_solution(pos)
        self.evaluate(cand, 0, n)
        if self.mode in self.AVAILABLE_MODES:
            self.pop = cand
        else:
            pop.buf[:, :pop.cX] = cand.buf[:, :pop.cX]

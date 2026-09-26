#!/usr/bin/env python
# Created by "Thieu" at 16:30, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.swarm_based.JA.DevJA cimport DevJA
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalJA(DevJA):
    """
    The original version of: Jaya Algorithm (JA)

    Links:
        1. https://www.growingscience.com/ijiec/Vol7/IJIEC_2015_32.pdf

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import JA    >>> import numpy as np
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
    >>> model = JA.OriginalJA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R., 2016. Jaya: A simple and new optimization algorithm for solving constrained and
    unconstrained optimization problems. International Journal of Industrial Engineering Computations, 7(1), pp.19-34.
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
        super().__init__(epoch, pop_size, name=name, mode=mode)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t n = pop.n, d = pop.d
        order = self.sorted_order(pop)
        g_best, g_worst = pop.X[order[0]], pop.X[order[n - 1]]
        # per agent: two uniform(0, 1, d) draws (the pop is only read, then replaced by the candidates)
        R = self.generator.uniform(0, 1, (n, 2, d))
        X = pop.X
        pos_new = X + R[:, 0] * (g_best - np.abs(X)) - R[:, 1] * (g_worst - np.abs(X))
        cand.X[:] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        self.pop = cand

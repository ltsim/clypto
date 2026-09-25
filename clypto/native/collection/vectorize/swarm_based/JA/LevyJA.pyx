#!/usr/bin/env python
# Created by "Thieu" at 16:30, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.swarm_based.JA.DevJA cimport DevJA
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class LevyJA(DevJA):
    """
    The original version of: Levy-flight Jaya Algorithm (LJA)

    Notes
        + All third loops in this version also are removed
        + The beta value of Levy-flight equal to 1.8 as the best value in the paper.
        + https://doi.org/10.1016/j.eswa.2020.113902

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import JA    >>> import numpy as np
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
    >>> model = JA.LevyJA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Iacca, G., dos Santos Junior, V.C. and de Melo, V.V., 2021. An improved Jaya optimization
    algorithm with Lévy flight. Expert Systems with Applications, 165, p.113902.
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
        self.sort_flag = False

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        order = self.sorted_order(pop)
        best, worst = X[order[0]], X[order[n - 1]]
        L1 = self.get_levy_flight_step(multiplier=1.0, beta=1.8, size=(n, 1), case=-1)
        L2 = self.get_levy_flight_step(multiplier=1.0, beta=1.8, size=(n, 1), case=-1)
        ops.step(self, X + np.abs(L1) * (best - np.abs(X)) - np.abs(L2) * (worst - np.abs(X)))

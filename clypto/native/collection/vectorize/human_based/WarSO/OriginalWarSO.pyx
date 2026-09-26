#!/usr/bin/env python
# Created by "Thieu" at 17:41, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalWarSO(LegacyNativeOptimizer):
    """
    The original version of: War Strategy Optimization (WarSO) algorithm

    Links:
       1. https://www.researchgate.net/publication/358806739_War_Strategy_Optimization_Algorithm_A_New_Effective_Metaheuristic_Algorithm_for_Global_Optimization

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + rr (float): [0.1, 0.9], the probability of switching position updating, default=0.1

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import WarSO    >>> import numpy as np
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
    >>> model = WarSO.OriginalWarSO(epoch=1000, pop_size=50, rr=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ayyarao, Tummala SLV, and Polamarasetty P. Kumar. "Parameter estimation of solar PV models with a new proposed
    war strategy optimization algorithm." International Journal of Energy Research (2022).
    """

    cdef public object rr
    cdef public object wl
    cdef public object wg

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        rr: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            rr (float): the probability of switching position updating, default=0.1
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "rr"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.rr = cy.validator(float, rr, (0.0, 1.0), "rr")

    cdef void initialize_variables(self):
        self.wl = 2 * np.ones(self.pop_size)
        self.wg = np.zeros(self.pop_size)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        order = self.sorted_order(pop)
        self.wl = self.wl[order]
        self.wg = self.wg[order]
        Xs = X[order]  # the sorted warriors
        com = rng.permutation(n)
        r1 = rng.random((n, 1))
        pos = np.where(r1 < self.rr,
                       2 * r1 * (g - X[com]) + self.wl[:, None] * rng.random((n, 1)) * (Xs - X),
                       2 * r1 * (Xs - g) + rng.random((n, 1)) * (self.wl[:, None] * g - X))
        before = np.array(pop.F)
        ops.step(self, pos)
        improved = ops.better(self, np.asarray(self.pop.F), before)
        self.wg[improved] += 1
        self.wl[improved] = 1 * self.wl[improved] * (1 - self.wg[improved] / self.epoch) ** 2

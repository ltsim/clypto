#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class DevGCO(LegacyNativeOptimizer):
    """
    The developed version: Germinal Center Optimization (GCO)

    Notes:
        + The global best solution and 2 random solutions are used instead of randomizing 3 solutions

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + cr (float): [0.5, 0.95], crossover rate, default = 0.7 (Same as DE algorithm)
        + wf (float): [1.0, 2.0], weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)

    Examples
    ~~~~~~~~
    >>> from clypto.collection.system_based import GCO    >>> import numpy as np
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
    >>> model = GCO.DevGCO(epoch=1000, pop_size=50, cr = 0.7, wf = 1.25)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        cr: float = 0.7,
        wf: float = 1.25,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            cr (float): crossover rate, default = 0.7 (Same as DE algorithm)
            wf (float): weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "cr", "wf"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.cr = cy.validator(float, cr, (0, 1.0), "cr")
        self.wf = cy.validator(float, wf, (0, 3.0), "wf")

    cdef void initialize_variables(self):
        self.dyn_list_cell_counter = np.ones(self.pop_size)  # CEll Counter
        self.dyn_list_life_signal = 70 * np.ones(
            self.pop_size
        )  # 70% to duplicate, and 30% to die  # LIfe-Signal

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        g_best = np.array(self.g_best_x())
        Xp, Xc = pop.X, cand.X
        ## Dark-zone process    (can be parallelization)
        for idx in range(0, self.pop_size):
            if self.generator.uniform(0, 100) < self.dyn_list_life_signal[idx]:
                self.dyn_list_cell_counter[idx] += 1
            else:
                self.dyn_list_cell_counter[idx] = 1
            # Mutate process
            r1, r2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            pos_new = g_best + self.wf * (Xp[r2] - Xp[r1])
            condition = self.generator.random(d) < self.cr
            pos_new = np.where(condition, pos_new, Xp[idx])
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        better = cand.F < pop.F
        if self.problem.minmax != "min":
            better = ~better
        self.dyn_list_cell_counter[better] += 10
        rows = np.flatnonzero(better)
        pop.buf[rows] = cand.buf[rows]
        ## Light-zone process   (no needs parallelization)
        fit_list = np.array(pop.F)
        fit_max = np.max(fit_list)
        fit_min = np.min(fit_list)
        self.dyn_list_cell_counter[:] = 10
        self.dyn_list_cell_counter += 10 * (pop.F - fit_max) / (fit_min - fit_max + self.EPSILON)

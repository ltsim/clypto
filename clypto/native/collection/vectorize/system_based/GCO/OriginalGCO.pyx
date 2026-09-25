#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.system_based.GCO.DevGCO cimport DevGCO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalGCO(DevGCO):
    """
    The original version of: Germinal Center Optimization (GCO)

    Links:
        1. https://doi.org/10.2991/ijcis.2018.25905179
        2. https://www.atlantis-press.com/journals/ijcis/25905179/view

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
    >>> model = GCO.OriginalGCO(epoch=1000, pop_size=50, cr = 0.7, wf = 1.25)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Villaseñor, C., Arana-Daniel, N., Alanis, A.Y., López-Franco, C. and Hernandez-Vargas, E.A., 2018.
    Germinal center optimization algorithm. International Journal of Computational Intelligence Systems, 12(1), p.13.
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
        super().__init__(epoch, pop_size, cr, wf, name=name, mode=mode)
        self.is_parallelizable = False

    cdef void evolve(self, int epoch):
        # Every agent draws its parents from the population updated so far, with the
        # cell counters as weights: sequential on the buffer rows.
        cdef NativePopulation pop = self.pop
        cdef NativeTarget tar
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        Xp = pop.X
        ## Dark-zone process (can't be parallelization)
        for idx in range(0, self.pop_size):
            if self.generator.uniform(0, 100) < self.dyn_list_life_signal[idx]:
                self.dyn_list_cell_counter[idx] += 1
            elif self.dyn_list_cell_counter[idx] > 1:
                self.dyn_list_cell_counter[idx] -= 1
            # Mutate process
            p = self.dyn_list_cell_counter / np.sum(self.dyn_list_cell_counter)
            r1, r2, r3 = self.generator.choice(list(set(range(0, self.pop_size))), 3, replace=False, p=p)
            pos_new = Xp[r1] + self.wf * (Xp[r2] - Xp[r3])
            condition = self.generator.random(d) < self.cr
            pos_new = np.where(condition, pos_new, Xp[idx])
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)
            # for each pos_new, generate the fitness
            if self.compare_fitness(tar.fitness, pop.F[idx], self.problem.minmax):
                ops.set_row(pop, idx, pos_new, tar)
                self.dyn_list_life_signal[idx] += 10
        ## Light-zone process   (no needs parallelization)
        self.dyn_list_life_signal -= 10
        fit_list = np.array(pop.F)
        fit_max = np.max(fit_list)
        fit_min = np.min(fit_list)
        fit = (fit_list - fit_max) / (fit_min - fit_max + self.EPSILON)
        if self.problem.minmax != "min":
            fit = 1 - fit
        self.dyn_list_life_signal += 10 * fit

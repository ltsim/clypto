#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.system_based.GCO.DevGCO cimport DevGCO
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


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
    >>> from clypto.native.collection.vectorize.system_based import GCO    >>> import numpy as np
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

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        cnt, life = self.dyn_list_cell_counter, self.dyn_list_life_signal
        up = rng.uniform(0, 100, n) < life
        cnt[:] = np.where(up, cnt + 1, np.where(cnt > 1, cnt - 1, cnt))
        p = cnt / np.sum(cnt)
        # three distinct agents per agent, drawn with probabilities p (weighted sampling without replacement)
        keys = rng.random((n, n)) ** (1.0 / p[None, :])
        r = np.argpartition(-keys, 2, axis=1)[:, :3]
        pos = np.where(rng.random((n, d)) < self.cr, X[r[:, 0]] + self.wf * (X[r[:, 1]] - X[r[:, 2]]), X)
        before = np.array(pop.F)
        ops.step(self, pos)
        life[ops.better(self, np.asarray(pop.F), before)] += 10
        life -= 10
        fit_list = np.array(pop.F)
        fit_max = np.max(fit_list)
        fit_min = np.min(fit_list)
        fit = (fit_list - fit_max) / (fit_min - fit_max + self.EPSILON)
        if self.problem.sense != "min":
            fit = 1 - fit
        life += 10 * fit

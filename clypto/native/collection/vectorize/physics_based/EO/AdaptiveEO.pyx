#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.physics_based.EO.OriginalEO cimport OriginalEO
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class AdaptiveEO(OriginalEO):
    """
    The original version of: Adaptive Equilibrium Optimization (AEO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2020.103836

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import EO    >>> import numpy as np
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
    >>> model = EO.AdaptiveEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wunnava, A., Naik, M.K., Panda, R., Jena, B. and Abraham, A., 2020. A novel interdependence based
    multilevel thresholding technique using adaptive equilibrium optimizer. Engineering Applications of
    Artificial Intelligence, 94, p.103836.
    """

    cdef public object pop_len

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
        self.pop_len = int(self.pop_size / 3)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        c_pool = self.make_equilibrium_pool__(pop.take(self.sorted_order(pop)[:4]))
        t = (<object>(1 - epoch_c / self.epoch)) ** (<object>(self.a2 * epoch_c / self.epoch))
        lamda = rng.uniform(0, 1, (n, d))
        r = rng.uniform(0, 1, (n, d))
        c_eq = c_pool.X[rng.integers(0, c_pool.n, size=n)]  # random candidate from the pool
        f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 14
        gcp = 0.5 * rng.uniform(size=(n, 1)) * (rng.uniform(size=(n, 1)) >= self.GP)
        g = gcp * (c_eq - lamda * X) * f
        pos = c_eq + (X - c_eq) * f + (g * self.V / lamda) * (1.0 - f)  # Eq. 9
        fit_average = np.mean(np.ascontiguousarray(pop.F))  # Eq. 19
        pos = np.where(ops.better(self, np.asarray(pop.F), fit_average)[:, None], pos * (0.5 + rng.uniform(0, 1, (n, d))), pos)
        ops.step(self, pos)

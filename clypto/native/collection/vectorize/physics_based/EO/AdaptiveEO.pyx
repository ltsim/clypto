#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.physics_based.EO.OriginalEO cimport OriginalEO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


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

    cdef void evolve(self, int epoch):
        # The mean fitness is recomputed for every agent from the rows updated so far, so the
        # loop is sequential on the buffer rows (batched in swarm modes).
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        # ---------------- Memory saving-------------------  make equilibrium pool
        c_pool = self.make_equilibrium_pool__(pop.take(self.sorted_order(pop)[:4]))
        # Eq. 9
        t = (<object>(1 - epoch / self.epoch)) ** (<object>(self.a2 * epoch / self.epoch))
        ## Memory saving, Eq 20, 21
        Xp = pop.X
        for idx in range(0, self.pop_size):
            lamda = self.generator.uniform(0, 1, d)
            r = self.generator.uniform(0, 1, d)
            c_eq = c_pool.X[self.generator.integers(0, c_pool.n)]  # random selection 1 of candidate from the pool
            f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 14
            r1 = self.generator.uniform()
            r2 = self.generator.uniform()
            gcp = 0.5 * r1 * np.ones(d) * (r2 >= self.GP)
            g0 = gcp * (c_eq - lamda * Xp[idx])
            g = g0 * f
            fit_average = np.mean(np.ascontiguousarray(pop.F))  # Eq. 19
            pos_new = (
                    c_eq
                    + (Xp[idx] - c_eq) * f
                    + (g * self.V / lamda) * (1.0 - f)
            )  # Eq. 9
            if self.compare_fitness(pop.F[idx], fit_average, self.problem.minmax):
                pos_new = np.multiply(pos_new, (0.5 + self.generator.uniform(0, 1, d)))
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

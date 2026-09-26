#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.physics_based.EO.OriginalEO cimport OriginalEO
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class ModifiedEO(OriginalEO):
    """
    The original version of: Modified Equilibrium Optimizer (MEO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2020.106542

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.physics_based import EO    >>> import numpy as np
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
    >>> model = EO.ModifiedEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Gupta, S., Deep, K. and Mirjalili, S., 2020. An efficient equilibrium optimizer with mutation
    strategy for numerical optimization. Applied Soft Computing, 96, p.106542.
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

    def _evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef NativePopulation pop_s1, pop_s2, pop_s2_new, pop_s3
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        # ---------------- Memory saving-------------------  make equilibrium pool
        c_pool = self.make_equilibrium_pool__(pop.take(self.sorted_order(pop)[:4]))
        cand.X[:] = self._correct_solution(self.candidates__(pop, c_pool, epoch))
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)
        ## Sort the updated population based on fitness
        pop_s1 = pop.take(self.sorted_order(pop)[:self.pop_len])
        ## Mutation scheme
        pop_s2 = pop_s1.take(np.arange(pop_s1.n))
        pop_s2_new = pop_s1.empty_like()
        pop_s2_new.X[:] = self._correct_solution(pop_s2.X * (1 + self.generator.normal(0, 1, (self.pop_len, d))))  # Eq. 12
        self.evaluate(pop_s2_new, 0, pop_s2_new.n)
        if self.mode in self.AVAILABLE_MODES:
            # greedy_selection_population(pop_s2_new, pop_s2): the mutated agent stays unless the original is strictly better
            keep = pop_s2.F < pop_s2_new.F if self.problem.sense == "min" else pop_s2.F > pop_s2_new.F
            rows = np.flatnonzero(~keep)
            pop_s2.buf[rows] = pop_s2_new.buf[rows]
        else:
            ops.accept(self, pop_s2_new, dst=pop_s2)
        ## Search Mechanism
        pos_s1_mean = np.mean(np.ascontiguousarray(pop_s1.X), axis=0)
        R = self.generator.random((self.pop_len, 2))
        pop_s3 = pop_s1.empty_like()
        pop_s3.X[:] = self._correct_solution(
            (c_pool.X[0] - pos_s1_mean) - R[:, 0:1] * (self.problem.bounds.low + R[:, 1:2] * (self.problem.bounds.up - self.problem.bounds.low))
        )
        self.evaluate(pop_s3, 0, pop_s3.n)
        ## Construct a new population
        merged = pop_s1.concat(pop_s2).concat(pop_s3)
        n_left = self.pop_size - merged.n
        idx_selected = self.generator.choice(range(0, c_pool.n), n_left, replace=False)
        self.pop = merged.concat(c_pool.take(idx_selected[:n_left]))

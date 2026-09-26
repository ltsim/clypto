#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np

from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation, select_better


cdef class _PSOBase(LegacyNativeOptimizer):
    """Shared PSO state: the agent layout, velocity limits and greedy updates."""

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m):
        return [("V", d), ("P", d), ("PO", m), ("PF", 1)]

    cdef void initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max

    cdef void init_fields(self, NativePopulation pop):
        # generate_agent: velocity ~ U(-v_max, v_max); personal best = itself.
        self.cV, self.cP = pop.offset("V"), pop.offset("P")
        self.cPO, self.cPF = pop.offset("PO"), pop.offset("PF")
        pop.field("V")[:] = self.generator.uniform(-self.v_max, self.v_max, (pop.n, pop.d))
        pop.field("P")[:] = pop.X
        pop.field("PO")[:] = pop.O
        pop.field("PF")[:, 0] = pop.F

    cdef list chunks(self, Py_ssize_t n):
        """Row ranges that share one ``g_best`` position (see module docstring)."""
        cdef Py_ssize_t b = self._g_best_row
        if 0 <= b < n - 1:
            return [(0, b + 1), (b + 1, n)]
        return [(0, n)]

    cdef void accept(self, NativePopulation cand, Py_ssize_t start, Py_ssize_t stop):
        """Classic greedy updates: the agent, then its personal best, if improved."""
        cdef NativePopulation pop = self.pop
        cdef bint maximize = self.problem.minmax != "min"
        select_better(pop, pop.cX, pop.cO, pop.cF, cand, cand.cX, cand.cO, cand.cF,
                      start, stop, maximize)
        select_better(pop, self.cP, self.cPO, self.cPF, cand, cand.cX, cand.cO, cand.cF,
                      start, stop, maximize)

    cdef object amend_random(self, object pos, object U):
        """Classic PSO amend: out-of-bounds values become U(lb, ub) draws."""
        lb, ub = self.problem.lb, self.problem.ub
        return np.where(np.logical_and(lb <= pos, pos <= ub), pos, lb + (ub - lb) * U)

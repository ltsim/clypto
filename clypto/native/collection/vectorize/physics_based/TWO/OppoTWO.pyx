#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.collection.physics_based.TWO.OriginalTWO cimport OriginalTWO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OppoTWO(OriginalTWO):
    """
    The opossition-based learning version: Tug of War Optimization (OTWO)

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import TWO    >>> import numpy as np
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
    >>> model = TWO.OppoTWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
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

    cdef void initialization(self):
        cdef NativePopulation pop
        LegacyNativeOptimizer.initialization(self)
        pop = self.pop
        half_size = -(-self.pop_size // 2)  # ceil division, safe for odd pop_size
        list_idx = self.generator.choice(range(0, self.pop_size), half_size, replace=False)
        pop_temp = pop.take(list_idx[:half_size])
        pop_oppo = self.new_population(self.correct_solution(self.problem.ub + self.problem.lb - pop_temp.X))
        merged = pop_temp.concat(pop_oppo)
        self.pop = merged.take(np.arange(min(self.pop_size, merged.n)))
        self.update_weight__(self.pop)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef NativeTarget tar
        cdef Py_ssize_t idx
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        lb, ub = self.problem.lb, self.problem.ub
        ## Apply force of others solution on each individual solution
        self.forces__(pop, epoch)
        ## Amend solution and update fitness value
        Xp = pop.X
        for idx in range(self.pop_size):
            g_best = self.g_best_x()  # aliased to the best row: sees the updates made so far
            pos_new = g_best + self.generator.normal(0, 1, self.problem.n_dims) / (epoch) * (g_best - Xp[idx])
            conditions = np.logical_or(Xp[idx] < lb, Xp[idx] > ub)
            conditions = np.logical_and(conditions, self.generator.random(self.problem.n_dims) < 0.5)
            pos_new = np.where(conditions, pos_new, Xp[idx])
            pos_c = self.correct_solution(pos_new)
            if swarm:
                Xp[idx] = pos_c
            else:
                # the classic sequential path evaluates the *uncorrected* position
                ops.set_row(pop, idx, pos_c, self.get_target(pos_new))
        if swarm:
            self.evaluate(pop, 0, pop.n)
        ## Opposition-based here
        cand = pop.empty_like()
        g_best = np.array(self.g_best_x())
        for idx in range(self.pop_size):
            C_op = self.correct_solution(lb + ub - g_best + self.generator.uniform() * (g_best - Xp[idx]))
            ops.commit(self, pop, cand, idx, self.correct_solution(C_op), swarm)
        if swarm:
            ops.finish(self, cand, 0, pop.n)
        self.update_weight__(pop)

#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.native.collection.vectorize.physics_based.TWO.OriginalTWO cimport OriginalTWO
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
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        lb, ub = self.problem.lb, self.problem.ub
        pos = self.forces__(pop, epoch_c)
        g = np.array(self.g_best_x())
        around = g + rng.normal(0, 1, (n, d)) / epoch_c * (g - pos)
        redraw = ((pos < lb) | (pos > ub)) & (rng.random((n, d)) < 0.5)
        pop.X[:] = self.correct_solution(np.where(redraw, around, pos))
        self.evaluate(pop, 0, n)
        # opposition-based candidates around the best
        g = np.array(self.g_best_x())
        X = pop.X
        ops.step(self, self.correct_solution(lb + ub - g + rng.uniform(size=(n, 1)) * (g - X)))
        self.update_weight__(pop)

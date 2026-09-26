#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.native.collection.vectorize.physics_based.TWO.OriginalTWO cimport OriginalTWO
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class LevyTWO(OriginalTWO):
    """
    The Levy-flight version of: Tug of War Optimization (LevyTWO)

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
    >>> model = TWO.LevyTWO(epoch=1000, pop_size=50)
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

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation sub
        pos = self.forces__(pop, epoch_c)
        pop.X[:] = self.correct_solution(self.bound__(pos, epoch_c))
        self.evaluate(pop, 0, pop.n)
        # half of the teams try a Levy jump
        jump = np.flatnonzero(self.generator.random(pop.n) < 0.5)
        if len(jump):
            sub = pop.take(jump)
            sub.X[:] = self.correct_solution(pop.X[jump] + self.get_levy_flight_step(beta=1.0, multiplier=0.01, size=(len(jump), pop.d), case=-1))
            self.evaluate(sub, 0, len(jump))
            ops.scatter(self, sub, jump)
        self.update_weight__(pop)

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
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativeTarget tar
        cdef Py_ssize_t idx
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        self.forces__(pop, epoch)
        for idx in range(self.pop_size):
            pos_new = self.bound__(pop, idx, epoch)
            pos_c = self.correct_solution(pos_new)
            if swarm:
                pop.X[idx] = pos_c
            else:
                # the classic sequential path evaluates the *uncorrected* position
                ops.set_row(pop, idx, pos_c, self.get_target(pos_new))
        if swarm:
            self.evaluate(pop, 0, pop.n)
        ### Apply levy-flight here
        for idx in range(self.pop_size):
            ## Chance for each agent to update using levy is 50%
            if self.generator.random() < 0.5:
                levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.01, size=self.problem.n_dims, case=-1)
                pos_new = self.correct_solution(pop.X[idx] + levy_step)
                tar = self.get_target(pos_new)
                if self.compare_fitness(tar.fitness, pop.F[idx], self.problem.minmax):
                    ops.set_row(pop, idx, pos_new, tar)
        self.update_weight__(pop)

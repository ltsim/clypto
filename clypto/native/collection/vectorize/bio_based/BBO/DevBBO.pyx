#!/usr/bin/env python
# Created by "Thieu" at 12:24, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.bio_based.BBO.OriginalBBO cimport OriginalBBO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class DevBBO(OriginalBBO):
    """
    The developed version: Biogeography-Based Optimization (BBO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_m (float): (0, 1) -> better [0.01, 0.2], Mutation probability
        + n_elites (int): (2, pop_size/2) -> better [2, 5], Number of elites will be keep for next generation

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import BBO    >>> import numpy as np
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
    >>> model = BBO.DevBBO(epoch=1000, pop_size=50, p_m=0.01, n_elites=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        p_m: float = 0.01,
        n_elites: int = 2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 100
            p_m: Mutation probability, default=0.01
            n_elites: Number of elites will be keep for next generation, default=2
        """
        super().__init__(epoch, pop_size, p_m, n_elites, name=name, mode=mode)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp, Xc = pop.X, cand.X
        pop_elites = pop.take(self.sorted_order(pop)[:self.n_elites])
        list_fitness = list(pop.F)
        pop_new = []
        for idx in range(0, self.pop_size):
            # Probabilistic migration to the i-th position
            # Pick a position from which to emigrate (roulette wheel selection)
            idx_selected = self.get_index_roulette_wheel_selection(list_fitness)
            # this is the migration step
            condition = self.generator.random(self.problem.n_dims) < self.mr[idx]
            pos_new = np.where(
                condition, Xp[idx_selected], Xp[idx]
            )
            # Mutation
            mutated = self.generator.uniform(self.problem.lb, self.problem.ub)
            pos_new = np.where(
                self.generator.random(self.problem.n_dims) < self.p_m, mutated, pos_new
            )
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm, True)
        if swarm:
            ops.finish(self, cand, 0, n)
        # replace the solutions with their new migrated and mutated versions then Merge Populations
        merged = pop.concat(pop_elites)
        self.pop = merged.take(self.sorted_order(merged)[:self.pop_size])

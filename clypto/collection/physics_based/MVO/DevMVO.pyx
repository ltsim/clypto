#!/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class DevMVO(LegacyNativeOptimizer):
    """
    The developed version: Multi-Verse Optimizer (MVO)

    Notes:
        + New routtele wheel selection can handle negative values
        + Removed condition when self.generator.normalize fitness. So the chance to choose while whole higher --> better

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wep_min (float): [0.05, 0.3], Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
        + wep_max (float: [0.75, 1.0], Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import MVO    >>> import numpy as np
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
    >>> model = MVO.DevMVO(epoch=1000, pop_size=50, wep_min = 0.2, wep_max = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        wep_min: float = 0.2,
        wep_max: float = 1.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wep_min (float): Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
            wep_max (float: Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "wep_min", "wep_max"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.wep_min = cy.validator(float, wep_min, (0, 0.5), "wep_min")
        self.wep_max = cy.validator(float, wep_max, [0.5, 3.0], "wep_max")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        # Eq. (3.3) in the paper
        wep = self.wep_max - epoch * ((self.wep_max - self.wep_min) / self.epoch)
        # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
        tdr = 1 - epoch ** (1.0 / 6) / (<object>self.epoch) ** (<object>(1.0 / 6))
        pop_new = []
        for idx in range(0, self.pop_size):
            if self.generator.uniform() < wep:
                list_fitness = np.array(pop.F)
                white_hole_id = self.get_index_roulette_wheel_selection(list_fitness)
                black_hole_pos_1 = Xp[idx] + tdr * self.generator.normal(
                    0, 1
                ) * (Xp[white_hole_id] - Xp[idx])
                black_hole_pos_2 = g_best + tdr * self.generator.normal(
                    0, 1
                ) * (g_best - Xp[idx])
                black_hole_pos = np.where(
                    self.generator.random(self.problem.n_dims) < 0.5,
                    black_hole_pos_1,
                    black_hole_pos_2,
                )
            else:
                black_hole_pos = self.problem.generate_solution()
            ops.commit(self, pop, cand, idx, self.correct_solution(black_hole_pos), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

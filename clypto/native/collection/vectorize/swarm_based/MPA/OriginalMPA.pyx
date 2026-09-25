#!/usr/bin/env python
# Created by "Thieu" at 17:28, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalMPA(LegacyNativeOptimizer):
    """
    The developed version: Marine Predators Algorithm (MPA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0957417420302025
        2. https://www.mathworks.com/matlabcentral/fileexchange/74578-marine-predators-algorithm-mpa

    Notes:
        1. To use the original paper, set the training mode = "swarm"
        2. They update the whole population at the same time before update the fitness
        3. Two variables that they consider it as constants which are FADS = 0.2 and P = 0.5

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import MPA    >>> import numpy as np
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
    >>> model = MPA.OriginalMPA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
    Marine Predators Algorithm: A nature-inspired metaheuristic. Expert systems with applications, 152, 113377.
    """

    cdef public object FADS
    cdef public object P

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
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void initialize_variables(self):
        self.FADS = 0.2
        self.P = 0.5

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        CF = (1 - epoch / self.epoch) ** (2 * epoch / self.epoch)
        RL = self.get_levy_flight_step(
            beta=1.5,
            multiplier=0.05,
            size=(self.pop_size, self.problem.n_dims),
            case=-1,
        )
        RB = self.generator.standard_normal((self.pop_size, self.problem.n_dims))
        per1 = self.generator.permutation(self.pop_size)
        per2 = self.generator.permutation(self.pop_size)
        for idx in range(0, self.pop_size):
            R = self.generator.random(self.problem.n_dims)
            if epoch < self.epoch / 3:  # Phase 1 (Eq.12)
                step_size = RB[idx] * (g_best - RB[idx] * Xp[idx])
                pos_new = Xp[idx] + self.P * R * step_size
            elif self.epoch / 3 < epoch < 2 * self.epoch / 3:  # Phase 2 (Eqs. 13 & 14)
                if idx > self.pop_size / 2:
                    step_size = RB[idx] * (RB[idx] * g_best - Xp[idx])
                    pos_new = g_best + self.P * CF * step_size
                else:
                    step_size = RL[idx] * (g_best - RL[idx] * Xp[idx])
                    pos_new = Xp[idx] + self.P * R * step_size
            else:  # Phase 3 (Eq. 15)
                step_size = RL[idx] * (RL[idx] * g_best - Xp[idx])
                pos_new = g_best + self.P * CF * step_size
            pos_new = self.correct_solution(pos_new)
            if self.generator.random() < self.FADS:
                u = np.where(self.generator.random(self.problem.n_dims) < self.FADS, 1, 0)
                pos_new = (
                        pos_new
                        + CF
                        * (
                                self.problem.lb
                                + self.generator.random(self.problem.n_dims)
                                * (self.problem.ub - self.problem.lb)
                        )
                        * u
                )
            else:
                r = self.generator.random()
                step_size = (self.FADS * (1 - r) + r) * (Xp[per1[idx]] - Xp[per2[idx]])
                pos_new = pos_new + step_size
            pos_new = self.correct_solution(pos_new)
            ops.commit(self, pop, cand, idx, pos_new, swarm, True)
        if swarm:
            ops.finish(self, cand, 0, pop.n)

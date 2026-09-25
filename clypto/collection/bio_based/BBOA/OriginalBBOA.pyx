#!/usr/bin/env python
# Created by "Thieu" at 12:51, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalBBOA(LegacyNativeOptimizer):
    """
    The original version of: Brown-Bear Optimization Algorithm (BBOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/125490-brown-bear-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import BBOA    >>> import numpy as np
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
    >>> model = BBOA.OriginalBBOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Prakash, T., Singh, P. P., Singh, V. P., & Singh, S. N. (2023). A Novel Brown-bear Optimization
    Algorithm for Solving Economic Dispatch Problem. In Advanced Control & Optimization Paradigms for
    Energy System Operation and Management (pp. 137-164). River Publishers.
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

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        gw_pos, gw_fit = self.g_worst.solution, self.g_worst.target.fitness
        pp = epoch / self.epoch

        ## Pedal marking behaviour
        pop_new = []
        for idx in range(0, self.pop_size):
            if pp <= 1 / 3:  # Gait while walking
                pos_new = Xp[idx] + (
                        -pp
                        * self.generator.random(self.problem.n_dims)
                        * Xp[idx]
                )
            elif 1 / 3 < pp <= 2 / 3:  # Careful Stepping
                qq = pp * self.generator.random(self.problem.n_dims)
                pos_new = Xp[idx] + (
                        qq
                        * (
                                g_best
                                - self.generator.integers(1, 3) * gw_pos
                        )
                )
            else:
                ww = 2 * pp * np.pi * self.generator.random(self.problem.n_dims)
                pos_new = (
                        Xp[idx]
                        + (ww * g_best - np.abs(Xp[idx]))
                        - (ww * gw_pos - np.abs(Xp[idx]))
                )
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

        ## Sniffing of pedal marks
        pop_new = []
        for idx in range(0, self.pop_size):
            kk = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            if self.compare_fitness(pop.F[idx], pop.F[kk], self.problem.minmax):
                pos_new = Xp[idx] + self.generator.random() * (
                        Xp[idx] - Xp[kk]
                )
            else:
                pos_new = Xp[idx] + self.generator.random() * (
                        Xp[kk] - Xp[idx]
                )
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

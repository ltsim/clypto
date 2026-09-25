#!/usr/bin/env python
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalHHO(LegacyNativeOptimizer):
    """
    The original version of: Harris Hawks Optimization (HHO)

    Links:
        1. https://doi.org/10.1016/j.future.2019.02.028

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import HHO    >>> import numpy as np
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
    >>> model = HHO.OriginalHHO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Heidari, A.A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M. and Chen, H., 2019.
    Harris hawks optimization: Algorithm and applications. Future generation computer systems, 97, pp.849-872.
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
        cdef NativeTarget tar_y, tar_z
        cdef Py_ssize_t idx, n = pop.n
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        minmax = self.problem.minmax
        X_m = None
        for idx in range(0, self.pop_size):
            # -1 < E0 < 1
            E0 = 2 * self.generator.uniform() - 1
            # factor to show the decreasing energy of rabbit
            E = 2 * E0 * (1.0 - epoch * 1.0 / self.epoch)
            J = 2 * (1 - self.generator.uniform())

            # -------- Exploration phase Eq. (1) in paper -------------------
            if np.abs(E) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                if self.generator.random() >= 0.5:  # perch based on other family members
                    X_rand = np.array(Xp[self.generator.integers(0, self.pop_size)])
                    pos_new = X_rand - self.generator.uniform() * np.abs(
                        X_rand - 2 * self.generator.uniform() * Xp[idx]
                    )
                else:  # perch on a random tall tree (random site inside group's home range)
                    if X_m is None:
                        X_m = np.mean(np.array(Xp))
                    pos_new = (g_best - X_m) - self.generator.uniform() * (
                        self.problem.lb + self.generator.uniform() * (self.problem.ub - self.problem.lb)
                    )
                Xc[idx] = self.correct_solution(pos_new)
            # -------- Exploitation phase -------------------
            else:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                # phase 1: ----- surprise pounce (seven kills) ----------
                if self.generator.random() >= 0.5:
                    delta_X = g_best - Xp[idx]
                    if np.abs(E) >= 0.5:  # Hard besiege Eq. (6) in paper
                        pos_new = delta_X - E * np.abs(J * g_best - Xp[idx])
                    else:  # Soft besiege Eq. (4) in paper
                        pos_new = g_best - E * np.abs(delta_X)
                    Xc[idx] = self.correct_solution(pos_new)
                else:
                    LF_D = self.get_levy_flight_step(beta=1.5, multiplier=0.01, case=-1)
                    if np.abs(E) >= 0.5:  # Soft besiege Eq. (10) in paper
                        Y = g_best - E * np.abs(J * g_best - Xp[idx])
                    else:  # Hard besiege Eq. (11) in paper
                        if X_m is None:
                            X_m = np.mean(np.array(Xp))
                        Y = g_best - E * np.abs(J * g_best - X_m)
                    pos_Y = self.correct_solution(Y)
                    tar_y = self.get_target(pos_Y)
                    Z = Y + self.generator.uniform(self.problem.lb, self.problem.ub) * LF_D
                    pos_Z = self.correct_solution(Z)
                    tar_z = self.get_target(pos_Z)
                    if self.compare_fitness(tar_y.fitness, pop.F[idx], minmax):
                        Xc[idx] = pos_Y
                    elif self.compare_fitness(tar_z.fitness, pop.F[idx], minmax):
                        Xc[idx] = pos_Z
                    else:
                        Xc[idx] = Xp[idx]
        # (the classic code re-evaluates every candidate here, including the Y/Z ones)
        self.evaluate(cand, 0, n)
        ops.greedy(self, cand)

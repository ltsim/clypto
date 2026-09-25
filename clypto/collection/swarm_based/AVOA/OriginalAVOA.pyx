#!/usr/bin/env python
# Created by "Thieu" at 21:45, 26/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalAVOA(LegacyNativeOptimizer):
    """
    The original version of: African Vultures Optimization Algorithm (AVOA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0360835221003120
        2. https://www.mathworks.com/matlabcentral/fileexchange/94820-african-vultures-optimization-algorithm

    Notes (parameters):
        + p1 (float): probability of status transition, default 0.6
        + p2 (float): probability of status transition, default 0.4
        + p3 (float): probability of status transition, default 0.6
        + alpha (float): probability of 1st best, default = 0.8
        + gama (float): a factor in the paper (not much affect to algorithm), default = 2.5

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import AVOA    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = AVOA.OriginalAVOA(epoch=1000, pop_size=50, p1=0.6, p2=0.4, p3=0.6, alpha=0.8, gama=2.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Gharehchopogh, F. S., & Mirjalili, S. (2021). African vultures optimization algorithm: A new
    nature-inspired metaheuristic algorithm for global optimization problems. Computers & Industrial Engineering, 158, 107408.
    """

    cdef public object p1
    cdef public object p2
    cdef public object p3
    cdef public object alpha
    cdef public object gama

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        p1: float = 0.6,
        p2: float = 0.4,
        p3: float = 0.6,
        alpha: float = 0.8,
        gama: float = 2.5,
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
            parameters=["epoch", "pop_size", "p1", "p2", "p3", "alpha", "gama"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.p1 = cy.validator(float, p1, (0, 1), "p1")
        self.p2 = cy.validator(float, p2, (0, 1), "p2")
        self.p3 = cy.validator(float, p3, (0, 1), "p3")
        self.alpha = cy.validator(float, alpha, (0, 1), "alpha")
        self.gama = cy.validator(float, gama, (0, 5.0), "gama")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n
        Xp, Xc = pop.X, cand.X
        a = self.generator.uniform(-2, 2) * (
                (np.sin((np.pi / 2) * (epoch / self.epoch)) ** self.gama)
                + np.cos((np.pi / 2) * (epoch / self.epoch))
                - 1
        )
        ppp = (2 * self.generator.random() + 1) * (1 - epoch / self.epoch) + a
        order = self.sorted_order(pop)
        best_list = [np.array(Xp[order[0]]), np.array(Xp[order[1]])]
        for idx in range(0, self.pop_size):
            F = ppp * (2 * self.generator.random() - 1)
            rand_idx = self.generator.choice([0, 1], p=[self.alpha, 1 - self.alpha])
            rand_pos = best_list[rand_idx]
            if np.abs(F) >= 1:  # Exploration
                if self.generator.random() < self.p1:
                    pos_new = (
                            rand_pos
                            - (
                                np.abs(
                                    (2 * self.generator.random()) * rand_pos
                                    - Xp[idx]
                                )
                            )
                            * F
                    )
                else:
                    pos_new = (
                            rand_pos
                            - F
                            + self.generator.random()
                            * (
                                    (self.problem.ub - self.problem.lb)
                                    * self.generator.random()
                                    + self.problem.lb
                            )
                    )
            else:  # Exploitation
                if np.abs(F) < 0.5:  # Phase 1
                    best_x1 = best_list[0]
                    best_x2 = best_list[1]
                    if self.generator.random() < self.p2:
                        A = (
                                best_x1
                                - (
                                        (best_x1 * Xp[idx])
                                        / (best_x1 - Xp[idx] ** 2 + self.EPSILON)
                                )
                                * F
                        )
                        B = (
                                best_x2
                                - (
                                        (best_x2 * Xp[idx])
                                        / (best_x2 - Xp[idx] ** 2 + self.EPSILON)
                                )
                                * F
                        )
                        pos_new = (A + B) / 2
                    else:
                        pos_new = rand_pos - np.abs(
                            rand_pos - Xp[idx]
                        ) * F * self.get_levy_flight_step(
                            beta=1.5, multiplier=1.0, size=self.problem.n_dims, case=-1
                        )
                else:  # Phase 2
                    if self.generator.random() < self.p3:
                        pos_new = (
                                      np.abs(
                                          (2 * self.generator.random()) * rand_pos
                                          - Xp[idx]
                                      )
                                  ) * (F + self.generator.random()) - (
                                          rand_pos - Xp[idx]
                                  )
                    else:
                        s1 = (
                                rand_pos
                                * (
                                        self.generator.random()
                                        * Xp[idx]
                                        / (2 * np.pi)
                                )
                                * np.cos(Xp[idx])
                        )
                        s2 = (
                                rand_pos
                                * (
                                        self.generator.random()
                                        * Xp[idx]
                                        / (2 * np.pi)
                                )
                                * np.sin(Xp[idx])
                        )
                        pos_new = rand_pos - (s1 + s2)
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        self.pop = cand

#!/usr/bin/env python
# Created by "Thieu" at 15:53, 07/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalAO(LegacyNativeOptimizer):
    """
    The original version of: Aquila Optimization (AO)

    Links:
        1. https://doi.org/10.1016/j.cie.2021.107250

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import AO    >>> import numpy as np
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
    >>> model = AO.OriginalAO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A.A., Al-Qaness, M.A. and Gandomi, A.H., 2021.
    Aquila optimizer: a novel meta-heuristic optimization algorithm. Computers & Industrial Engineering, 157, p.107250.
    """

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
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
        cdef Py_ssize_t i, idx
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        alpha = delta = 0.1
        g1 = 2 * self.generator.random() - 1  # Eq. 16
        g2 = 2 * (1 - epoch / self.epoch)  # Eq. 17
        dim_list = np.array(list(range(1, self.problem.n_dims + 1)))
        miu = 0.00565
        r0 = 10
        r = r0 + miu * dim_list
        w = 0.005
        phi0 = 3 * np.pi / 2
        phi = -w * dim_list + phi0
        x = r * np.sin(phi)  # Eq.(9)
        y = r * np.cos(phi)  # Eq.(10)
        QF = epoch ** (
                (2 * self.generator.random() - 1) / (1 - self.epoch) ** 2
        )  # Eq.(15)        Quality function
        for i in range(0, self.pop_size):
            idx = i
            x_mean = np.mean(Xp, axis=0)
            levy_step = self.get_levy_flight_step(beta=1.5, multiplier=1.0, case=-1)
            if epoch <= (2 / 3) * self.epoch:  # Eq. 3, 4
                if self.generator.random() < 0.5:
                    pos_new = g_best * (1 - epoch / self.epoch) + self.generator.random() * (x_mean - g_best)
                else:
                    # (the classic loop variable is reassigned here: the replacement below targets this agent)
                    idx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                    pos_new = g_best * levy_step + Xp[idx] + self.generator.random() * (y - x)  # Eq. 5
            else:
                if self.generator.random() < 0.5:
                    pos_new = (
                            alpha * (g_best - x_mean)
                            - self.generator.random()
                            * (
                                    self.generator.random()
                                    * (self.problem.ub - self.problem.lb)
                                    + self.problem.lb
                            )
                            * delta
                    )  # Eq. 13
                else:
                    pos_new = (
                            QF * g_best
                            - (g2 * Xp[idx] * self.generator.random())
                            - g2 * levy_step
                            + self.generator.random() * g1
                    )  # Eq. 14
            pos_new = self.correct_solution(pos_new)
            ops.commit(self, pop, cand, i if swarm else idx, pos_new, swarm)
        if swarm:
            ops.finish(self, cand, 0, pop.n)

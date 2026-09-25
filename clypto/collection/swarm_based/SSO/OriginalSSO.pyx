#!/usr/bin/env python
# Created by "Thieu" at 11:38, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalSSO(LegacyNativeOptimizer):
    """
    The original version of: Salp Swarm Optimization (SSO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2017.07.002

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import SSO    >>> import numpy as np
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
    >>> model = SSO.OriginalSSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., Gandomi, A.H., Mirjalili, S.Z., Saremi, S., Faris, H. and Mirjalili, S.M., 2017.
    Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems. Advances in Engineering Software, 114, pp.163-191.
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
            sort_flag=True,
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
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        ## Eq. (3.2) in the paper
        c1 = 2 * np.exp(-((4 * epoch / self.epoch) ** 2))
        half = [idx for idx in range(n) if idx < self.pop_size / 2]
        h = len(half)
        # first half: c2 and c3 per agent, drawn agent by agent
        R = self.generator.random((h, 2, d))
        for idx in range(h):
            c2_list, c3_list = R[idx, 0], R[idx, 1]
            pos_new_1 = g_best + c1 * ((ub - lb) * c2_list + lb)
            pos_new_2 = g_best - c1 * ((ub - lb) * c2_list + lb)
            Xc[idx] = self.correct_solution(np.where(c3_list < 0.5, pos_new_1, pos_new_2))
        if swarm:
            for idx in range(h, n):
                Xc[idx] = self.correct_solution((Xp[idx] + Xp[idx - 1]) / 2)  # Eq. (3.4) in the paper
            ops.finish(self, cand, 0, n)
        else:
            self.evaluate(cand, 0, h)
            ops.accept(self, cand, 0, h)
            for idx in range(h, n):
                # Eq. (3.4): reads the agent before it, already replaced in this epoch
                pos_new = self.correct_solution((Xp[idx] + Xp[idx - 1]) / 2)
                ops.commit(self, pop, cand, idx, pos_new, False)

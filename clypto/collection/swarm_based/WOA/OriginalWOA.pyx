#!/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalWOA(LegacyNativeOptimizer):
    """
    The original version of: Whale Optimization Algorithm (WOA)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2016.01.008
        2. https://mathworks.com/matlabcentral/fileexchange/55667-the-whale-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import WOA    >>> import numpy as np
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
    >>> model = WOA.OriginalWOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm. Advances in engineering software, 95, pp.51-67.
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
        # The classic code only refreshes the *targets* of the population with the candidates'
        # fitness (sequential mode) and never moves it; swarm modes replace the population.
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef NativeTarget tar
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        cols = np.arange(d)
        a = 2 - 2 * epoch / self.epoch  # linearly decreased from 2 to 0
        a2 = -1 + epoch * ((-1) / self.epoch)
        for idx in range(0, self.pop_size):
            r1, r2 = self.generator.random(size=2)
            A = a * (2 * r1 - a)
            C = 2 * r2
            b = 1
            l = (a2 - 1) * self.generator.random() + 1
            p = self.generator.random()
            if p < 0.5:
                if np.abs(A) >= 1:
                    # one random whale per dimension
                    ids = np.array([self.generator.choice(list(set(range(0, self.pop_size)) - {idx})) for _ in range(d)])
                    x_rand = Xp[ids, cols]
                    pos_new = x_rand - A * np.abs(C * x_rand - Xp[idx])
                else:
                    pos_new = g_best - A * np.abs(C * g_best - Xp[idx])
            else:
                D1 = np.abs(g_best - Xp[idx])
                pos_new = D1 * np.exp(b * l) * np.cos(l * 2 * np.pi) + g_best
            Xc[idx] = self.correct_solution(pos_new)
            if not swarm:
                tar = self.get_target(Xc[idx])
                pop.O[idx] = tar.objectives
                pop.F[idx] = tar.fitness
        if swarm:
            self.evaluate(cand, 0, n)
            self.pop = cand

#!/usr/bin/env python
# Created by "Thieu" at 17:55, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalHBA(LegacyNativeOptimizer):
    """
    The original version of: Honey Badger Algorithm (HBA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0378475421002901
        2. https://www.mathworks.com/matlabcentral/fileexchange/98204-honey-badger-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import HBA    >>> import numpy as np
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
    >>> model = HBA.OriginalHBA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F. A., Houssein, E. H., Hussain, K., Mabrouk, M. S., & Al-Atabany, W. (2022). Honey Badger Algorithm: New metaheuristic
    algorithm for solving optimization problems. Mathematics and Computers in Simulation, 192, 84-110.
    """

    cdef public object beta
    cdef public object C

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
        self.beta = 6  # the ability of HB to get the food  Eq.(4)
        self.C = 2  # constant in Eq. (3)

    def get_intensity__(self, best, NativePopulation pop):
        size = pop.n
        Xp = pop.X
        di = np.zeros(size)
        si = np.zeros(size)
        for idx in range(0, size):
            di[idx] = (np.linalg.norm(Xp[idx] - best) + self.EPSILON) ** 2
            if idx == size - 1:
                si[idx] = (np.linalg.norm(Xp[idx] - Xp[0]) + self.EPSILON) ** 2
            else:
                si[idx] = (np.linalg.norm(Xp[idx] - Xp[idx + 1]) + self.EPSILON) ** 2
        r2 = self.generator.random(size)
        return r2 * si / (4 * np.pi * di)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        tt = self.epoch
        alpha = self.C * np.exp(-tt / self.epoch)  # density factor in Eq. (3)
        I = self.get_intensity__(g_best, pop)  # intensity in Eq. (2)
        for idx in range(0, self.pop_size):
            r = self.generator.random()
            F = self.generator.choice([1, -1])
            di = g_best - Xp[idx]
            r3 = self.generator.random(d)
            r4 = self.generator.random(d)
            r5 = self.generator.random(d)
            r6 = self.generator.random(d)
            r7 = self.generator.random(d)
            temp1 = (
                    g_best
                    + F * self.beta * I[idx] * g_best
                    + F
                    * r3
                    * alpha
                    * di
                    * np.abs(np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5)))
            )
            temp2 = g_best + F * r7 * alpha * di
            pos_new = np.where(r6 < 0.5, temp1, temp2)
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

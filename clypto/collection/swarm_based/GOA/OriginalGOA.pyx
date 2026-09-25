#!/usr/bin/env python
# Created by "Thieu" at 14:53, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalGOA(LegacyNativeOptimizer):
    """
    The original version of: Grasshopper Optimization Algorithm (GOA)

    Links:
        1. https://dx.doi.org/10.1016/j.advengsoft.2017.01.004
        2. https://www.mathworks.com/matlabcentral/fileexchange/61421-grasshopper-optimisation-algorithm-goa

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c_min (float): coefficient c min, default = 0.00004
        + c_max (float): coefficient c max, default = 2.0

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GOA    >>> import numpy as np
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
    >>> model = GOA.OriginalGOA(epoch=1000, pop_size=50, c_min = 0.00004, c_max = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Saremi, S., Mirjalili, S. and Lewis, A., 2017. Grasshopper optimisation algorithm:
    theory and application. Advances in Engineering Software, 105, pp.30-47.
    """

    cdef public object c_min
    cdef public object c_max

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c_min: float = 0.00004,
        c_max: float = 2.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c_min (float): coefficient c min, default=0.00004
            c_max (float): coefficient c max, default=2.0
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "c_min", "c_max"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.c_min = cy.validator(float, c_min, [0.00001, 0.2], "c_min")
        self.c_max = cy.validator(float, c_max, [0.2, 5.0], "c_max")

    def s_function__(self, r_vector=None):
        f = 0.5
        l = 1.5
        # Eq.(2.3) in the paper
        return f * np.exp(-r_vector / l) - np.exp(-r_vector)

    cdef void evolve(self, int epoch_c):
        # The social interaction of every grasshopper with all the others is vectorized over the swarm;
        # in sequential mode each agent sees the rows replaced before it.
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        # Eq.(2.8) in the paper
        c = self.c_max - epoch * ((self.c_max - self.c_min) / self.epoch)
        N = self.generator.normal(0, 1, (n, d))  # one draw of d normals per agent
        ran = (c / 2) * (ub - lb)
        for idx in range(0, self.pop_size):
            diff = Xp[idx] - Xp[:self.pop_size]
            dist = np.sqrt(np.sum(diff ** 2, axis=1))
            r_ij_vector = diff / (dist[:, None] + self.EPSILON)  # xj - xi / dij in Eq.(2.7)
            xj_xi = 2 + np.remainder(dist, 2)  # |xjd - xid| in Eq. (2.7)
            s_ij = ran * self.s_function__(xj_xi)[:, None] * r_ij_vector
            S_i_total = np.add.reduce(s_ij, axis=0)
            x_new = c * N[idx] * S_i_total + g_best  # Eq. (2.7) in the paper
            ops.commit(self, pop, cand, idx, self.correct_solution(x_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

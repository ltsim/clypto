#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalDE(LegacyNativeOptimizer):
    """
    The original version of: Differential Evolution (DE)

    Links:
        1. https://doi.org/10.1016/j.swevo.2018.10.006

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wf (float): [-1., 1.0], weighting factor, default = 0.1
        + cr (float): [0.5, 0.95], crossover rate, default = 0.9
        + strategy (int): [0, 5], there are lots of variant version of DE algorithm,
            + 0: DE/current-to-rand/1/bin
            + 1: DE/best/1/bin
            + 2: DE/best/2/bin
            + 3: DE/rand/2/bin
            + 4: DE/current-to-best/1/bin
            + 5: DE/current-to-rand/1/bin

    Examples
    ~~~~~~~~
    >>> from clypto.collection.evolutionary_based import DE    >>> import numpy as np
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
    >>> model = DE.OriginalDE(epoch=1000, pop_size=50, wf = 0.7, cr = 0.9, strategy = 0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Jambi, K.M., 2019. Novel mutation strategy for enhancing SHADE and
    LSHADE algorithms for global numerical optimization. Swarm and Evolutionary Computation, 50, p.100455.
    """

    cdef public object wf
    cdef public object cr
    cdef public object strategy

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        wf: float = 0.1,
        cr: float = 0.9,
        strategy: int = 0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wf (float): weighting factor, default = 0.1
            cr (float): crossover rate, default = 0.9
            strategy (int): Different variants of DE, default = 0
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "wf", "cr", "strategy"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.wf = cy.validator(float, wf, (-3.0, 3.0), "wf")
        self.cr = cy.validator(float, cr, (0, 1.0), "cr")
        self.strategy = cy.validator(int, strategy, [0, 5], "strategy")

    def mutation__(self, current_pos, new_pos):
        condition = self.generator.random(self.problem.n_dims) < self.cr
        pos_new = np.where(condition, new_pos, current_pos)
        return self.correct_solution(pos_new)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        # the agents read the rows replaced before them (sequential mode)
        for idx in range(0, self.pop_size):
            if self.strategy == 0:
                # Choose 3 random element and different to i
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = Xp[idx_list[0]] + self.wf * (Xp[idx_list[1]] - Xp[idx_list[2]])
            elif self.strategy == 1:
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = g_best + self.wf * (Xp[idx_list[0]] - Xp[idx_list[1]])
            elif self.strategy == 2:
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 4, replace=False)
                pos_new = (
                    g_best
                    + self.wf * (Xp[idx_list[0]] - Xp[idx_list[1]])
                    + self.wf * (Xp[idx_list[2]] - Xp[idx_list[3]])
                )
            elif self.strategy == 3:
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 5, replace=False)
                pos_new = (
                    Xp[idx_list[0]]
                    + self.wf * (Xp[idx_list[1]] - Xp[idx_list[2]])
                    + self.wf * (Xp[idx_list[3]] - Xp[idx_list[4]])
                )
            elif self.strategy == 4:
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = (
                    Xp[idx]
                    + self.wf * (g_best - Xp[idx])
                    + self.wf * (Xp[idx_list[0]] - Xp[idx_list[1]])
                )
            else:
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = (
                    Xp[idx]
                    + self.wf * (Xp[idx_list[0]] - Xp[idx])
                    + self.wf * (Xp[idx_list[1]] - Xp[idx_list[2]])
                )
            pos_new = self.mutation__(Xp[idx], pos_new)
            ops.commit(self, pop, cand, idx, pos_new, swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

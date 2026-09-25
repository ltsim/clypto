#!/usr/bin/env python
# Created by "Thieu" at 17:52, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalTSO(LegacyNativeOptimizer):
    """
    The original version of: Tuna Swarm Optimization (TSO)

    Notes:
        1. Two variables that authors consider it as a constants (aa = 0.7 and zz = 0.05)
        2. https://www.hindawi.com/journals/cin/2021/9210050/
        3. https://www.mathworks.com/matlabcentral/fileexchange/101734-tuna-swarm-optimization

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import TSO    >>> import numpy as np
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
    >>> model = TSO.OriginalTSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Xie, L., Han, T., Zhou, H., Zhang, Z. R., Han, B., & Tang, A. (2021). Tuna swarm optimization: a novel swarm-based
    metaheuristic algorithm for global optimization. Computational intelligence and Neuroscience, 2021.
    """

    cdef public object aa
    cdef public object zz

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

    cdef void initialize_variables(self):
        self.aa = 0.7
        self.zz = 0.05

    def get_new_local_pos__(self, C, a1, a2, t, epoch, g_best, Xp):
        if self.generator.random() < self.zz:
            local_pos = self.problem.generate_solution()
        else:
            if self.generator.random() < 0.5:
                r1 = self.generator.random()
                beta = np.exp(
                    r1 * np.exp(3 * np.cos(np.pi * ((self.epoch - epoch) / self.epoch)))
                ) * np.cos(2 * np.pi * r1)
                if self.generator.random() < C:
                    local_pos = (
                            a1
                            * (
                                    g_best
                                    + beta * np.abs(g_best - Xp[0])
                            )
                            + a2 * Xp[0]
                    )  # Eq (8.3)
                else:
                    rand_pos = self.problem.generate_solution()
                    local_pos = (
                            a1 * (rand_pos + beta * np.abs(rand_pos - Xp[0]))
                            + a2 * Xp[0]
                    )  # Eq (8.1)
            else:
                tf = self.generator.choice([-1, 1])
                if self.generator.random() < 0.5:
                    local_pos = tf * t ** 2 * Xp[0]  # Eq 9.2
                else:
                    local_pos = (
                            g_best
                            + self.generator.random(self.problem.n_dims)
                            * (g_best - Xp[0])
                            + tf * t ** 2 * (g_best - Xp[0])
                    )
        return local_pos

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        C = epoch / self.epoch
        a1 = self.aa + (1 - self.aa) * C
        a2 = (1 - self.aa) - (1 - self.aa) * C
        tt = (1 - epoch / self.epoch) ** (epoch / self.epoch)
        for idx in range(0, self.pop_size):
            if idx == 0:
                pos_new = self.get_new_local_pos__(C, a1, a2, tt, epoch, g_best, Xp)
            else:
                if self.generator.random() < self.zz:
                    pos_new = self.problem.generate_solution()
                else:
                    if self.generator.random() > 0.5:
                        r1 = self.generator.random()
                        beta = np.exp(
                            r1
                            * np.exp(
                                3 * np.cos(np.pi * (self.epoch - epoch) / self.epoch)
                            )
                        ) * np.cos(2 * np.pi * r1)
                        if self.generator.random() < C:
                            pos_new = (
                                    a1
                                    * (
                                            g_best
                                            + beta
                                            * np.abs(
                                        g_best - Xp[idx]
                                    )
                                    )
                                    + a2 * Xp[idx - 1]
                            )  # Eq. 8.4
                        else:
                            rand_pos = self.problem.generate_solution()
                            pos_new = (
                                    a1
                                    * (
                                            rand_pos
                                            + beta * np.abs(rand_pos - Xp[idx])
                                    )
                                    + a2 * Xp[idx - 1]
                            )  # Eq 8.2
                    else:
                        tf = self.generator.choice([-1, 1])
                        if self.generator.random() < 0.5:
                            pos_new = (
                                    g_best
                                    + self.generator.random(self.problem.n_dims)
                                    * (g_best - Xp[idx])
                                    + tf
                                    * tt ** 2
                                    * (g_best - Xp[idx])
                            )  # Eq 9.1
                        else:
                            pos_new = tf * tt ** 2 * Xp[idx]  # Eq 9.2
            Xc[idx] = self.correct_solution(pos_new)
        # every agent is replaced by its candidate
        self.evaluate(cand, 0, n)
        self.pop = cand

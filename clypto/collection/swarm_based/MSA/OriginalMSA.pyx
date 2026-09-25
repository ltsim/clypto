#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from math import gamma
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalMSA(LegacyNativeOptimizer):
    """
    The original version: Moth Search Algorithm (MSA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/59010-moth-search-ms-algorithm
        2. https://doi.org/10.1007/s12293-016-0212-3

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_best (int): [3, 10], how many of the best moths to keep from one generation to the next, default=5
        + partition (float): [0.3, 0.8], The proportional of first partition, default=0.5
        + max_step_size (float): [0.5, 2.0], Max step size used in Levy-flight technique, default=1.0

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import MSA    >>> import numpy as np
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
    >>> model = MSA.OriginalMSA(epoch=1000, pop_size=50, n_best = 5, partition = 0.5, max_step_size = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., 2018. Moth search algorithm: a bio-inspired metaheuristic algorithm for
    global optimization problems. Memetic Computing, 10(2), pp.151-164.
    """

    cdef public object n_best
    cdef public object partition
    cdef public object max_step_size
    cdef public object n_moth1
    cdef public object n_moth2
    cdef public object golden_ratio

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_best: int = 5,
        partition: float = 0.5,
        max_step_size: float = 1.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_best (int): how many of the best moths to keep from one generation to the next, default=5
            partition (float): The proportional of first partition, default=0.5
            max_step_size (float): Max step size used in Levy-flight technique, default=1.0
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "n_best", "partition", "max_step_size"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.n_best = cy.validator(int, n_best, [2, int(self.pop_size / 2)], "n_best")
        self.partition = cy.validator(float, partition, (0, 1.0), "partition")
        self.max_step_size = cy.validator(float, max_step_size, (0, 5.0), "max_step_size")
        self.n_moth1 = int(np.ceil(self.partition * self.pop_size))
        self.n_moth2 = self.pop_size - self.n_moth1
        self.golden_ratio = (np.sqrt(5) - 1) / 2.0

    def _levy_walk(self, iteration):
        beta = 1.5  # Eq. 2.23
        sigma = (
                        gamma(1 + beta)
                        * np.sin(np.pi * (beta - 1) / 2)
                        / (gamma(beta / 2) * (beta - 1) * 2 ** ((beta - 2) / 2))
                ) ** (1 / (beta - 1))
        u = self.generator.uniform(self.problem.lb, self.problem.ub) * sigma
        v = self.generator.uniform(self.problem.lb, self.problem.ub)
        step = u / np.abs(v) ** (1.0 / (beta - 1))  # Eq. 2.21
        scale = self.max_step_size / iteration
        delta_x = scale * step
        return delta_x

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        pop_best = pop.take(np.arange(self.n_best))
        for idx in range(0, self.pop_size):
            # Migration operator
            if idx < self.n_moth1:
                pos_new = Xp[idx] + self.generator.random(d) * self._levy_walk(epoch)
            else:
                # Flying in a straight line
                temp_case1 = Xp[idx] + self.generator.random(d) * self.golden_ratio * (g_best - Xp[idx])
                temp_case2 = Xp[idx] + self.generator.random(d) * (1.0 / self.golden_ratio) * (g_best - Xp[idx])
                pos_new = np.where(self.generator.random(d) < 0.5, temp_case2, temp_case1)
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand, old_first=True)
        pop = self.pop = pop.take(self.sorted_order(pop))
        # Replace the worst with the previous generation's elites.
        for idx in range(0, self.n_best):
            pop.buf[n - 1 - idx] = pop_best.buf[idx]

#!/usr/bin/env python
# Created by "Thieu" at 23:41, 15/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalCDDO(LegacyNativeOptimizer):
    """
    The original version of: Child Drawing Development Optimization (CCDO)

    Notes:
        + This source code was converted from the original Matlab implementation in the paper into Python.
        The Matlab code itself has many issues, for example, parameters are defined but never used.
        Several variables are declared, such as p1, p2, p3. Parameters like child skill rate and child level
        rate are initialized as hyperparameters at the beginning, but inside the loop they are randomly generated,
        which is inconsistent with the paper.

        + Moreover, the biggest flaw of this algorithm lies in the if–else condition during the update process.
        There is a high chance that neither condition will be executed, because the golden ratio is not necessarily
        within the interval [1.5, 2], as it is computed based on a random position. In addition, when comparing
        the position with a random integer T (hand pressure), it is unclear why this is done. It is highly likely
        that the algorithm will only execute that single condition.

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import CDDO    >>> import numpy as np
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
    >>> model = CDDO.OriginalCDDO(epoch=1000, pop_size=50, pattern_size=10, creativity_rate=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdulhameed, S., Rashid, T.A. Child Drawing Development Optimization Algorithm Based on
    Child’s Cognitive Development. Arab J Sci Eng 47, 1337–1351 (2022). https://doi.org/10.1007/s13369-021-05928-6
    """

    cdef public object pattern_size
    cdef public object creativity_rate
    cdef public object LR
    cdef public object SR
    cdef public object pop_local
    cdef public object list_gr

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pattern_size = 10,
        creativity_rate = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pattern_size (int): size of the pattern matrix, default = 10
            creativity_rate (float): creativity rate, default = 0.1
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pattern_size", "creativity_rate"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pattern_size = cy.validator(int, pattern_size, [1, 1000], "pattern_size")
        self.creativity_rate = cy.validator(float, creativity_rate, [0.0, 1.0], "creativity_rate")

    cdef void before_main_loop(self):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        self.LR = rng.uniform(0.1, 1.0)  # Child level rate
        self.SR = rng.uniform(0.1, 1.0)  # Child Skill Rate
        self.pop_local = pop.take(np.arange(n))
        p1 = rng.integers(0, d, size=n)
        p2 = rng.integers(0, d, size=n)
        x1, x2 = pop.X[np.arange(n), p1], pop.X[np.arange(n), p2]
        self.list_gr = np.where(x1 == 0, x2, x1 + x2 / np.where(x1 == 0, 1, x1))

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        Pl = self.pop_local.X
        g = np.array(self.g_best_x())
        pattern = np.array(X[self.sorted_order(pop)[:self.pattern_size]])
        hand_pressure = rng.integers(self.problem.lb[0], self.problem.ub[0] + 1, size=n)
        pp = rng.integers(0, d, size=n)
        cond1 = X[np.arange(n), pp] <= hand_pressure
        cond2 = ~cond1 & (1.5 < self.list_gr) & (self.list_gr < 2)
        pos1 = self.list_gr[:, None] + self.SR * rng.random((n, d)) * (Pl - X) + self.LR * rng.random((n, d)) * (g - X)
        pos2 = pattern[rng.integers(0, self.pattern_size, size=n)] - self.creativity_rate * Pl
        pos = np.where(cond1[:, None], pos1, np.where(cond2[:, None], pos2, X))
        if cond1.any():
            self.LR = rng.integers(6, 11) / 10
            self.SR = rng.integers(6, 11) / 10
        elif cond2.any():
            self.LR = rng.integers(0, 6) / 10
            self.SR = rng.integers(0, 6) / 10
        ops.replace(self, pos)
        ops.greedy(self, self.pop, dst=self.pop_local)

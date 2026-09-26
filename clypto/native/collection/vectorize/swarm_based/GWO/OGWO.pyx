#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OGWO(LegacyNativeOptimizer):
    """
    The original version of: Opposition-based learning Grey Wolf Optimizer (OGWO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2021.107139

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.OGWO(epoch=1000, pop_size=50, miu_factor=2.0, jumping_rate=0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yu, X., Xu, W., & Li, C. (2021). Opposition-based learning grey wolf optimizer for global optimization. Knowledge-Based Systems, 226, 107139.
    """

    cdef public double miu_factor
    cdef public double jumping_rate

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        miu_factor: float = 2.0,
        jumping_rate: float = 0.05,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_factor (float): nonlinear coefficient for equation (11), default = 2.0
            jumping_rate (float):  jumping rate for OBL, default = 0.05
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "miu_factor", "jumping_rate"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.miu_factor = cy.validator(float, miu_factor, [0.0, 10.0], "miu_factor")
        self.jumping_rate = cy.validator(float, jumping_rate, [0.0, 1.0], "jumping_rate")

    cdef void initialization(self):
        LegacyNativeOptimizer.initialization(self)
        self.merge_opposition()

    def merge_opposition(self):
        # Opposition population using equation (12); keep the best pop_size of both
        cdef NativePopulation pop = self.pop
        cdef NativePopulation merged = pop.concat(self.new_population(self.problem.lb + self.problem.ub - pop.X))
        self.pop = merged.take(self.sorted_order(merged)[:self.pop_size])

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t n = pop.n, d = pop.d
        # linearly decreased from 2 to 0
        a = 2.0 * (1 - (<object>(epoch / self.epoch)) ** (<object>self.miu_factor))
        best = pop.X[self.sorted_order(pop)[:3]][None]
        R = self.generator.random((n, 6, d))  # per agent: A1..A3 then C1..C3 draws
        A = a * (2 * R[:, :3] - 1)
        C = 2 * R[:, 3:]
        Xs = best - A * np.abs(C * best - pop.X[:, None, :])
        cand.X[:] = self.correct_solution((Xs[:, 0] + Xs[:, 1] + Xs[:, 2]) / 3.0)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

        # Apply opposition-based learning
        if self.generator.random() < self.jumping_rate:
            self.merge_opposition()

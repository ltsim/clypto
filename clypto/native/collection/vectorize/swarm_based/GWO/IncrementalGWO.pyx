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


cdef class IncrementalGWO(LegacyNativeOptimizer):
    """
    The original version of: Incremental model-based Grey Wolf Optimizer (IncrementalGWO)

    Notes:
        + When calling the solve() function, you need to set the mode to "swarm" to use this algorithm as original version.
        + They update the position of whole population before calculating the fitness of each agent.

    Links:
        1. https://doi.org/10.1007/s00366-019-00837-7

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.IncrementalGWO(epoch=1000, pop_size=50, explore_factor=1.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Seyyedabbasi, A., & Kiani, F. (2021). I-GWO and Ex-GWO: improved algorithms of the Grey Wolf Optimizer to solve global optimization problems. Engineering with Computers, 37(1), 509-532.
    """

    cdef public double explore_factor

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        explore_factor: float = 1.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            explore_factor (float): factor to control exploration, default = 1.5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "explore_factor"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.explore_factor = cy.validator(float, explore_factor, [0.0, 5.0], "explore_factor")

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        # linearly decreased from 2 to 0
        a = 2 * (1.0 - (<object>(epoch / self.epoch)) ** (<object>self.explore_factor))
        ps = np.ascontiguousarray(pop.X[self.sorted_order(pop)])  # sorted positions
        pos = np.empty((n, d))
        # Alpha wolf updates based on hunting mechanism
        A = a * (2 * self.generator.random(d) - 1)
        C = 2 * self.generator.random(d)
        pos[0] = ps[0] - A * np.abs(C * ps[0] - pop.X[0])
        # Other wolves update based on all previous wolves (Equation 19)
        for idx in range(1, n):
            mask = np.arange(n) != idx
            pos[idx] = ps[mask].mean(axis=0)
        cand.X[:] = self.correct_solution(pos)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

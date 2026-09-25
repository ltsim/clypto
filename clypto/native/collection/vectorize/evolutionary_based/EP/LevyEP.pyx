#!/usr/bin/env python
# Created by "Thieu" at 19:27, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.native.collection.vectorize.evolutionary_based.EP.OriginalEP cimport OriginalEP
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class LevyEP(OriginalEP):
    """
    The developed Levy-flight version: Evolutionary Programming (LevyEP)

    Notes:
        + Levy-flight is applied to EP, flow and some equations is changed.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + bout_size (float): [0.05, 0.2], percentage of child agents implement tournament selection

    Examples
    ~~~~~~~~
    >>> from clypto.collection.evolutionary_based import EP    >>> import numpy as np
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
    >>> model = EP.LevyEP(epoch=1000, pop_size=50, bout_size = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        bout_size: float = 0.05,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            bout_size (float): percentage of child agents implement tournament selection
        """
        super().__init__(epoch, pop_size, bout_size, name=name, mode=mode)
        self.sort_flag = True

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation child = self.offspring__(pop)
        cdef NativePopulation comeback
        both = child.take(self.sorted_order(child)).concat(pop)
        self.tournament__(both)
        order = np.argsort(-both.field("WIN")[:, 0], kind="stable")
        pop_new = both.take(order[:self.pop_size])
        pop_left = both.take(order[self.pop_size:])
        idx_list = self.generator.choice(pop_left.n, int(0.5 * pop_left.n), replace=False)
        k = len(idx_list)
        comeback = pop_left.take(idx_list)
        comeback.X[:] = self.correct_solution(pop_left.X[idx_list] + self.get_levy_flight_step(multiplier=0.01, size=(k, pop.d), case=-1) * self.generator.random((k, 1)))
        comeback.field("S")[:] = self.generator.uniform(0, self.distance, (k, pop.d))
        comeback.field("WIN")[:] = 0
        self.evaluate(comeback, 0, k)
        merged = pop_new.concat(comeback)
        self.pop = merged.take(self.sorted_order(merged)[:self.pop_size])

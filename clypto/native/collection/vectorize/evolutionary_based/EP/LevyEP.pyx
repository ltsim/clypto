#!/usr/bin/env python
# Created by "Thieu" at 19:27, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.collection.evolutionary_based.EP.OriginalEP cimport OriginalEP
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
        cdef NativePopulation child = pop.empty_like()
        cdef NativePopulation both, pop_new, pop_left, comeback
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        Xp, S = pop.X, pop.field("S")
        for idx in range(0, self.pop_size):
            pos_new = Xp[idx] + S[idx] * self.generator.normal(0, 1.0, d)
            child.X[idx] = self.correct_solution(pos_new)
            child.field("S")[idx] = S[idx] + self.generator.normal(0, 1.0, d) * np.abs(S[idx]) ** 0.5
            self.generator.uniform(0, self.distance, d)  # generate_empty_agent draws a strategy
        child.field("WIN")[:] = 0
        self.evaluate(child, 0, n)
        # Update the global best
        both = child.take(self.sorted_order(child)).concat(pop)
        self.tournament__(both)
        ## Keep the top population, but 50% of left population will make a comeback an take the good position
        order = sorted(range(both.n), key=lambda i: both.field("WIN")[i, 0], reverse=True)
        pop_new = both.take(order[:self.pop_size])
        pop_left = both.take(order[self.pop_size:])
        ## Choice random 50% of population left
        idx_list = self.generator.choice(range(0, pop_left.n), int(0.5 * pop_left.n), replace=False)
        comeback = pop_left.take(idx_list)
        for k in range(len(idx_list)):
            pos_new = pop_left.X[idx_list[k]] + self.get_levy_flight_step(multiplier=0.01, size=d, case=0)
            comeback.X[k] = self.correct_solution(pos_new)
            strategy = self.distance = 0.05 * (self.problem.ub - self.problem.lb)
            self.generator.uniform(0, self.distance, d)  # generate_empty_agent draws a strategy
            comeback.field("S")[k] = strategy
        comeback.field("WIN")[:] = 0
        self.evaluate(comeback, 0, comeback.n)
        merged = pop_new.concat(comeback)
        self.pop = merged.take(self.sorted_order(merged)[:self.pop_size])

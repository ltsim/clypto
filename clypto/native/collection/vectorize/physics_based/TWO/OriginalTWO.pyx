#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np



from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalTWO(LegacyNativeOptimizer):
    """
    The original version of: Tug of War Optimization (TWO)

    Links:
        1. https://www.researchgate.net/publication/332088054_Tug_of_War_Optimization_Algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import TWO    >>> import numpy as np
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
    >>> model = TWO.OriginalTWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kaveh, A., 2017. Tug of war optimization. In Advances in metaheuristic algorithms for
    optimal design of structures (pp. 451-487). Springer, Cham.
    """


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
        self.muy_s = 1
        self.muy_k = 1
        self.delta_t = 1
        self.alpha = 0.99
        self.beta = 0.1

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m):
        return [("W", 1)]  # weight of each team

    cdef void init_fields(self, NativePopulation pop):
        pop.field("W")[:] = 0.0

    cdef void initialization(self):
        LegacyNativeOptimizer.initialization(self)
        self.update_weight__(self.pop)

    def update_weight__(self, NativePopulation teams):
        list_fits = np.array(teams.F)
        maxx, minn = np.max(list_fits), np.min(list_fits)
        if maxx == minn:
            list_fits = self.generator.uniform(0.0, 1.0, self.pop_size)
        list_weights = np.exp(-(list_fits - maxx) / (maxx - minn))
        list_weights = list_weights / np.sum(list_weights) + 0.1
        teams.field("W")[:self.pop_size, 0] = list_weights
        return teams

    def forces__(self, NativePopulation pop, epoch):
        """The teams pull each other (Loop 1); returns the new positions (n, d), not yet bounded.

        Team i is pulled by every heavier team j; the noise terms of the pulls are summed as one Gaussian.
        """
        lb, ub = self.problem.lb, self.problem.ub
        X, W = np.array(pop.X), np.array(pop.field("W")[:, 0])
        n, d = X.shape
        pulled = W[:, None] < W[None, :]  # (i, j): j heavier than i
        force = np.maximum(W[:, None] * self.muy_s, W[None, :] * self.muy_s)
        C = np.where(pulled, (force - W[:, None] * self.muy_k) / (W[:, None] * self.muy_k), 0.0)
        noise = np.power(self.alpha, epoch) * self.beta * (ub - lb) * np.sqrt(pulled.sum(axis=1))[:, None] * self.generator.normal(0, 1, (n, d))
        return X + 0.5 * (C @ X - C.sum(axis=1)[:, None] * X) + noise

    def bound__(self, pos, epoch):
        """Out-of-bound coordinates: half the time re-drawn around the best, else clipped (returns the position)."""
        lb, ub = self.problem.lb, self.problem.ub
        g = np.array(self.g_best_x())
        out = (pos < lb) | (pos > ub)
        around = g + self.generator.standard_normal(pos.shape) / epoch * (g - pos)
        around = np.where((around < lb) | (around > ub), pos, around)
        return np.where(out & (self.generator.random(pos.shape) <= 0.5), around, np.where(out, np.clip(pos, lb, ub), pos))

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        pos = self.forces__(pop, epoch_c)
        pop.X[:] = self.correct_solution(self.bound__(pos, epoch_c))
        self.evaluate(pop, 0, pop.n)
        self.update_weight__(pop)

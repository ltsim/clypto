#!/usr/bin/env python
# Created by "Thieu" at 04:43, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalDO(LegacyNativeOptimizer):
    """
    The original version of: Dragonfly Optimization (DO)

    Links:
        1. https://link.springer.com/article/10.1007/s00521-015-1920-1

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import DO    >>> import numpy as np
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
    >>> model = DO.OriginalDO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2016. Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective,
    discrete, and multi-objective problems. Neural computing and applications, 27(4), pp.1053-1073.
    """

    cdef public object pop_delta
    cdef public object radius
    cdef public object delta_max

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

    cdef void initialization(self):
        LegacyNativeOptimizer.initialization(self)
        self.pop_delta = self.generate_population(self.pop_size)
        self.radius = (self.problem.ub - self.problem.lb) / 10
        self.delta_max = (self.problem.ub - self.problem.lb) / 10

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = np.array(pop.X)
        D = np.array(self.pop_delta.X)
        lb, ub = self.problem.lb, self.problem.ub
        F = np.asarray(pop.F)
        g = X[ops.best_row(self, pop)]
        gw = X[int(F.argmax() if self.problem.minmax == "min" else F.argmin())]
        r = (ub - lb) / 4 + ((ub - lb) * (2 * epoch_c / self.epoch))
        w = 0.9 - epoch_c * ((0.9 - 0.4) / self.epoch)
        my_c = max(0.1 - epoch_c * ((0.1 - 0) / (self.epoch / 2)), 0)
        s = 2 * rng.random() * my_c  # Seperation weight
        a = 2 * rng.random() * my_c  # Alignment weight
        c = 2 * rng.random() * my_c  # Cohesion weight
        f = 2 * rng.random()  # Food attraction weight
        e = my_c  # Enemy distraction weight
        # neighbours: all dimensions within the radius (and not the same point)
        diff = np.abs(X[:, None, :] - X[None, :, :])
        M = (np.all(diff <= r, axis=2) & np.all(diff != 0, axis=2)).astype(float)
        count = M.sum(axis=1)
        many = (count > 1)[:, None]
        cnt = np.maximum(count, 1)[:, None]
        sum_pos = M @ X
        S = np.where(many, sum_pos - count[:, None] * X, 0.0)
        A = np.where(many, (M @ D) / cnt, D)
        C = np.where(many, sum_pos / cnt, X) - X
        near_food = np.all(np.abs(X - g) <= r, axis=1)[:, None]
        Fd = np.where(near_food, g - X, 0.0)
        enemy = np.where(np.all(np.abs(X - gw) <= r, axis=1)[:, None], gw + X, 0.0)
        temp = w * D + rng.uniform(0, 1, (n, d)) * A + rng.uniform(0, 1, (n, d)) * C + rng.uniform(0, 1, (n, d)) * S
        temp = np.clip(temp, -1 * self.delta_max, self.delta_max)
        levy = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=(n, 1), case=-1)
        near = (a * A + c * C + s * S + f * Fd + e * enemy) + w * D
        far = ~near_food  # any dimension beyond the radius
        delta_new = np.where(far, np.where(many, temp, 0.0), near)
        pos = X + np.where(far, np.where(many, temp, levy * X), near)
        ops.step(self, pos)
        cand = self.pop_delta.empty_like()
        cand.X[:] = self.correct_solution(delta_new)
        self.evaluate(cand, 0, n)
        ops.greedy(self, cand, dst=self.pop_delta)

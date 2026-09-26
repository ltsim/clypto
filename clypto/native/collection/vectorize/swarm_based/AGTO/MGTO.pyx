#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class MGTO(LegacyNativeOptimizer):
    """
    The original version of: Modified Gorilla Troops Optimization (mGTO)

    Notes (parameters):
        1. pp (float): the probability of transition in exploration phase (p in the paper), default = 0.03

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import AGTO    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = AGTO.MGTO(epoch=1000, pop_size=50, pp=0.03)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mostafa, R. R., Gaheen, M. A., Abd ElAziz, M., Al-Betar, M. A., & Ewees, A. A. (2023). An improved gorilla
    troops optimizer for global optimization problems and feature selection. Knowledge-Based Systems, 110462.
    """

    cdef public object pp

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pp: float = 0.03,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pp (float): the probability of transition in exploration phase (p in the paper), default = 0.03
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pp"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pp = cy.validator(float, pp, (0, 1), "p1")

    cdef object amend_solution(self, object solution):
        condition = np.logical_and(
            self.problem.lb <= solution, solution <= self.problem.ub
        )
        random_pos = self.generator.uniform(self.problem.lb, self.problem.ub, size=np.shape(solution))
        return np.where(condition, solution, random_pos)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = np.array(pop.X)
        lb, ub = self.problem.lb, self.problem.ub
        F = 1 + np.cos(2 * rng.random())
        C = F * (1 - epoch_c / self.epoch)
        L = C * rng.choice([-1, 1])
        # opposition-based population (replaces the population)
        ops.replace(self, X.min(axis=0) + X.max(axis=0) - X)
        ## Exploration
        X = pop.X
        rand_agent = X[rng.integers(0, n, size=n)]
        pos_a = (rng.random((n, 1)) - C) * rand_agent + L * rng.uniform(-C, C, (n, 1)) * X
        id1, id2 = ops.two_others(self, n, 1)
        pos_b = X - L * (L * X - X[id1[:, 0]]) + rng.random((n, 1)) * (X - X[id2[:, 0]])
        pos = np.where((rng.random(n) >= 0.5)[:, None], pos_a, pos_b)
        pos = np.where((rng.random(n) < self.pp)[:, None], lb + rng.random((n, d)) * (ub - lb), pos)
        ops.step(self, pos)
        ## Exploitation, around the best agent found so far
        X = pop.X
        g = np.array(X[ops.best_row(self, self.pop)])
        if np.abs(C) >= 1:
            gg = rng.choice([-0.5, 2.0], size=(n, 1))
            M = (np.abs(np.mean(np.ascontiguousarray(X), axis=0)) ** gg) ** (1.0 / gg)
            pos = L * M * (X - g) * (0.01 * np.tan(np.pi * (rng.uniform(0, 1, (n, d)) - 0.5)))
        else:
            pos = g - (2 * rng.random((n, 1)) - 1) * (g - X) * np.tan(rng.uniform(0, 1, (n, 1)) * np.pi / 2)
        ops.step(self, pos)

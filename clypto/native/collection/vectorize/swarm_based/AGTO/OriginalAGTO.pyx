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


cdef class OriginalAGTO(LegacyNativeOptimizer):
    """
    The original version of: Artificial Gorilla Troops Optimization (AGTO)

    Links:
        1. https://doi.org/10.1002/int.22535
        2. https://www.mathworks.com/matlabcentral/fileexchange/95953-artificial-gorilla-troops-optimizer

    Notes (parameters):
        1. p1 (float): the probability of transition in exploration phase (p in the paper), default = 0.03
        2. p2 (float): the probability of transition in exploitation phase (w in the paper), default = 0.8
        3. beta (float): coefficient in updating equation, should be in [-5.0, 5.0], default = 3.0

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import AGTO    >>> import numpy as np
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
    >>> model = AGTO.OriginalAGTO(epoch=1000, pop_size=50, p1=0.03, p2=0.8, beta=3.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021). Artificial gorilla troops optimizer: a new
    nature‐inspired metaheuristic algorithm for global optimization problems. International Journal of Intelligent Systems, 36(10), 5887-5958.
    """

    cdef public object p1
    cdef public object p2
    cdef public object beta

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        p1: float = 0.03,
        p2: float = 0.8,
        beta: float = 3.0,
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
            parameters=["epoch", "pop_size", "p1", "p2", "beta"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.p1 = cy.validator(float, p1, (0, 1), "p1")
        self.p2 = cy.validator(float, p2, (0, 1), "p2")
        self.beta = cy.validator(float, beta, [-10.0, 10.0], "beta")

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        lb, ub = self.problem.lb, self.problem.ub
        a = (np.cos(2 * rng.random()) + 1) * (1 - epoch_c / self.epoch)
        c = a * (2 * rng.random() - 1)
        ## Exploration
        z = rng.uniform(-a, a, (n, d))
        rand_agent = X[rng.integers(0, n, size=n)]
        pos_a = (rng.random((n, 1)) - a) * rand_agent + c * z * X
        id1, id2 = ops.two_others(self, n, 1)
        pos_b = X - c * (c * X - X[id1[:, 0]]) + rng.random((n, 1)) * (X - X[id2[:, 0]])
        pos = np.where((rng.random(n) >= 0.5)[:, None], pos_a, pos_b)
        pos = np.where((rng.random(n) < self.p1)[:, None], lb + rng.random((n, d)) * (ub - lb), pos)
        ops.step(self, pos)
        ## Exploitation, around the best agent found so far
        X = pop.X
        g = np.array(X[ops.best_row(self, self.pop)])
        if a >= self.p2:
            g2 = 2 ** c
            delta = (np.abs(np.mean(np.ascontiguousarray(X), axis=0)) ** g2) ** (1.0 / g2)
            pos = c * delta * (X - g) + X
        else:
            h = np.where((rng.random(n) >= 0.5)[:, None], rng.normal(0, 1, (n, d)), rng.normal(0, 1, (n, 1)))
            pos = g - (2 * rng.random((n, 1)) - 1) * (g - X) * (self.beta * h)
        ops.step(self, pos)

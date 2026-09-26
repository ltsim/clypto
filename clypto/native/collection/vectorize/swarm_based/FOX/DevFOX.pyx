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


cdef class DevFOX(LegacyNativeOptimizer):
    """
    The developed version of: Fox Optimizer (FOX)

    Notes (parameters):
        1. c1 (float): the coefficient of jumping (c1 in the paper), default = 0.18
        2. c2 (float): the coefficient of jumping (c2 in the paper), default = 0.82
        3. pp (float): the probability of choosing the exploration and exploitation phase, default=0.5

    Notes:
        1. Set parameter pp = 0.18 if you want to same as Original version
        2. The different between Dev and Original version is the equation: self.g_best.solution + self.generator.standard_normal(self.problem.n_dims) * (self.mint * aa)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import FOX    >>> import numpy as np
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
    >>> model = FOX.DevFOX(epoch=1000, pop_size=50, c1=0.18, c2=0.82, pp=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohammed, H., & Rashid, T. (2023). FOX: a FOX-inspired optimization algorithm. Applied Intelligence, 53(1), 1030-1050.
    """

    cdef public object c1
    cdef public object c2
    cdef public object pp
    cdef public object mint

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 0.18,
        c2: float = 0.82,
        pp = 0.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "c1", "c2", "pp"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.c1 = cy.validator(float, c1, (-100.0, 100.0), "c1")
        self.c2 = cy.validator(float, c2, (-100.0, 100.0), "c2")
        self.pp = cy.validator(float, pp, (0.0, 1.0), "pp")

    cdef void initialize_variables(self):
        self.mint = 10000000

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        g = np.array(self.g_best_x())
        aa = 2 * (1 - (1.0 / self.epoch))
        jump_mask = rng.random(n) >= 0.5
        T1 = rng.random((n, d))
        tt = np.mean(T1, axis=1)
        jump = 0.5 * 9.81 * (tt / 2) ** 2
        coef = np.where(rng.random(n) > self.pp, self.c1, self.c2)
        pos_jump = 0.5 * (g / T1) * T1 * (jump * coef)[:, None]
        pos_walk = g + rng.standard_normal((n, d)) * (self.mint * aa)
        if jump_mask.any():
            self.mint = min(self.mint, float(tt[jump_mask].min()))
        ops.replace(self, np.where(jump_mask[:, None], pos_jump, pos_walk))

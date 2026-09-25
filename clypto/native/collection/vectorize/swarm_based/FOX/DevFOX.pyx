#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


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
    >>> from clypto.collection.swarm_based import FOX    >>> import numpy as np
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
        # Every agent is replaced by its candidate (nothing reads the population in the loop).
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        g_best = np.array(self.g_best_x())
        aa = 2 * (1 - (1.0 / self.epoch))
        Xc = cand.X
        for idx in range(0, self.pop_size):
            if self.generator.random() >= 0.5:
                t1 = self.generator.random(d)
                sps = g_best / t1
                dis = 0.5 * sps * t1
                tt = np.mean(t1)
                t = tt / 2
                jump = 0.5 * 9.81 * t ** 2
                if self.generator.random() > self.pp:
                    pos_new = dis * jump * self.c1
                else:
                    pos_new = dis * jump * self.c2
                if self.mint > tt:
                    self.mint = tt
            else:
                pos_new = g_best + self.generator.standard_normal(d) * (self.mint * aa)
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        self.pop = cand

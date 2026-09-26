#!/usr/bin/env python
# Created by "Thieu" at 12:48, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevSBO(LegacyNativeOptimizer):
    """
    The developed version: Satin Bowerbird Optimizer (SBO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2017.01.006

    Notes:
        The original version can't handle negative fitness value.
        I remove all third loop for faster training, remove equation (1, 2) in the paper, calculate probability by roulette-wheel.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.5, 3.0] -> better [0.5, 2.0], the greatest step size
        + p_m (float): (0, 1.0) -> better [0.01, 0.2], mutation probability
        + psw (float): (0, 1.0) -> better [0.01, 0.1], proportion of space width (z in the paper)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.bio_based import SBO    >>> import numpy as np
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
    >>> model = SBO.DevSBO(epoch=1000, pop_size=50, alpha = 0.9, p_m =0.05, psw = 0.02)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        alpha: float = 0.94,
        p_m: float = 0.05,
        psw: float = 0.02,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): the greatest step size, default=0.94
            p_m (float): mutation probability, default=0.05
            psw (float): proportion of space width (z in the paper), default=0.02
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "alpha", "p_m", "psw"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.alpha = cy.validator(float, alpha, [0.5, 3.0], "alpha")
        self.p_m = cy.validator(float, p_m, (0, 1.0), "p_m")
        self.psw = cy.validator(float, psw, (0, 1.0), "psw")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        self.sigma = self.psw * (self.problem.ub - self.problem.lb)
        rdx = ops.roulette(self, pop.F, n)
        lamda = self.alpha * rng.uniform(size=(n, 1))
        pos = X + lamda * ((X[rdx] + g) / 2 - X)
        pos = np.where(rng.random((n, d)) < self.p_m, X + rng.normal(0, 1, (n, d)) * self.sigma, pos)
        ops.step(self, pos)

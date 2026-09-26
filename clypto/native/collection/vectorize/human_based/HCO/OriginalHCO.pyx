#!/usr/bin/env python
# Created by "Thieu" at 08:57, 12/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalHCO(LegacyNativeOptimizer):
    """
    The original version of: Human Conception Optimizer (HCO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/124200-human-conception-optimizer-hco
        2. https://www.nature.com/articles/s41598-022-25031-6

    Notes:
        1. This algorithm shares some similarities with the PSO algorithm (equations)
        2. The implementation of Matlab code is kinda different to the paper

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wfp (float): (0, 1.) - weight factor for probability of fitness selection, default=0.65
        + wfv (float): (0, 1.0) - weight factor for velocity update stage, default=0.05
        + c1 (float): (0., 3.0) - acceleration coefficient, same as PSO, default=1.4
        + c2 (float): (0., 3.0) - acceleration coefficient, same as PSO, default=1.4

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import HCO    >>> import numpy as np
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
    >>> model = HCO.OriginalHCO(epoch=1000, pop_size=50, wfp=0.65, wfv=0.05, c1=1.4, c2=1.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Acharya, D., & Das, D. K. (2022). A novel Human Conception Optimizer for solving optimization problems. Scientific Reports, 12(1), 21631.
    """

    cdef public object wfp
    cdef public object wfv
    cdef public object c1
    cdef public object c2
    cdef public object vec
    cdef public object pop_p

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        wfp: float = 0.65,
        wfv: float = 0.05,
        c1: float = 1.4,
        c2: float = 1.4,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wfp (float): weight factor for probability of fitness selection, default=0.65
            wfv (float): weight factor for velocity update stage, default=0.05
            c1 (float): acceleration coefficient, same as PSO, default=1.4
            c2 (float): acceleration coefficient, same as PSO, default=1.4
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "wfp", "wfv", "c1", "c2"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.wfp = cy.validator(float, wfp, [0, 1.0], "wfp")
        self.wfv = cy.validator(float, wfv, [0, 1.0], "wfv")
        self.c1 = cy.validator(float, c1, [0.0, 100.0], "c1")
        self.c2 = cy.validator(float, c2, [1.0, 100.0], "c2")

    cdef void initialization(self):
        cdef NativePopulation pop, cand
        cdef Py_ssize_t n, d
        LegacyNativeOptimizer.initialization(self)
        pop = self.pop
        n, d = pop.n, pop.d
        lb, ub = self.problem.lb, self.problem.ub
        ops.step(self, ub + lb - pop.X)  # opposition-based initialization
        pop = self.pop
        F = np.asarray(pop.F)
        best, worst = F.min(), F.max()
        if self.problem.minmax == "max":
            best, worst = worst, best
        pfit = (worst - best) * self.wfp + best
        # agents worse than pfit are re-drawn until they are better than pfit
        bad = np.flatnonzero(ops.better(self, pfit, F))
        while len(bad):
            cand = pop.take(bad)
            cand.X[:] = self.generator.uniform(lb, ub, (len(bad), d))
            self.evaluate(cand, 0, len(bad))
            ok = ops.better(self, np.asarray(cand.F), pfit)
            pop.buf[bad[ok]] = cand.buf[ok]
            bad = bad[~ok]
        self.vec = self.generator.uniform(lb, ub, (n, d))
        self.pop_p = pop.take(np.arange(n))

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        gb_fit = self.current_g_best().target.fitness
        lamda = rng.random()
        neu = 2
        fits = np.array(pop.F)
        fit_mean = np.mean(fits)
        RR = (gb_fit - fits) ** 2
        rr = (fit_mean - fits) ** 2
        LL = gb_fit - fit_mean
        VV = lamda * ((RR - rr) / (4 * neu * LL))
        s = np.sin(2 * np.pi * epoch_c / self.epoch)
        self.vec = self.wfv * (VV[:, None] + self.vec) + self.c1 * (self.pop_p.X - X) * s + self.c2 * (g - X) * s
        ops.step(self, X + self.vec)
        ops.greedy(self, self.pop, dst=self.pop_p)  # personal bests

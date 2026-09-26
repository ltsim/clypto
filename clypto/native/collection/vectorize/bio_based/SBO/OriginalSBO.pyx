#!/usr/bin/env python
# Created by "Thieu" at 12:48, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.bio_based.SBO.DevSBO cimport DevSBO
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalSBO(DevSBO):
    """
    The original version of: Satin Bowerbird Optimizer (SBO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2017.01.006
        2. https://www.mathworks.com/matlabcentral/fileexchange/62009-satin-bowerbird-optimizer-sbo-2017

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.5, 3.0] -> better [0.5, 0.99], the greatest step size
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
    >>> model = SBO.OriginalSBO(epoch=1000, pop_size=50, alpha = 0.9, p_m=0.05, psw = 0.02)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Moosavi, S.H.S. and Bardsiri, V.K., 2017. Satin bowerbird optimizer: A new optimization algorithm
    to optimize ANFIS for software development effort estimation. Engineering Applications of Artificial Intelligence, 60, pp.1-15.
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
        super().__init__(epoch, pop_size, alpha, p_m, psw, name=name, mode=mode)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        self.sigma = self.psw * (self.problem.ub - self.problem.lb)
        fx = np.array(pop.F)
        fit = np.where(fx < 0, 1.0 + np.abs(fx), 1.0 / (1.0 + np.abs(fx)))
        prob = fit / np.sum(fit)
        # per (agent, dimension): roulette wheel on prob, step towards the mean of that agent and the best
        rdx = np.minimum(np.searchsorted(np.cumsum(prob), rng.random((n, d)), side="right"), n - 1)
        lamda = self.alpha / (1 + prob[rdx])
        pos = X + lamda * ((X[rdx, np.arange(d)[None, :]] + g) / 2 - X)
        pos = np.where(rng.random((n, d)) < self.p_m, X + rng.normal(0, 1, (n, d)) * self.sigma, pos)
        # The classic code only refreshes the fitness of the population with the candidates' values in the
        # sequential mode (positions stay) and replaces the population in swarm modes; both are kept.
        cdef NativePopulation cand = pop.empty_like()
        cand.X[:] = self.correct_solution(pos)
        self.evaluate(cand, 0, n)
        if self.mode in self.AVAILABLE_MODES:
            self.pop = cand
        else:
            pop.buf[:, :pop.cX] = cand.buf[:, :pop.cX]

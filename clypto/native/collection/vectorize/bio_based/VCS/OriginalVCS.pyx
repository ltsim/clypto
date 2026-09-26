#!/usr/bin/env python
# Created by "Thieu" at 22:07, 11/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.bio_based.VCS.DevVCS cimport DevVCS
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalVCS(DevVCS):
    """
    The original version of: Virus Colony Search (VCS)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2015.11.004

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): (0, 1.0) -> better [0.2, 0.5], Percentage of the number of the best will keep, default = 0.5
        + sigma (float): (0, 5.0) -> better [0.1, 2.0], Weight factor

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.bio_based import VCS    >>> import numpy as np
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
    >>> model = VCS.OriginalVCS(epoch=1000, pop_size=50, lamda = 0.5, sigma = 0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Li, M.D., Zhao, H., Weng, X.W. and Han, T., 2016. A novel nature-inspired algorithm
    for optimization: Virus colony search. Advances in Engineering Software, 92, pp.65-88.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        lamda: float = 0.5,
        sigma: float = 1.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            lamda (float): Number of the best will keep, default = 0.5
            sigma (float): Weight factor, default = 1.5
        """
        super().__init__(epoch, pop_size, lamda, sigma, name=name, mode=mode)

    cdef object amend_solution(self, object solution):
        condition = np.clip(solution, self.problem.lb, self.problem.ub)
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, rand_pos)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        # Phase 1: Gaussian search around the best
        sigma = (np.log1p(epoch) / self.epoch) * (X - g)
        pos = rng.normal(g, np.abs(sigma)) + rng.uniform(size=(n, 1)) * g - rng.uniform(size=(n, 1)) * X
        ops.step(self, pos)
        # Phase 2: around the (weighted) mean of the best agents
        x_mean = self.calculate_xmean__(self.pop)
        sigma = self.sigma * (1 - epoch / self.epoch)
        ops.step(self, x_mean + sigma * rng.normal(0, 1, (n, d)))
        # Phase 3: recombination with two random agents
        X = self.pop.X
        pr = ((d - np.arange(n) + 1) / d)[:, None]
        id1, id2 = ops.two_others(self, n, d)
        cols = np.arange(d)[None, :]
        pos = np.where(rng.uniform(size=(n, d)) > pr, X[id1, cols] - (X[id2, cols] - X) * rng.uniform(size=(n, d)), X)
        ops.step(self, pos)

#!/usr/bin/env python
# Created by "Thieu" at 22:07, 11/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class DevVCS(LegacyNativeOptimizer):
    """
    The developed version: Virus Colony Search (VCS)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2015.11.004

    Notes:
        + In Immune response process, updates the whole position instead of updating each variable in position

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): (0, 1.0) -> better [0.2, 0.5], Percentage of the number of the best will keep, default = 0.5
        + sigma (float): (0, 5.0) -> better [0.1, 2.0], Weight factor

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import VCS    >>> import numpy as np
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
    >>> model = VCS.DevVCS(epoch=1000, pop_size=50, lamda = 0.5, sigma = 0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
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
            lamda (float): Percentage of the number of the best will keep, default = 0.5
            sigma (float): Weight factor, default = 1.5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "lamda", "sigma"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.lamda = cy.validator(float, lamda, (0, 1.0), "lamda")
        self.sigma = cy.validator(float, sigma, (0, 5.0), "sigma")
        self.n_best = int(self.lamda * self.pop_size)

    def calculate_xmean__(self, NativePopulation pop):
        ## Calculate the weighted mean of the λ best individuals by
        pos_list = np.array(pop.X[self.sorted_order(pop)[: self.n_best]])
        factor_down = self.n_best * np.log1p(self.n_best + 1) - np.log1p(
            np.prod(range(1, self.n_best + 1))
        )
        weight = np.log1p(self.n_best + 1) / factor_down
        weight = weight / self.n_best
        x_mean = weight * np.sum(pos_list, axis=0)
        return x_mean

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        sigma = (np.log1p(epoch + 1) / self.epoch) * (X - g)
        ops.step(self, rng.normal(rng.normal(g, np.abs(sigma))) + rng.uniform(size=(n, 1)) * g - rng.uniform(size=(n, 1)) * X)
        x_mean = self.calculate_xmean__(self.pop)
        sigma = self.sigma * (1 - epoch / self.epoch)
        ops.step(self, x_mean + sigma * rng.normal(0, 1, (n, d)))
        pop = self.pop = self.pop.take(self.sorted_order(self.pop))
        X = pop.X
        pr = ((d - np.arange(n) + 1) / d)[:, None]
        id1, id2 = ops.two_others(self, n, 1)
        temp = X[id1[:, 0]] - (X[id2[:, 0]] - X) * rng.uniform(size=(n, 1))
        ops.step(self, np.where(rng.random((n, d)) < pr, X, temp))

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
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        ## Viruses diffusion
        for idx in range(0, self.pop_size):
            sigma = (np.log1p(epoch + 1) / self.epoch) * (
                    Xp[idx] - g_best
            )
            gauss = self.generator.normal(
                self.generator.normal(g_best, np.abs(sigma))
            )
            pos_new = (
                    gauss
                    + self.generator.uniform() * g_best
                    - self.generator.uniform() * Xp[idx]
            )
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)
        ## Host cells infection
        x_mean = self.calculate_xmean__(pop)
        sigma = self.sigma * (1 - epoch / self.epoch)
        for idx in range(0, self.pop_size):
            ## Basic / simple version, not the original version in the paper
            pos_new = x_mean + sigma * self.generator.normal(0, 1, self.problem.n_dims)
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)
        ## Calculate the weighted mean of the λ best individuals by
        pop = pop.take(self.sorted_order(pop))
        self.pop = pop
        Xp = pop.X
        ## Immune response
        for idx in range(0, self.pop_size):
            pr = (self.problem.n_dims - idx + 1) / self.problem.n_dims
            id1, id2 = self.generator.choice(
                list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
            )
            temp = (
                    Xp[id1]
                    - (Xp[id2] - Xp[idx])
                    * self.generator.uniform()
            )
            condition = self.generator.random(self.problem.n_dims) < pr
            pos_new = np.where(condition, Xp[idx], temp)
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

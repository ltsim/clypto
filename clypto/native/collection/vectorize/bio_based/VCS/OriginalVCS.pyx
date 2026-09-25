#!/usr/bin/env python
# Created by "Thieu" at 22:07, 11/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.bio_based.VCS.DevVCS cimport DevVCS
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


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
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        ## Viruses diffusion
        for idx in range(0, self.pop_size):
            sigma = (np.log1p(epoch) / self.epoch) * (
                    Xp[idx] - g_best
            )
            gauss = np.array(
                [
                    self.generator.normal(g_best[j], np.abs(sigma[j]))
                    for j in range(0, self.problem.n_dims)
                ]
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
        ## Immune response
        for idx in range(0, self.pop_size):
            pr = (self.problem.n_dims - idx + 1) / self.problem.n_dims
            pos_new = Xp[idx].copy()
            for j in range(0, self.problem.n_dims):
                if self.generator.uniform() > pr:
                    id1, id2 = self.generator.choice(
                        list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
                    )
                    pos_new[j] = (
                            Xp[id1][j]
                            - (Xp[id2][j] - Xp[idx][j])
                            * self.generator.uniform()
                    )
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

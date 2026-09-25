#!/usr/bin/env python
# Created by "Thieu" at 07:02, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import math
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalNRO(LegacyNativeOptimizer):
    """
    The original version of: Nuclear Reaction Optimization (NRO)

    Links:
        1. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8720256

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import NRO    >>> import numpy as np
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
    >>> model = NRO.OriginalNRO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wei, Z., Huang, C., Wang, X., Han, T. and Li, Y., 2019. Nuclear reaction optimization: A novel and
    powerful physics-based algorithm for global optimization. IEEE Access, 7, pp.66084-66109.
    [2] Wei, Z.L., Zhang, Z.R., Huang, C.Q., Han, B., Tang, S.Q. and Wang, L., 2019, June. An Approach
    Inspired from Nuclear Reaction Processes for Numerical Optimization. In Journal of Physics:
    Conference Series (Vol. 1213, No. 3, p. 032009). IOP Publishing.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
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
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef object amend_solution(self, object solution):
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        return np.where(condition, solution, rand_pos)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        xichma_v = 1
        xichma_u = (
                           (math.gamma(1 + 1.5) * math.sin(math.pi * 1.5 / 2))
                           / (math.gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))
                   ) ** (1.0 / 1.5)
        levy_b = (self.generator.normal(0, xichma_u)) / (
                np.sqrt(np.abs(self.generator.normal(0, xichma_v))) ** (1.0 / 1.5)
        )
        # NFi phase
        Pb = self.generator.uniform()
        Pfi = self.generator.uniform()
        freq = 0.05
        alpha = 0.01
        pop_new = []
        for idx in range(self.pop_size):
            ## Calculate neutron vector Nei by Eq. (2)
            ## Random 1 more index to select neutron
            temp1 = list(set(range(0, self.pop_size)) - {idx})
            i1 = self.generator.choice(temp1, replace=False)
            Nei = (Xp[idx] + Xp[i1]) / 2
            ## Update population of fission products according to Eq.(3), (6) or (9);
            if self.generator.uniform() <= Pfi:
                ### Update based on Eq. 3
                if self.generator.uniform() <= Pb:
                    xichma1 = (np.log(epoch) * 1.0 / epoch) * np.abs(
                        np.subtract(Xp[idx], g_best)
                    )
                    gauss = np.array(
                        [
                            self.generator.normal(g_best[j], xichma1[j])
                            for j in range(self.problem.n_dims)
                        ]
                    )
                    Xi = (
                            gauss
                            + self.generator.uniform() * g_best
                            - round(self.generator.random() + 1) * Nei
                    )
                ### Update based on Eq. 6
                else:
                    i2 = self.generator.choice(temp1, replace=False)
                    xichma2 = (np.log(epoch) * 1.0 / epoch) * np.abs(
                        np.subtract(Xp[i2], g_best)
                    )
                    gauss = np.array(
                        [
                            self.generator.normal(Xp[idx][j], xichma2[j])
                            for j in range(self.problem.n_dims)
                        ]
                    )
                    Xi = (
                            gauss
                            + self.generator.uniform() * g_best
                            - round(self.generator.random() + 2) * Nei
                    )
            ## Update based on Eq. 9
            else:
                i3 = self.generator.choice(temp1, replace=False)
                xichma2 = (np.log(epoch) * 1.0 / epoch) * np.abs(
                    np.subtract(Xp[i3], g_best)
                )
                Xi = np.array(
                    [
                        self.generator.normal(Xp[idx][j], xichma2[j])
                        for j in range(self.problem.n_dims)
                    ]
                )
            ## Check the boundary and evaluate the fitness function
            ops.commit(self, pop, cand, idx, self.correct_solution(Xi), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

        # NFu phase
        ## Ionization stage
        ## Calculate the Pa through Eq. (10)
        pop_child = []
        ranked_pop = np.argsort(
            [pop.F[i] for i in range(self.pop_size)]
        )
        for idx in range(self.pop_size):
            X_ion = Xp[idx].copy()
            if (ranked_pop[idx] * 1.0 / self.pop_size) < self.generator.random():
                i1, i2 = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
                )
                for j in range(self.problem.n_dims):
                    #### Levy flight strategy is described as Eq. 18
                    if Xp[i2][j] == Xp[idx][j]:
                        X_ion[j] = Xp[idx][j] + alpha * levy_b * (
                                Xp[idx][j] - g_best[j]
                        )
                    #### If not, based on Eq. 11, 12
                    else:
                        if self.generator.uniform() <= 0.5:
                            X_ion[j] = Xp[i1][
                                           j
                                       ] + self.generator.uniform() * (
                                               Xp[i2][j] - Xp[idx][j]
                                       )
                        else:
                            X_ion[j] = Xp[i1][
                                           j
                                       ] - self.generator.uniform() * (
                                               Xp[i2][j] - Xp[idx][j]
                                       )
            else:  #### Levy flight strategy is described as Eq. 21
                worst = [pop.agent(self.sorted_order(pop)[pop.n - 1])]
                X_worst = worst[0]
                for j in range(self.problem.n_dims):
                    ##### Based on Eq. 21
                    if X_worst.solution[j] == g_best[j]:
                        X_ion[j] = Xp[idx][j] + alpha * levy_b * (
                                self.problem.ub[j] - self.problem.lb[j]
                        )
                    ##### Based on Eq. 13
                    else:
                        X_ion[j] = Xp[idx][j] + round(
                            self.generator.uniform()
                        ) * self.generator.uniform() * (
                                           X_worst.solution[j] - g_best[j]
                                   )
            ## Check the boundary and evaluate the fitness function for X_ion
            ops.commit(self, pop, cand, idx, self.correct_solution(X_ion), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

        ## Fusion Stage
        ### all ions obtained from ionization are ranked based on (14) - Calculate the Pc through Eq. (14)
        pop_new = []
        ranked_pop = np.argsort(
            [pop.F[i] for i in range(self.pop_size)]
        )
        for idx in range(self.pop_size):
            i1, i2 = self.generator.choice(
                list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
            )
            #### Generate fusion nucleus
            if (ranked_pop[idx] * 1.0 / self.pop_size) < self.generator.random():
                t1 = self.generator.uniform() * (
                        Xp[i1] - g_best
                )
                t2 = self.generator.uniform() * (
                        Xp[i2] - g_best
                )
                temp2 = Xp[i1] - Xp[i2]
                X_fu = (
                        Xp[idx]
                        + t1
                        + t2
                        - np.exp(-np.linalg.norm(temp2)) * temp2
                )
            #### Else
            else:
                ##### Based on Eq. 22
                if np.allclose(Xp[i1], Xp[i2]):
                    X_fu = Xp[idx] + alpha * levy_b * (
                            Xp[idx] - g_best
                    )
                ##### Based on Eq. 16, 17
                else:
                    if self.generator.uniform() > 0.5:
                        X_fu = Xp[idx] - 0.5 * (
                                np.sin(2 * np.pi * freq * epoch + np.pi)
                                * (self.epoch - epoch)
                                / self.epoch
                                + 1
                        ) * (Xp[i1] - Xp[i2])
                    else:
                        X_fu = Xp[idx] - 0.5 * (
                                np.sin(2 * np.pi * freq * epoch + np.pi)
                                * epoch
                                / self.epoch
                                + 1
                        ) * (Xp[i1] - Xp[i2])
            ops.commit(self, pop, cand, idx, self.correct_solution(X_fu), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

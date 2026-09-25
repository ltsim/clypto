#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation



cdef class AugmentedAEO(LegacyNativeOptimizer):
    """
    The original version of: Augmented Artificial Ecosystem Optimization (AAEO)

    Notes:
        + Used linear weight factor reduce from 2 to 0 through time
        + Applied Levy-flight technique and the global best solution

    Examples
    ~~~~~~~~
    >>> from clypto.collection.system_based import AEO    >>> import numpy as np
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
    >>> model = AEO.AugmentedAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Van Thieu, N., Barma, S. D., Van Lam, T., Kisi, O., & Mahesha, A. (2022). Groundwater level modeling
    using Augmented Artificial Ecosystem Optimization. Journal of Hydrology, 129034.
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
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void evolve(self, int epoch):
        # Agents read the population updated so far (worst agent, random members): in sequential
        # mode the loops run on the buffer rows; swarm/parallel modes batch the evaluation.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        ## Production - Update the worst agent
        # Eq. 2, 3, 1
        wf = 2 * (1 - epoch / self.epoch)  # Weight factor
        a = (1.0 - epoch / self.epoch) * self.generator.random()
        x1 = (1 - a) * Xp[n - 1] + a * self.generator.uniform(
            self.problem.lb, self.problem.ub
        )
        pos_new = self.correct_solution(x1)
        ops.set_row(pop, n - 1, pos_new, self.get_target(pos_new))
        ## Consumption - Update the whole population left
        for idx in range(0, n - 1):
            if self.generator.random() < 0.5:
                rand = self.generator.random()
                # Eq. 4, 5, 6
                c = (
                        0.5
                        * self.generator.normal(0, 1)
                        / np.abs(self.generator.normal(0, 1))
                )  # Consumption factor
                j = 1 if idx == 0 else self.generator.integers(0, idx)
                ### Herbivore
                if rand < 1.0 / 3:
                    pos_new = Xp[idx] + wf * c * (
                            Xp[idx] - Xp[0]
                    )  # Eq. 6
                ### Omnivore
                elif 1.0 / 3 <= rand <= 2.0 / 3:
                    pos_new = Xp[idx] + wf * c * (
                            Xp[idx] - Xp[j]
                    )  # Eq. 7
                ### Carnivore
                else:
                    r2 = self.generator.uniform()
                    pos_new = Xp[idx] + wf * c * (
                            r2 * (Xp[idx] - Xp[0])
                            + (1 - r2) * (Xp[idx] - Xp[j])
                    )
            else:
                pos_new = Xp[idx] + self.get_levy_flight_step(
                    1.0, 0.001, case=-1
                ) * (1.0 / np.sqrt(epoch)) * np.sign(self.generator.random() - 0.5) * (
                                  Xp[idx] - g_best
                          )
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n - 1)
        ## find current best used in decomposition
        best = Xp[self.sorted_order(pop)[0]].copy()
        cand = pop.empty_like()
        for idx in range(0, n):
            if self.generator.random() < 0.5:
                pos_new = best + self.generator.normal(
                    0, 1, self.problem.n_dims
                ) * (best - Xp[idx])
            else:
                beta = self.generator.uniform(0.01, 1.0)
                pos_new = best + self.get_levy_flight_step(
                    beta=beta, multiplier=0.01, size=self.problem.n_dims, case=0
                ) * (best - Xp[idx])
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

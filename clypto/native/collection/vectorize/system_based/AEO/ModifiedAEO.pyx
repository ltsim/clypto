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



cdef class ModifiedAEO(LegacyNativeOptimizer):
    """
    The original version of: Modified Artificial Ecosystem-Based Optimization (MAEO)

    Links:
        1. https://doi.org/10.1109/ACCESS.2020.2973351

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
    >>> model = AEO.ModifiedAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Menesy, A.S., Sultan, H.M., Korashy, A., Banakhr, F.A., Ashmawy, M.G. and Kamel, S., 2020. Effective
    parameter extraction of different polymer electrolyte membrane fuel cell stack models using a
    modified artificial ecosystem optimization algorithm. IEEE Access, 8, pp.31892-31909.
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
        ## Production
        # Eq. 22
        H = 2 * (1 - epoch / self.epoch)
        a = (1 - epoch / self.epoch) * self.generator.random()
        x1 = (1 - a) * Xp[n - 1] + a * self.generator.uniform(
            self.problem.lb, self.problem.ub
        )
        pos_new = self.correct_solution(x1)
        ops.set_row(pop, n - 1, pos_new, self.get_target(pos_new))
        ## Consumption - Update the whole population left
        for idx in range(0, n - 1):
            rand = self.generator.random()
            # Eq. 4, 5, 6
            v1 = self.generator.normal(0, 1)
            v2 = self.generator.normal(0, 1)
            c = 0.5 * v1 / abs(v2)  # Consumption factor
            j = 1 if idx == 0 else self.generator.integers(0, idx)
            ### Herbivore
            if rand <= 1.0 / 3:  # Eq. 23
                pos_new = Xp[idx] + H * c * (
                        Xp[idx] - Xp[0]
                )
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:  # Eq. 24
                pos_new = Xp[idx] + H * c * (
                        Xp[idx] - Xp[j]
                )
            ### Omnivore
            else:  # Eq. 25
                r5 = self.generator.random()
                pos_new = Xp[idx] + H * c * (
                        r5 * (Xp[idx] - Xp[0])
                        + (1 - r5) * (Xp[idx] - Xp[j])
                )
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n - 1)
        ## find current best used in decomposition
        best = Xp[self.sorted_order(pop)[0]].copy()
        cand = pop.empty_like()
        for idx in range(0, n):
            r3 = self.generator.uniform()
            d = 3 * self.generator.normal(0, 1)
            e = r3 * self.generator.integers(1, 3) - 1
            h = 2 * r3 - 1
            if self.generator.random() < 0.5:
                beta = 1 - (1 - 0) * (epoch / self.epoch)  # Eq. 21
                r_idx = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx})
                )
                x_r = Xp[r_idx]
                if self.generator.random() < 0.5:
                    x_new = beta * x_r + (1 - beta) * Xp[idx]
                else:
                    x_new = (1 - beta) * x_r + beta * Xp[idx]
            else:
                x_new = best + d * (
                        e * best - h * Xp[idx]
                )
            ops.commit(self, pop, cand, idx, self.correct_solution(x_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

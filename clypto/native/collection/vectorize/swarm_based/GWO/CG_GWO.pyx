#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class CG_GWO(LegacyNativeOptimizer):
    """
    The original version of: Cauchy‑Gaussian mutation and improved search strategy GWO (CG‑GWO)

    Notes:
        + This algorithm can't be parallelized because of the 'single' update mode.
        + Meaning that the updating of the pack is based on order and sequence of the wolves.

    Links:
        1. https://doi.org/10.1038/s41598-022-23713-9

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.CG_GWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Li, K., Li, S., Huang, Z. et al. Grey Wolf Optimization algorithm based on Cauchy-Gaussian mutation and improved search strategy. Sci Rep 12, 18961 (2022).
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
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def cauchy_gaussian_mutation(self, best_fit, leader_fit, leader_pos, epoch):
        # Calculate dynamic parameters (equations 11 and 12)
        eps2 = (epoch / self.epoch) ** 2
        eps1 = 1 - eps2

        # Calculate sigma (equation 9)
        if abs(best_fit) > 1e-10:
            sigma = np.exp((leader_fit - best_fit) / abs(best_fit))
        else:
            sigma = 1.0
        # Generate Cauchy and Gaussian random variables
        c_rand = self.generator.standard_cauchy(size=self.problem.n_dims) * sigma**2 + 0
        g_rand = self.generator.normal(loc=0, scale=sigma**2, size=self.problem.n_dims)

        # Apply mutation (equation 8)
        return leader_pos * (1 + eps1 * c_rand + eps2 * g_rand)

    cdef void evolve(self, int epoch):
        # Agents read the population they just updated (random wolf, mean position),
        # so the loop stays sequential on the buffer rows.
        cdef NativePopulation pop = self.pop
        cdef NativeTarget tar
        cdef Py_ssize_t idx, k, n = pop.n
        minmax = self.problem.minmax
        lb, ub = self.problem.lb, self.problem.ub
        # linearly decreased from 2 to 0
        a = 2 - 2.0 * epoch / self.epoch
        X = pop.X
        order = self.sorted_order(pop)[:3]
        best_pos = [X[i].copy() for i in order]
        best_fit = [float(pop.F[i]) for i in order]

        # Apply Cauchy-Gaussian mutation to leaders, then greedy selection
        leaders = []
        for k in range(3):
            pos = self.correct_solution(self.cauchy_gaussian_mutation(best_fit[0], best_fit[k], best_pos[k], epoch))
            leaders.append((pos, self.get_target(pos)))
        for k in range(3):
            pos, tar = leaders[k]
            if (tar.fitness < best_fit[k]) if minmax == "min" else (tar.fitness > best_fit[k]):
                best_pos[k], best_fit[k] = pos, tar.fitness

        for idx in range(n):
            # Apply improved search strategy (equation 13)
            r1, r2, r3, r4, r5 = self.generator.random(5)
            if r5 >= 0.5:  # Exploration around random wolf
                jdx = self.generator.choice(list(set(range(self.pop_size)) - {idx}))
                x_rand = X[jdx]
                pos_new = x_rand - r1 * np.abs(x_rand - 2 * r2 * X[idx])
            else:  # Exploration around alpha wolf
                x_avg = np.mean(np.ascontiguousarray(X), axis=0)
                pos_new = (best_pos[0] - x_avg) - r3 * (lb + r4 * (ub - lb))
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)

            if self.compare_fitness(pop.F[idx], tar.fitness, minmax):
                # If new position is not better, use original GWO update
                A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C1 = 2 * self.generator.random(self.problem.n_dims)
                C2 = 2 * self.generator.random(self.problem.n_dims)
                C3 = 2 * self.generator.random(self.problem.n_dims)
                X1 = best_pos[0] - A1 * np.abs(C1 * best_pos[0] - X[idx])
                X2 = best_pos[1] - A2 * np.abs(C2 * best_pos[1] - X[idx])
                X3 = best_pos[2] - A3 * np.abs(C3 * best_pos[2] - X[idx])
                pos_new = (X1 + X2 + X3) / 3.0
                pos_new = self.correct_solution(pos_new)
                tar = self.get_target(pos_new)

            if self.compare_fitness(tar.fitness, pop.F[idx], minmax):
                # If new position is better, update the agent
                ops.set_row(pop, idx, pos_new, tar)

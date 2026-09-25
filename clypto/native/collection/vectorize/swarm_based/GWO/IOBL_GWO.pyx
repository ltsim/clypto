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


cdef class IOBL_GWO(LegacyNativeOptimizer):
    """
    The original version of: Improved Opposite-based Learning Grey Wolf Optimizer (IOBL-GWO)

    Notes:
        + In the paper, they called it "Improved Grey Wolf Optimizer (IGWO)", but there are many improved versions of GWO.
        + So based on their proposed equations, we called it as "Improved Opposite-based Learning Grey Wolf Optimizer (IOBL-GWO)".
        + This algorithm is heavily (4x - 6X slower than original) because of multiple times of calculating the fitness of agent in each population.

    Links:
        1. https://doi.org/10.1007/s12652-020-02153-1

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
    >>> model = GWO.IOBL_GWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Bansal, J. C., & Singh, S. (2021). A better exploration strategy in Grey Wolf Optimizer. Journal of Ambient Intelligence and Humanized Computing, 12(1), 1099-1118.
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

    cdef void evolve(self, int epoch):
        # Every agent is compared and replaced before the next one moves (it reads the
        # population it just updated), so the loop stays sequential on the buffer rows.
        cdef NativePopulation pop = self.pop
        cdef NativeTarget tar_new
        cdef Py_ssize_t idx, k, n = pop.n
        minmax = self.problem.minmax
        lb, ub = self.problem.lb, self.problem.ub
        # linearly decreased from 2 to 0
        a = 2 - 2.0 * epoch / self.epoch
        best = pop.X[self.sorted_order(pop)[:3]]
        X = pop.X
        for idx in range(n):
            # Try explorative equation first
            r1, r2, r3, r4, r5 = self.generator.random(5)
            if r5 >= 0.5:  # Exploration around random wolf
                # Select random wolf from population
                jdx = self.generator.choice(list(set(range(self.pop_size)) - {idx}))
                x_rand = X[jdx]
                pos_new = x_rand - r1 * np.abs(x_rand - 2 * r2 * X[idx])
            else:  # Exploration around alpha wolf
                # Calculate average position of all wolves
                x_avg = np.mean(np.ascontiguousarray(X), axis=0)
                pos_new = (best[0] - x_avg) - r3 * (lb + r4 * (ub - lb))
            # Apply boundary constraints
            pos_new = self.correct_solution(pos_new)
            tar_new = self.get_target(pos_new)
            if self.compare_fitness(tar_new.fitness, pop.F[idx], minmax):
                # If new position is better, update the agent
                ops.set_row(pop, idx, pos_new, tar_new)
            else:
                # If not better, use original GWO update
                A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C1 = 2 * self.generator.random(self.problem.n_dims)
                C2 = 2 * self.generator.random(self.problem.n_dims)
                C3 = 2 * self.generator.random(self.problem.n_dims)
                X1 = best[0] - A1 * np.abs(C1 * best[0] - X[idx])
                X2 = best[1] - A2 * np.abs(C2 * best[1] - X[idx])
                X3 = best[2] - A3 * np.abs(C3 * best[2] - X[idx])
                pos_new = (X1 + X2 + X3) / 3.0
                pos_new = self.correct_solution(pos_new)
                tar_new = self.get_target(pos_new)
                if self.compare_fitness(tar_new.fitness, pop.F[idx], minmax):
                    ops.set_row(pop, idx, pos_new, tar_new)

        # Apply Opposition-Based Learning (OBL) for leading wolves
        order = self.sorted_order(pop)
        obl = []
        for k in range(3):
            pos_obl = lb + ub - X[order[k]]
            obl.append((pos_obl, self.get_target(pos_obl)))
        # Replace worst 3 wolves with opposite solutions if they are better
        for k in range(3):
            if self.compare_fitness(obl[k][1].fitness, pop.F[order[-3 + k]], minmax):
                ops.set_row(pop, k, obl[k][0], obl[k][1])

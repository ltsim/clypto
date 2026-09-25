#!/usr/bin/env python
# Created by "Thieu" at 13:42, 06/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalSeaHO(LegacyNativeOptimizer):
    """
    The original version of: Sea-Horse Optimization (SeaHO)

    Links:
        1. https://link.springer.com/article/10.1007/s10489-022-03994-3
        2. https://www.mathworks.com/matlabcentral/fileexchange/115945-sea-horse-optimizer

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import SeaHO    >>> import numpy as np
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
    >>> model = SeaHO.OriginalSeaHO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, S., Zhang, T., Ma, S., & Wang, M. (2022). Sea-horse optimizer: a novel nature-inspired
    meta-heuristic for global optimization problems. Applied Intelligence, 1-28.
    """

    cdef public object uu
    cdef public object vv
    cdef public object ll

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

    cdef void initialize_variables(self):
        self.uu = 0.05
        self.vv = 0.05
        self.ll = 0.05

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d, half = int(self.pop_size / 2)
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        # The motor behavior of sea horses
        step_length = self.get_levy_flight_step(
            beta=1.5,
            multiplier=0.01,
            size=(self.pop_size, self.problem.n_dims),
            case=-1,
        )
        moved = np.empty((n, d))
        for idx in range(0, self.pop_size):
            beta = self.generator.normal(0, 1, self.problem.n_dims)
            theta = 2 * np.pi * self.generator.random(self.problem.n_dims)
            row = self.uu * np.exp(theta * self.vv)
            xx, yy, zz = row * np.cos(theta), row * np.sin(theta), row * theta
            if self.generator.normal(0, 1) > 0:  # Eq. 4
                pos_new = Xp[idx] + step_length[idx] * ((g_best - Xp[idx]) * xx * yy * zz + g_best)
            else:  # Eq. 7
                pos_new = Xp[idx] + self.generator.random(self.problem.n_dims) * self.ll * beta * (g_best - beta * g_best)
            moved[idx] = self.correct_solution(pos_new)

        # The predation behavior of sea horses
        alpha = (1 - epoch / self.epoch) ** (2 * epoch / self.epoch)
        Xc = cand.X
        for idx in range(0, self.pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            if self.generator.random() >= 0.1:
                pos_new = alpha * (g_best - r1 * moved[idx]) + (1 - alpha) * g_best  # Eq. 10
            else:
                pos_new = (1 - alpha) * (moved[idx] - r1 * g_best) + alpha * moved[idx]  # Eq. 11
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        child = cand.take(self.sorted_order(cand))  # Sorted population

        # The reproductive behavior of sea horses
        offspring = cand.take(np.zeros(half, dtype=int))
        Xd, Xo = child.X, offspring.X
        for kdx in range(0, half):
            r3 = self.generator.random()
            pos_new = r3 * Xd[kdx] + (1 - r3) * Xd[half + kdx]  # Eq. 13
            Xo[kdx] = self.correct_solution(pos_new)
        self.evaluate(offspring, 0, half)
        # Sea horses selection
        both = child.concat(offspring)
        self.pop = both.take(self.sorted_order(both)[:n])

#!/usr/bin/env python
# Created by "Thieu" at 10:08, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class SwarmHC(LegacyNativeOptimizer):
    """
    The developed version: Swarm-based Hill Climbing (S-HC)

    Notes
    ~~~~~
    + Based on swarm-of people are trying to climb on the mountain idea
    + The number of neighbour solutions are equal to population size
    + The step size to calculate neighbour is randomized and based on rank of solution.
        + The guys near on top of mountain will move slower than the guys on bottom of mountain.
        + Imagination: exploration when far from global best, and exploitation when near global best
    + Who on top of mountain first will be the winner. (global optimal)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + neighbour_size (int): [2, pop_size/2], fixed parameter, sensitive exploitation parameter, Default: 10

    Examples
    ~~~~~~~~
    >>> from clypto.collection.math_based import HC    >>> import numpy as np
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
    >>> model = HC.SwarmHC(epoch=1000, pop_size=50, neighbour_size = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    cdef public int neighbour_size

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        neighbour_size = 10,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 10
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "neighbour_size"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.neighbour_size = cy.validator(int, neighbour_size, [2, int(self.pop_size / 2)], "neighbour_size")

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation best = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, k = self.neighbour_size, d = pop.d
        ranks = np.array(list(range(1, self.pop_size + 1)))
        ranks = ranks / np.sum(ranks)
        step_size = np.mean(self.problem.ub - self.problem.lb) * np.exp(-2 * epoch / self.epoch)
        ss = step_size * ranks
        # per agent, k neighbours: one normal(0, 1, d) draw each
        N = self.generator.normal(0, 1, (n, k, d))
        pos = pop.X[:, None, :] + N * ss[:, None, None]
        pos = self.correct_solution(pos.reshape(n * k, d))
        R = self.evaluate_rows(pos)
        F = self._fitness(R)
        for idx in range(n):
            # best neighbour of agent idx (the classic argsort tie rule)
            order = np.argsort(F[idx * k:(idx + 1) * k])
            j = idx * k + (order[::-1][0] if self.problem.minmax == "max" else order[0])
            best.X[idx] = pos[j]
            best.O[idx] = R[j]
            best.F[idx] = F[j]
        ops.accept(self, best)

    def evaluate_rows(self, X):
        return self._objectives(X)

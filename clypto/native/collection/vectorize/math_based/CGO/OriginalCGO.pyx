#!/usr/bin/env python
# Created by "Thieu" at 22:24, 02/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalCGO(LegacyNativeOptimizer):
    """
    The original version of: Chaos Game Optimization (CGO)

    Links:
        1. https://doi.org/10.1007/s10462-020-09867-w

    Notes:
        + 4th seed is mutation process, but it is not clear mutation on multiple variables or 1 variable
        + There is no usage of the variable alpha 4th in the paper
        + The replacement of the worst solutions by generated seed are not clear (Lots of grammar errors in this section)

    Examples
    ~~~~~~~~

    >>> from clypto.collection.math_based import CGO    >>> import numpy as np
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
    >>> model = CGO.OriginalCGO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Talatahari, S. and Azizi, M., 2021. Chaos Game Optimization: a novel metaheuristic algorithm.
    Artificial Intelligence Review, 54(2), pp.917-1004.
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

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation seeds
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        me = np.arange(n)
        # three distinct random agents per agent -> their mean group MG
        s = np.argpartition(rng.random((n, n)), 2, axis=1)[:, :3]
        MG = X[s].mean(axis=1)
        alpha = np.empty((n, 4, 1))
        alpha[:, 0] = rng.random((n, 1))
        alpha[:, 1] = 2 * rng.random((n, 1))
        alpha[:, 2] = 1 + rng.random((n, 1)) * rng.random((n, 1))
        esp = rng.random((n, 1))
        alpha[:, 3] = esp + esp * rng.random((n, 1))
        beta = rng.integers(0, 2, size=(n, 3, 1))
        gama = rng.integers(0, 2, size=(n, 3, 1))
        # seed4: k random dimensions receive a uniform(0, 1) kick
        k = rng.integers(0, d, size=n)
        picked = rng.random((n, d)).argsort(axis=1).argsort(axis=1) < k[:, None]
        cand_pos = np.empty((n, 4, d))
        cand_pos[:, 0] = X + alpha[:, 0] * (beta[:, 0] * g - gama[:, 0] * MG)  # Eq. 3
        cand_pos[:, 1] = g + alpha[:, 1] * (beta[:, 1] * X - gama[:, 1] * MG)  # Eq. 4
        cand_pos[:, 2] = MG + alpha[:, 2] * (beta[:, 2] * X - gama[:, 2] * g)  # Eq. 5
        cand_pos[:, 3] = X + picked * rng.uniform(0, 1, (n, d))
        seeds = pop.take(np.repeat(me, 4))
        seeds.X[:] = self.correct_solution(cand_pos.reshape(4 * n, d))
        self.evaluate(seeds, 0, 4 * n)
        F = np.asarray(seeds.F).reshape(n, 4)
        best = (F.argmin(axis=1) if self.problem.minmax == "min" else F.argmax(axis=1))
        rows = 4 * me + best
        win = np.flatnonzero(ops.better(self, F[me, best], np.asarray(pop.F)))
        pop.buf[win] = seeds.buf[rows[win]]

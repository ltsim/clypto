#!/usr/bin/env python
# Created by "Thieu" at 23:58, 03/09/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalDOA(LegacyNativeOptimizer):
    """
    The original version of: Dream Optimization Algorithm (DOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/178419-dream-optimization-algorithm-doa

    Notes:
        1. The Matlab code is sloppy and incorrect. Many variables are defined and computed but never actually used
        in the solution update process. For example, the variable fitness is calculated during the
        exploitation phase but not applied.

        2. The agent’s position is also not updated properly, meaning it remains unchanged even after
        the supposed update in the Matlab code.

        3. I suspect the results reported in this paper might not exist at all but were fabricated by
        the authors, since the benchmark functions are completely missing from the Matlab code.

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import DOA    >>> import numpy as np
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
    >>> model = DOA.OriginalDOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Lang, Y., & Gao, Y. (2025). Dream Optimization Algorithm (DOA): A novel metaheuristic optimization
    algorithm inspired by human dreams and its applications to real-world engineering problems.
    Computer Methods in Applied Mechanics and Engineering, 436, 117718.
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

    def pick_dims__(self, n, d, k):
        """Boolean mask (n, d): ``k`` (scalar or per-row) random dimensions of every row are True."""
        ranks = self.generator.random((n, d)).argsort(axis=1).argsort(axis=1)
        return ranks < np.reshape(k, (-1, 1))

    def other_values__(self, X, n, d):
        """For every (agent, dimension): the value of that dimension in a random other agent."""
        return X[ops.others(self, n, d), np.arange(d)[None, :]]

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = np.array(pop.X)
        lb, ub = self.problem.lb, self.problem.ub
        F = np.asarray(pop.F)
        exploration_end = int(9 * self.epoch / 10)
        if epoch_c <= exploration_end:
            starts = np.array([int((m / 5) * n) for m in range(5)])
            group = np.searchsorted(starts, np.arange(n), side="right") - 1
            kk = np.empty(5, dtype=int)
            pbest = np.empty((5, d))
            for m in range(5):
                aa = max(1, np.ceil(d / 8 / (m + 1)))
                bb = np.ceil(d / 3 / (m + 1)) + 1
                kk[m] = rng.integers(aa, bb)
                s, e = starts[m], int(((m + 1) / 5) * n)
                sub = F[s:e]
                pbest[m] = X[s + (sub.argmin() if self.problem.minmax == "min" else sub.argmax())]
            base = pbest[group]
            cos_term = (np.cos((epoch_c + self.epoch / 10) * np.pi / self.epoch) + 1) / 2
            move = base + (rng.random((n, d)) * (ub - lb) + lb) * cos_term
            out = (move > ub) | (move < lb)
            fill = self.other_values__(X, n, d) if d > 15 else rng.random((n, d)) * (ub - lb) + lb
            move = np.where(out, fill, move)
            val = np.where((rng.random(n) < 0.9)[:, None], move, self.other_values__(X, n, d))
            pos = np.where(self.pick_dims__(n, d, kk[group]), val, base)
        else:  # Exploitation phase (last 10% of iterations)
            g = np.array(self.g_best_x())
            km = max(2, int(np.ceil(d / 3)))
            k = rng.integers(2, km + 1, size=n)
            cos_term = (np.cos(epoch_c * np.pi / self.epoch) + 1) / 2
            move = g + (rng.random((n, d)) * (ub - lb) + lb) * cos_term
            out = (move > ub) | (move < lb)
            fill = self.other_values__(X, n, d) if d > 15 else rng.random((n, d)) * (ub - lb) + lb
            pos = np.where(self.pick_dims__(n, d, k), np.where(out, fill, move), g)
        ops.replace(self, pos)

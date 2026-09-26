#!/usr/bin/env python
# Created by "Thieu" at 23:50, 28/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalEAO(LegacyNativeOptimizer):
    """
    The original version of: Enzyme Action Optimizer (EAO)

    Notes:
        + This algorithm used 3 fitness calculations for each update enzyme. Therefor, it is slower 3 times than other algorithms.

    Links:
        1. https://mathworks.com/matlabcentral/fileexchange/170296-enzyme-action-optimizer-a-novel-bio-inspired-optimization

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import EAO    >>> import numpy as np
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
    >>> model = EAO.OriginalEAO(epoch=1000, pop_size=50, p_m=0.01, n_elites=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rodan, A., Al-Tamimi, A. K., Al-Alnemer, L., Mirjalili, S., & Tiňo, P. (2025).
    Enzyme action optimizer: a novel bio-inspired optimization algorithm. The Journal of Supercomputing, 81(5), 686.
    """

    cdef public object ec

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        ec: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 100
            ec: Enzyme Concentration, default=0.1
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "ec"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.ec = cy.validator(float, ec, [0.0, 100], "ec")

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        AF = np.sqrt(epoch_c / self.epoch)
        ec = self.ec
        j1, j2 = ops.two_others(self, n, 1)
        j1, j2 = j1[:, 0], j2[:, 0]
        pos1 = (g - X) + rng.random((n, d)) * np.sin(AF * X)
        scA = ec + (1 - ec) * rng.random((n, d))
        exA = AF * (ec + (1 - ec) * rng.random((n, d)))
        posA = X + scA * (X[j1] - X[j2]) + exA * (g - X)
        scB = ec + (1 - ec) * rng.random((n, 1))
        exB = AF * (ec + (1 - ec) * rng.random((n, 1)))
        posB = X + scB * (X[j1] - X[j2]) + exB * (g - X)
        # three candidates per agent, one batch; the best of them replaces the agent when it is better
        cand = pop.take(np.repeat(np.arange(n), 3))
        cand.X[:] = self.correct_solution(np.stack([pos1, posA, posB], axis=1).reshape(3 * n, d))
        self.evaluate(cand, 0, 3 * n)
        F = np.asarray(cand.F).reshape(n, 3)
        best = F.argmin(axis=1) if self.problem.minmax == "min" else F.argmax(axis=1)
        rows = 3 * np.arange(n) + best
        win = np.flatnonzero(ops.better(self, F[np.arange(n), best], np.asarray(pop.F)))
        pop.buf[win] = cand.buf[rows[win]]

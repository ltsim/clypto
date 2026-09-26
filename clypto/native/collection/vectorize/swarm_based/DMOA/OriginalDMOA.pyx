#!/usr/bin/env python
# Created by "Thieu" at 17:48, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalDMOA(VectorizeOptimizer):
    """
    The original version of: Dwarf Mongoose Optimization Algorithm (DMOA)

    Links:
        1. https://doi.org/10.1016/j.cma.2022.114570
        2. https://www.mathworks.com/matlabcentral/fileexchange/105125-dwarf-mongoose-optimization-algorithm

    Notes:
        1. The Matlab code differs slightly from the original paper
        2. There are some parameters and equations in the Matlab code that don't seem to have any meaningful purpose.
        3. The algorithm seems to be weak on solving several problems.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import DMOA    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = DMOA.OriginalDMOA(epoch=1000, pop_size=50, n_baby_sitter = 3, peep = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Agushaka, J. O., Ezugwu, A. E., & Abualigah, L. (2022). Dwarf mongoose optimization algorithm.
    Computer methods in applied mechanics and engineering, 391, 114570.
    """

    cdef public object n_baby_sitter
    cdef public object peep
    cdef public object n_scout
    cdef public object C
    cdef public object tau
    cdef public object L

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_baby_sitter: int = 3,
        peep: float = 2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "n_baby_sitter", "peep"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.n_baby_sitter = cy.validator(int, n_baby_sitter, [2, 10], "n_baby_sitter")
        self.peep = cy.validator(float, peep, [1, 10.0], "peep")
        self.n_scout = self.pop_size - self.n_baby_sitter

    def _initialize_variables(self):
        self.C = np.zeros(self.pop_size)
        self.tau = -np.inf
        self.L = np.round(0.6 * self.problem.n_dims * self.n_baby_sitter)

    def alpha_phase__(self, NativePopulation pop):
        """Alpha group: follow a leader chosen by roulette wheel on exp(-fitness / mean)."""
        n, d = pop.n, pop.d
        rng = self.generator
        fit = np.array(pop.F)
        alpha = ops.roulette(self, np.exp(-fit / np.mean(fit)), n)
        me = np.arange(n)
        k = ops.exclude(rng.integers(0, n - 2, size=n), np.stack([me, alpha], axis=1))
        phi = (self.peep / 2) * rng.uniform(-1, 1, (n, d))
        X = pop.X
        before = np.array(pop.F)
        ops.step(self, X[alpha] + phi * (X[alpha] - X[k]))
        self.C += ~ops.better(self, np.asarray(self.pop.F), before)

    def scout_phase__(self, NativePopulation pop, eps):
        """Scouts: explore around themselves; returns the sequential-move measure SM."""
        n, d = pop.n, pop.d
        rng = self.generator
        k = ops.others(self, n)[:, 0]
        phi = (self.peep / 2) * rng.uniform(-1, 1, (n, d))
        X = pop.X
        cand = pop.empty_like()
        cand.X[:] = self._correct_solution(X + phi * (X - X[k]))
        self.evaluate(cand, 0, n)
        cf, of = np.asarray(cand.F), np.asarray(pop.F)
        SM = (cf - of) / (np.maximum(cf, of) + eps)
        ok = ops.better(self, cf, of)
        pop.buf[ok] = cand.buf[ok]
        self.C += ~ok
        return SM

    def respawn__(self, NativePopulation pop, rows):
        """The sites of rows that ran out of patience are re-drawn at random."""
        if len(rows):
            fresh = pop.take(rows)
            fresh.X[:] = self.problem.bounds.low + self.generator.random((len(rows), pop.d)) * (self.problem.bounds.up - self.problem.bounds.low)
            self.evaluate(fresh, 0, len(rows))
            pop.buf[rows] = fresh.buf
            self.C[rows] = 0

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        CF = (1.0 - epoch_c / self.epoch) ** (2.0 * epoch_c / self.epoch)
        self.alpha_phase__(pop)
        SM = self.scout_phase__(pop, 0.0)
        pop = self.pop
        rows = np.arange(self.n_baby_sitter)
        self.respawn__(pop, rows[self.C[rows] >= self.L])
        X = pop.X
        new_tau = np.mean(SM)
        phi = (self.peep / 2) * rng.uniform(-1, 1, (n, d))
        sign = np.ones(n)
        sign[0] = -1.0 if new_tau > self.tau else 1.0  # (the classic code updates tau inside the loop)
        sign[1:] = 1.0
        self.tau = new_tau
        M = SM[:, None] * X / X
        ops.replace(self, X + sign[:, None] * CF * phi * rng.random((n, 1)) * (X - M))

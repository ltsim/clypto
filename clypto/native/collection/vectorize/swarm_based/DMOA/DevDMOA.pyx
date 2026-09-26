#!/usr/bin/env python
# Created by "Thieu" at 17:48, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevDMOA(LegacyNativeOptimizer):
    """
    The developed version of: Dwarf Mongoose Optimization Algorithm (DMOA)

    Notes:
        1. Removed the parameter n_baby_sitter
        2. Changed in section # Next Mongoose position
        3. Removed the meaningless variable tau

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import DMOA    >>> import numpy as np
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
    >>> model = DMOA.DevDMOA(epoch=1000, pop_size=50, peep = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    cdef public object peep
    cdef public object C
    cdef public object L

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        peep: float = 2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "peep"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.peep = cy.validator(float, peep, [1, 10.0], "peep")

    cdef void initialize_variables(self):
        self.C = np.zeros(self.pop_size)
        self.L = np.round(0.6 * self.epoch)

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
        cand.X[:] = self.correct_solution(X + phi * (X - X[k]))
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
            fresh.X[:] = self.problem.lb + self.generator.random((len(rows), pop.d)) * (self.problem.ub - self.problem.lb)
            self.evaluate(fresh, 0, len(rows))
            pop.buf[rows] = fresh.buf
            self.C[rows] = 0

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        CF = (1.0 - epoch_c / self.epoch) ** (2.0 * epoch_c / self.epoch)
        self.alpha_phase__(pop)
        SM = self.scout_phase__(pop, self.EPSILON)
        pop = self.pop
        self.respawn__(pop, np.flatnonzero(self.C >= self.L))
        g = np.array(self.g_best_x())
        X = pop.X
        new_tau = np.mean(SM)
        phi = (self.peep / 2) * rng.uniform(-1, 1, (n, d))
        step = CF * phi * (g - SM[:, None] * X)
        ops.step(self, np.where((new_tau > SM)[:, None], g - step, X + step))

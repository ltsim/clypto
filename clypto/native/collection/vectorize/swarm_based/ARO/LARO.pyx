#!/usr/bin/env python
# Created by "Thieu" at 22:46, 26/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class LARO(LegacyNativeOptimizer):
    """
    The improved version of:  Lévy flight, and the selective opposition version of the artificial rabbit algorithm (LARO)

    Links:
        1. https://doi.org/10.3390/sym14112282

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import ARO    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = ARO.LARO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, Y., Huang, L., Zhong, J., & Hu, G. (2022). LARO: Opposition-based learning boosted
    artificial rabbits-inspired optimization algorithm with Lévy flight. Symmetry, 14(11), 2282.
    """

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
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

    def random_dims__(self, n, d):
        """0/1 mask (n, d): ceil(u * d) random dimensions of every row are 1."""
        k = np.ceil(self.generator.random(n) * d)
        ranks = self.generator.random((n, d)).argsort(axis=1).argsort(axis=1)
        return (ranks < k[:, None]).astype(float)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        theta = 2 * (1 - (epoch + 1) / self.epoch)
        L = (np.exp(1) - np.exp((epoch / self.epoch) ** 2)) * np.sin(2 * np.pi * rng.random(n))
        R = L[:, None] * self.random_dims__(n, d)  # Eq 2
        A = 2 * np.log(1.0 / rng.random(n)) * theta  # Eq. 15
        # detour foraging strategy, Eq. 1
        rand_agent = X[rng.integers(0, n, size=n)]
        detour = rand_agent + R * (X - rand_agent) + np.round(0.5 * (0.05 + rng.random((n, 1)))) * rng.normal(0, 1, (n, 1))
        # random hiding stage, Eqs. 8, 11, 12, 13
        gr = self.random_dims__(n, d)
        H = rng.normal(0, 1, (n, 1)) * (epoch / self.epoch)
        b = X + H * gr * X
        hiding = X + R * (self.get_levy_flight_step(beta=1.5, multiplier=0.1, size=n, case=-1)[:, None] * b - X)
        ops.step(self, np.where((A > 1)[:, None], detour, hiding))
        # second phase: agents far from the best in most dimensions and with a negative rank correlation jump to the mirror image
        X = np.array(self.pop.X)
        gb_fit = self.current_g_best().target.fitness
        g = np.array(self.g_best_x())
        TS = 2 - (2 * epoch / self.epoch)
        dd = np.abs(g - X)
        far, close = dd < TS, dd > TS
        n_df, n_dc = far.sum(axis=1), close.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            src = 1 - 6 * np.sum(dd ** 2, axis=1) / np.sum(dd * (dd ** 2 - 1), axis=1)
        df_lb = np.where(n_df > 0, np.where(far, dd, np.inf).min(axis=1), dd.min(axis=1))
        df_ub = np.where(n_df > 0, np.where(far, dd, -np.inf).max(axis=1), dd.max(axis=1))
        sel = np.flatnonzero((np.asarray(self.pop.F) != gb_fit) & (src <= 0) & (n_df > n_dc))
        if len(sel):
            cand = self.pop.take(sel)
            cand.X[:] = self.correct_solution((df_lb + df_ub)[sel][:, None] - X[sel])
            self.evaluate(cand, 0, len(sel))
            ops.scatter(self, cand, sel)

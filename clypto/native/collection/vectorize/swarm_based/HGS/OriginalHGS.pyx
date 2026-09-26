#!/usr/bin/env python
# Created by "Thieu" at 15:37, 19/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport _LegacyAgent


from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalHGS(LegacyNativeOptimizer):
    """
    The original version of: Hunger Games Search (HGS)

    Links:
        https://aliasgharheidari.com/HGS.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + PUP (float): [0.01, 0.2], The probability of updating position (L in the paper), default = 0.08
        + LH (float): [1000, 20000], Largest hunger / threshold, default = 10000

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import HGS    >>> import numpy as np
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
    >>> model = HGS.OriginalHGS(epoch=1000, pop_size=50, PUP = 0.08, LH = 10000)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, Y., Chen, H., Heidari, A.A. and Gandomi, A.H., 2021. Hunger games search: Visions, conception, implementation,
    deep analysis, perspectives, and towards performance shifts. Expert Systems with Applications, 177, p.114864.
    """

    cdef public object PUP
    cdef public object LH

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        PUP: float = 0.08,
        LH: float = 10000,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            PUP (float): The probability of updating position (L in the paper), default = 0.08
            LH (float): Largest hunger / threshold, default = 10000
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "PUP", "LH"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.PUP = cy.validator(float, PUP, (0, 1.0), "PUP")
        self.LH = cy.validator(float, LH, [1, 20000], "LH")

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m):
        return [("HUNGER", 1)]

    cdef void init_fields(self, NativePopulation pop):
        pop.field("HUNGER")[:] = 1.0

    def sech__(self, x):
        return np.where(np.abs(x) > 50, 0.5, 2 / (np.exp(np.clip(x, -50, 50)) + np.exp(-np.clip(x, -50, 50))))

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        F = np.array(pop.F)
        hunger = pop.field("HUNGER")
        g = np.array(X[ops.best_row(self, pop)])
        fb = F.min() if self.problem.minmax == "min" else F.max()
        fw = F.max() if self.problem.minmax == "min" else F.min()
        # hunger grows with the distance to the best fitness (Eqs. 2.8, 2.9)
        r = rng.random(n)
        space = np.mean(self.problem.ub - self.problem.lb)
        H = (F - fb) / (fw - fb + self.EPSILON) * r * 2 * space
        H = np.where(H < self.LH, self.LH * (1 + r), H)
        hunger[:, 0] += H
        hunger[F == fb, 0] = 0
        h = np.array(hunger[:, 0])
        total_hunger = h.sum()
        shrink = 2 * (1 - epoch_c / self.epoch)  # Eq. (2.4)
        E = self.sech__(F - fb)[:, None]  # variation control
        R = 2 * shrink * rng.random((n, 1)) - shrink  # Eq. (2.3)
        W1 = np.where((rng.random(n) < self.PUP)[:, None], (h * n / (total_hunger + self.EPSILON))[:, None] * rng.random((n, 1)), 1.0)
        W2 = (1 - np.exp(-np.abs(h - total_hunger)))[:, None] * rng.random((n, 1)) * 2
        r1, r2 = rng.random((n, 1)), rng.random((n, 1))
        away = R * W2 * np.abs(g - X)
        pos = np.where(r1 < self.PUP, X * (1 + rng.normal(0, 1, (n, 1))),
                       np.where(r2 > E, W1 * g + away, W1 * g - away))  # Eq. (2.1)
        ops.step(self, pos)

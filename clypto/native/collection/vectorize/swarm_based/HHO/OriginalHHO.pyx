#!/usr/bin/env python
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalHHO(LegacyNativeOptimizer):
    """
    The original version of: Harris Hawks Optimization (HHO)

    Links:
        1. https://doi.org/10.1016/j.future.2019.02.028

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import HHO    >>> import numpy as np
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
    >>> model = HHO.OriginalHHO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Heidari, A.A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M. and Chen, H., 2019.
    Harris hawks optimization: Algorithm and applications. Future generation computer systems, 97, pp.849-872.
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

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation sub
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        minmax = self.problem.minmax
        E0 = 2 * rng.uniform(size=(n, 1)) - 1
        E = 2 * E0 * (1.0 - epoch * 1.0 / self.epoch)  # decreasing energy of the rabbit
        J = 2 * (1 - rng.uniform(size=(n, 1)))
        X_m = np.mean(np.array(X))
        R = rng.uniform(size=(n, 6))
        absE = np.abs(E)
        # -------- Exploration phase, Eq. (1) --------
        X_rand = X[rng.integers(0, n, size=n)]
        pos_family = X_rand - R[:, 0:1] * np.abs(X_rand - 2 * R[:, 1:2] * X)
        pos_tree = (g - X_m) - R[:, 2:3] * (lb + R[:, 3:4] * (ub - lb))
        explore = np.where((rng.random(n) >= 0.5)[:, None], pos_family, pos_tree)
        # -------- Exploitation phase --------
        delta_X = g - X
        pounce = np.where(absE >= 0.5, delta_X - E * np.abs(J * g - X), g - E * np.abs(delta_X))  # Eqs. (6), (4)
        Y = np.where(absE >= 0.5, g - E * np.abs(J * g - X), g - E * np.abs(J * g - X_m))  # Eqs. (10), (11)
        LF_D = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=(n, 1), case=-1)
        Z = Y + rng.uniform(lb, ub, size=(n, d)) * LF_D
        pos = np.where(absE >= 1, explore, pounce)
        levy = np.flatnonzero((absE[:, 0] < 1) & (rng.random(n) < 0.5))
        if len(levy):  # rapid dives: keep Y or Z only if better than the agent, else stay
            sub = pop.take(levy)
            sub.X[:] = self.correct_solution(Y[levy])
            self.evaluate(sub, 0, len(levy))
            better_y = ops.better(self, sub.F, pop.F[levy])
            sub.X[:] = self.correct_solution(Z[levy])
            self.evaluate(sub, 0, len(levy))
            better_z = ops.better(self, sub.F, pop.F[levy])
            pos[levy] = np.where(better_y[:, None], Y[levy], np.where(better_z[:, None], Z[levy], X[levy]))
        ops.step(self, pos)

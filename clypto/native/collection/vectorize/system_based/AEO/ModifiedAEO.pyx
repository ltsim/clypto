#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation



cdef class ModifiedAEO(LegacyNativeOptimizer):
    """
    The original version of: Modified Artificial Ecosystem-Based Optimization (MAEO)

    Links:
        1. https://doi.org/10.1109/ACCESS.2020.2973351

    Examples
    ~~~~~~~~
    >>> from clypto.collection.system_based import AEO    >>> import numpy as np
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
    >>> model = AEO.ModifiedAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Menesy, A.S., Sultan, H.M., Korashy, A., Banakhr, F.A., Ashmawy, M.G. and Kamel, S., 2020. Effective
    parameter extraction of different polymer electrolyte membrane fuel cell stack models using a
    modified artificial ecosystem optimization algorithm. IEEE Access, 8, pp.31892-31909.
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
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d, m = pop.n - 1
        cdef object rng = self.generator
        X = pop.X
        lb, ub = self.problem.lb, self.problem.ub
        g = np.array(self.g_best_x())
        ## Production: the worst agent (last row) is replaced by a new random-mixed agent
        H = 2 * (1 - epoch / self.epoch)
        a = (1 - epoch / self.epoch) * rng.random()
        pos = self.correct_solution((1 - a) * X[n - 1] + a * rng.uniform(lb, ub))
        ops.set_row(pop, n - 1, pos, self.get_target(pos))
        X = pop.X
        ## Consumption: the other agents feed on the producer, a random previous agent or both
        rand = rng.random(m)
        v = rng.normal(0, 1, (m, 2))
        c = (0.5 * v[:, 0] / np.abs(v[:, 1]))[:, None]  # consumption factor
        jdx = np.where(np.arange(m) == 0, 1, (rng.random(m) * np.arange(m)).astype(int))
        r2 = rng.random((m, 1))
        Xm, x0, xj = X[:m], X[0], X[jdx]
        her = Xm + H * c * (Xm - x0)
        car = Xm + H * c * (Xm - xj)
        omn = Xm + H * c * (r2 * (Xm - x0) + (1 - r2) * (Xm - xj))
        pos = np.where((rand < 1.0 / 3)[:, None], her, np.where((rand <= 2.0 / 3)[:, None], car, omn))
        ops.step(self, pos, stop=m)
        ## Decomposition around the best agent, or a mix with a random agent (Eq. 21)
        best = np.array(X[self.sorted_order(pop)[0]])
        X = pop.X
        r3 = rng.uniform(size=(n, 1))
        dd = 3 * rng.normal(size=(n, 1))
        e = r3 * rng.integers(1, 3, size=(n, 1)) - 1
        h = 2 * r3 - 1
        beta = 1 - (1 - 0) * (epoch / self.epoch)
        x_r = X[ops.others(self, n)[:, 0]]
        mixed = np.where((rng.random(n) < 0.5)[:, None], beta * x_r + (1 - beta) * X, (1 - beta) * x_r + beta * X)
        ops.step(self, np.where((rng.random(n) < 0.5)[:, None], mixed, best + dd * (e * best - h * X)))

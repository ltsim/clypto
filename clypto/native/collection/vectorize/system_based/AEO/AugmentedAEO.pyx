#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation



cdef class AugmentedAEO(LegacyNativeOptimizer):
    """
    The original version of: Augmented Artificial Ecosystem Optimization (AAEO)

    Notes:
        + Used linear weight factor reduce from 2 to 0 through time
        + Applied Levy-flight technique and the global best solution

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.system_based import AEO    >>> import numpy as np
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
    >>> model = AEO.AugmentedAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Van Thieu, N., Barma, S. D., Van Lam, T., Kisi, O., & Mahesha, A. (2022). Groundwater level modeling
    using Augmented Artificial Ecosystem Optimization. Journal of Hydrology, 129034.
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
        wf = 2 * (1 - epoch / self.epoch)  # weight factor
        a = (1.0 - epoch / self.epoch) * rng.random()
        pos = self.correct_solution((1 - a) * X[n - 1] + a * rng.uniform(lb, ub))
        ops.set_row(pop, n - 1, pos, self.get_target(pos))
        X = pop.X
        ## Consumption (or a Levy move towards the best)
        rand = rng.random(m)
        v = rng.normal(0, 1, (m, 2))
        c = (0.5 * v[:, 0] / np.abs(v[:, 1]))[:, None]
        jdx = np.where(np.arange(m) == 0, 1, (rng.random(m) * np.arange(m)).astype(int))
        r2 = rng.random((m, 1))
        Xm, x0, xj = X[:m], X[0], X[jdx]
        her = Xm + wf * c * (Xm - x0)
        car = Xm + wf * c * (Xm - xj)
        omn = Xm + wf * c * (r2 * (Xm - x0) + (1 - r2) * (Xm - xj))
        feed = np.where((rand < 1.0 / 3)[:, None], her, np.where((rand <= 2.0 / 3)[:, None], car, omn))
        levy = Xm + self.get_levy_flight_step(1.0, 0.001, size=(m, 1), case=-1) * (1.0 / np.sqrt(epoch)) * np.sign(rng.random((m, 1)) - 0.5) * (Xm - g)
        ops.step(self, np.where((rng.random(m) < 0.5)[:, None], feed, levy), stop=m)
        ## Decomposition
        best = np.array(X[self.sorted_order(pop)[0]])
        X = pop.X
        gauss = best + rng.normal(0, 1, (n, d)) * (best - X)
        beta = rng.uniform(0.01, 1.0)
        levy2 = best + self.get_levy_flight_step(beta=beta, multiplier=0.01, size=(n, d), case=-1) * rng.random((n, 1)) * (best - X)
        ops.step(self, np.where((rng.random(n) < 0.5)[:, None], gauss, levy2))

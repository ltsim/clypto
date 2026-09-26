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


cdef class OriginalARO(LegacyNativeOptimizer):
    """
    The original version of: Artificial Rabbits Optimization (ARO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2022.105082
        2. https://www.mathworks.com/matlabcentral/fileexchange/110250-artificial-rabbits-optimization-aro

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
    >>> model = ARO.OriginalARO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, L., Cao, Q., Zhang, Z., Mirjalili, S., & Zhao, W. (2022). Artificial rabbits optimization: A new bio-inspired
    meta-heuristic algorithm for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 114, 105082.
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
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        theta = 2 * (1 - epoch / self.epoch)
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
        hiding = X + R * (rng.random((n, 1)) * b - X)
        ops.step(self, np.where((A > 1)[:, None], detour, hiding))

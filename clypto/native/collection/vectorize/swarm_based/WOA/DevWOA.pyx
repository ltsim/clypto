#!/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevWOA(VectorizeOptimizer):
    """
    The developed version of: Whale Optimization Algorithm (WOA)

    Notes:
        + Hanlding simple vector instead of loop through whole dimensions
        + Using greedy to update position

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import WOA    >>> import numpy as np
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
    >>> model = WOA.DevWOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm. Advances in engineering software, 95, pp.51-67.
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
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def _evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        a = 2 - 2 * epoch / self.epoch  # linearly decreased from 2 to 0
        r = rng.random((n, 1))
        A = 2 * a * r - a
        C = 2 * r
        l = rng.uniform(-1, 1, size=(n, 1))
        pos1 = g - A * np.abs(C * g - X)
        x2 = X[ops.others(self, n)[:, 0]]
        pos2 = x2 - A * np.abs(C * x2 - X)
        pos3 = g + np.exp(l) * np.cos(2 * np.pi * l) * np.abs(g - X)
        pos = np.where(rng.random((n, d)) < 0.5, np.where(np.abs(A) < 1, pos1, pos2), pos3)
        ops.step(self, pos)

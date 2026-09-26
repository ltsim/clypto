#!/usr/bin/env python
# Created by "Thieu" at 12:51, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalBBOA(VectorizeOptimizer):
    """
    The original version of: Brown-Bear Optimization Algorithm (BBOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/125490-brown-bear-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.bio_based import BBOA    >>> import numpy as np
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
    >>> model = BBOA.OriginalBBOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Prakash, T., Singh, P. P., Singh, V. P., & Singh, S. N. (2023). A Novel Brown-bear Optimization
    Algorithm for Solving Economic Dispatch Problem. In Advanced Control & Optimization Paradigms for
    Energy System Operation and Management (pp. 137-164). River Publishers.
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
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        gw = np.array(self.g_worst.solution)
        pp = epoch_c / self.epoch
        ## Pedal marking behaviour
        if pp <= 1 / 3:  # Gait while walking
            pos = X + (-pp * rng.random((n, d)) * X)
        elif pp <= 2 / 3:  # Careful stepping
            pos = X + pp * rng.random((n, d)) * (g - rng.integers(1, 3, size=(n, 1)) * gw)
        else:
            ww = 2 * pp * np.pi * rng.random((n, d))
            pos = X + (ww * g - np.abs(X)) - (ww * gw - np.abs(X))
        ops.step(self, pos)
        ## Sniffing of pedal marks
        X = self.pop.X
        kk = ops.others(self, n)[:, 0]
        mine_better = ops.better(self, self.pop.F, self.pop.F[kk])[:, None]
        r = rng.random((n, 1))
        ops.step(self, np.where(mine_better, X + r * (X - X[kk]), X + r * (X[kk] - X)))

#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from math import gamma
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalMSA(VectorizeOptimizer):
    """
    The original version: Moth Search Algorithm (MSA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/59010-moth-search-ms-algorithm
        2. https://doi.org/10.1007/s12293-016-0212-3

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_best (int): [3, 10], how many of the best moths to keep from one generation to the next, default=5
        + partition (float): [0.3, 0.8], The proportional of first partition, default=0.5
        + max_step_size (float): [0.5, 2.0], Max step size used in Levy-flight technique, default=1.0

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import MSA    >>> import numpy as np
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
    >>> model = MSA.OriginalMSA(epoch=1000, pop_size=50, n_best = 5, partition = 0.5, max_step_size = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., 2018. Moth search algorithm: a bio-inspired metaheuristic algorithm for
    global optimization problems. Memetic Computing, 10(2), pp.151-164.
    """

    cdef public object n_best
    cdef public object partition
    cdef public object max_step_size
    cdef public object n_moth1
    cdef public object n_moth2
    cdef public object golden_ratio

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_best: int = 5,
        partition: float = 0.5,
        max_step_size: float = 1.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_best (int): how many of the best moths to keep from one generation to the next, default=5
            partition (float): The proportional of first partition, default=0.5
            max_step_size (float): Max step size used in Levy-flight technique, default=1.0
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "n_best", "partition", "max_step_size"],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.n_best = cy.validator(int, n_best, [2, int(self.pop_size / 2)], "n_best")
        self.partition = cy.validator(float, partition, (0, 1.0), "partition")
        self.max_step_size = cy.validator(float, max_step_size, (0, 5.0), "max_step_size")
        self.n_moth1 = int(np.ceil(self.partition * self.pop_size))
        self.n_moth2 = self.pop_size - self.n_moth1
        self.golden_ratio = (np.sqrt(5) - 1) / 2.0

    def _evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d, m1 = self.n_moth1, m2 = n - self.n_moth1
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        elites = pop.take(np.arange(self.n_best))
        beta = 1.5  # Eq. 2.23
        sigma = (
                        gamma(1 + beta)
                        * np.sin(np.pi * (beta - 1) / 2)
                        / (gamma(beta / 2) * (beta - 1) * 2 ** ((beta - 2) / 2))
                ) ** (1 / (beta - 1))
        lb, ub = self.problem.bounds.low, self.problem.bounds.up
        # Migration operator: Levy walk of the first moths (Eq. 2.21)
        u = rng.uniform(lb, ub, size=(m1, d)) * sigma
        v = rng.uniform(lb, ub, size=(m1, d))
        delta = (self.max_step_size / epoch) * (u / np.abs(v) ** (1.0 / (beta - 1)))
        pos = np.empty((n, d))
        pos[:m1] = X[:m1] + rng.random((m1, d)) * delta
        # Flying in a straight line
        R = rng.random((m2, 3, d))
        case1 = X[m1:] + R[:, 0] * self.golden_ratio * (g - X[m1:])
        case2 = X[m1:] + R[:, 1] * (1.0 / self.golden_ratio) * (g - X[m1:])
        pos[m1:] = np.where(R[:, 2] < 0.5, case2, case1)
        ops.step(self, pos)
        pop = self.pop = self.pop.take(self.sorted_order(self.pop))
        # Replace the worst with the previous generation's elites.
        pop.buf[n - self.n_best:] = elites.buf[:self.n_best][::-1]

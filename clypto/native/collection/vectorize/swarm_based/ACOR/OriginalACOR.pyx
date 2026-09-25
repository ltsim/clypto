#!/usr/bin/env python
# Created by "Thieu" at 14:14, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalACOR(LegacyNativeOptimizer):
    """
    The original version of: Ant Colony Optimization Continuous (ACOR)

    Notes:
        + Use Gaussian Distribution (np.random.normal() function) instead of random number (np.random.rand())
        + Amend solution when they went out of space

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + sample_count (int): [2, 10000], Number of Newly Generated Samples, default = 25
        + intent_factor (float): [0.2, 1.0], Intensification Factor (Selection Pressure), (q in the paper), default = 0.5
        + zeta (float): [1, 2, 3], Deviation-Distance Ratio, default = 1.0

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import ACOR    >>> import numpy as np
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
    >>> model = ACOR.OriginalACOR(epoch=1000, pop_size=50, sample_count = 25, intent_factor = 0.5, zeta = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Socha, K. and Dorigo, M., 2008. Ant colony optimization for continuous domains.
    European journal of operational research, 185(3), pp.1155-1173.
    """

    cdef public object sample_count
    cdef public object intent_factor
    cdef public object zeta

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        sample_count: int = 25,
        intent_factor: float = 0.5,
        zeta: float = 1.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            sample_count: Number of Newly Generated Samples, default = 25
            intent_factor: Intensification Factor (Selection Pressure) (q in the paper), default = 0.5
            zeta: Deviation-Distance Ratio, default = 1.0
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "sample_count", "intent_factor", "zeta"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.sample_count = cy.validator(int, sample_count, [2, 10000], "sample_count")
        self.intent_factor = cy.validator(float, intent_factor, (0, 1.0), "intent_factor")
        self.zeta = cy.validator(float, zeta, (0, 5), "zeta")

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation sample, both
        cdef Py_ssize_t n = pop.n, d = pop.d, m = self.sample_count
        cdef object rng = self.generator
        X = np.array(pop.X)
        qn = self.intent_factor * n
        w = 1 / (np.sqrt(2 * np.pi) * qn) * np.exp(-0.5 * ((np.arange(1, n + 1) - 1) / qn) ** 2)
        p = w / np.sum(w)  # probability of every rank
        # sigma[i, j] = zeta * sum_k |X[k, j] - X[i, j]| / (n - 1)
        sigma = np.empty((n, d))
        step = max(1, 4000000 // max(1, n * d))
        for i0 in range(0, n, step):
            i1 = min(n, i0 + step)
            sigma[i0:i1] = self.zeta * np.abs(X[None, :, :] - X[i0:i1, None, :]).sum(axis=1) / (n - 1)
        rdx = rng.choice(n, size=(m, d), p=p)  # a rank chosen per (sample, dimension)
        cols = np.arange(d)[None, :]
        sample = pop.take(np.zeros(m, dtype=int))
        sample.X[:] = self.correct_solution(X[rdx, cols] + rng.normal(size=(m, d)) * sigma[rdx, cols])
        self.evaluate(sample, 0, m)
        both = pop.concat(sample)
        self.pop = both.take(self.sorted_order(both)[:self.pop_size])

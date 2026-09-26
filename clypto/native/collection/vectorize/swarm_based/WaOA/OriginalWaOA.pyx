#!/usr/bin/env python
# Created by "Thieu" at 23:18, 11/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalWaOA(LegacyNativeOptimizer):
    """
    The original version of: Walrus Optimization Algorithm (WaOA)

    Links:
        1. https://www.researchgate.net/publication/364684780_Walrus_Optimization_Algorithm_A_New_Bio-Inspired_Metaheuristic_Algorithm

    Notes:
        1. This is somewhat concerning, as there appears to be a high degree of similarity between the source code for this algorithm and the Northern Goshawk Optimization (NGO)
        2. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA), Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Northern Goshawk Optimization (NGO), Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Pelican Optimization Algorithm (POA), Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        3. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        4. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import WaOA    >>> import numpy as np
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
    >>> model = WaOA.OriginalWaOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovský, P., & Dehghani, M. (2022). Walrus Optimization Algorithm: A New Bio-Inspired Metaheuristic Algorithm.
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
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        # Phase 1: feeding strategy (exploration), Eq. 4
        kk = rng.integers(0, n, size=n)
        toward = ops.better(self, pop.F[kk], pop.F)[:, None]
        r1 = rng.integers(1, 3, size=(n, 1))
        R = rng.random((n, 1))
        ops.step(self, np.where(toward, X + R * (X[kk] - r1 * X), X + R * (X - X[kk])))
        # PHASE 2: exploitation (Eq. 7)
        LB, UB = self.problem.lb / epoch, self.problem.ub / epoch
        ops.step(self, X + LB + (UB - rng.random((n, 1)) * LB))

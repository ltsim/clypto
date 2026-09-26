#!/usr/bin/env python
# Created by "Thieu" at 18:22, 11/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalFFO(LegacyNativeOptimizer):
    """
    The original version of: Fennec Fox Optimization (FFO)

    Links:
        1. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9853509

    Notes:
        1. This is somewhat concerning, as there appears to be a high degree of similarity between the source code for this algorithm and the Pelican Optimization Algorithm (POA).
        2. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA), Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA), Pelican Optimization Algorithm (POA), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO), Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        3. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        4. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import FFO    >>> import numpy as np
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
    >>> model = FFO.OriginalFFO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovská, E., Dehghani, M., & Trojovský, P. (2022). Fennec Fox Optimization: A New
    Nature-Inspired Optimization Algorithm. IEEE Access, 10, 84417-84443.
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
        # PHASE 1: the digging to look for prey under the sand (exploitation)
        rr = 0.2 * (1 - epoch / self.epoch) * X
        ops.step(self, X + (2 * rng.random((n, 1)) * 1) * rr)
        # PHASE 2: escape strategy from the predators' attack (exploration)
        kk = ops.others(self, n)[:, 0]
        toward = ops.better(self, pop.F[kk], pop.F)[:, None]
        r1 = rng.integers(1, 3, size=(n, 1))
        R = rng.random((n, 1))
        ops.step(self, np.where(toward, X + R * (X[kk] - r1 * X), X + R * (X - X[kk])))

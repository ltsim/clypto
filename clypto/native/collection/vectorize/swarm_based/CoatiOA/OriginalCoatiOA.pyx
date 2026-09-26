#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalCoatiOA(LegacyNativeOptimizer):
    """
    The original version of: Coati Optimization Algorithm (CoatiOA)

    Links:
        1. https://www.sciencedirect.com/science/article/pii/S0950705122011042
        2. https://www.mathworks.com/matlabcentral/fileexchange/116965-coa-coati-optimization-algorithm

    Notes:
        1. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Pelican optimization algorithm (POA), Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA), Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO), Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        2. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        3. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import CoatiOA    >>> import numpy as np
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
    >>> model = CoatiOA.OriginalCoatiOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Dehghani, M., Montazeri, Z., Trojovská, E., & Trojovský, P. (2023). Coati Optimization Algorithm: A new
    bio-inspired metaheuristic algorithm for solving optimization problems. Knowledge-Based Systems, 259, 110011.
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
        cdef NativePopulation iguana
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        h = int(n / 2)
        # Phase1: hunting and attacking strategy on iguana (exploration phase), Eqs. 4 and 6
        pos = np.empty((n, d))
        pos[:h] = X[:h] + rng.random((h, 1)) * (g - rng.integers(1, 3, size=(h, 1)) * X[:h])
        iguana = pop.take(np.arange(n - h))
        iguana.X[:] = lb + rng.random((n - h, d)) * (ub - lb)
        self.evaluate(iguana, 0, n - h)
        xi, xs = iguana.X, X[h:]
        toward = ops.better(self, iguana.F, pop.F[h:])[:, None]
        R = rng.random((n - h, 1))
        pos[h:] = np.where(toward, xs + R * (xi - rng.integers(1, 3, size=(n - h, 1)) * xs), xs + R * (xs - xi))
        ops.step(self, pos)
        # Phase2: the process of escaping from predators (exploitation phase), Eq. 8
        X = pop.X
        LO, HI = lb / epoch, ub / epoch
        ops.step(self, X + (1 - 2 * rng.random((n, 1))) * (LO + rng.random((n, 1)) * (HI - LO)))

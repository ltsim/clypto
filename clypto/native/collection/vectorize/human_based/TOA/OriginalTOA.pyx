#!/usr/bin/env python
# Created by "Thieu" at 18:22, 11/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalTOA(VectorizeOptimizer):
    """
    The original version of: Teamwork Optimization Algorithm (TOA)

    Links:
        1. https://www.mdpi.com/1424-8220/21/13/4567

    Notes:
        1. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA),
        Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA),
        Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Pelican Optimization Algorithm (POA), Northern goshawk optimization (NGO),
        Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)

        2. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.

        3. While this article may share some similarities with previous work by the same authors, it is important to recognize the potential value in exploring
        different meta-metaphors and concepts to drive innovation and progress in optimization research.

        4. Further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import TOA    >>> import numpy as np
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
    >>> model = TOA.OriginalTOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Dehghani, M., & Trojovský, P. (2021). Teamwork optimization algorithm: A new optimization
    approach for function minimization/maximization. Sensors, 21(13), 4567.
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
        cdef NativePopulation sf
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        gb_fit = self.current_g_best().target.fitness
        # phase 1: towards the best
        ops.step(self, X + rng.random((n, 1)) * (g - rng.integers(1, 3, size=(n, 1)) * X))
        # phase 2: towards the mean of the agents that are better than the agent
        X = pop.X
        F = np.asarray(pop.F)
        B = (F[None, :] < F[:, None]) if self.problem.sense == "min" else (F[None, :] > F[:, None])
        count = B.sum(axis=1)
        has = count > 0
        sf_pos = np.tile(g, (n, 1))
        sf_fit = np.full(n, gb_fit, dtype=float)
        if has.any():
            rows = np.flatnonzero(has)
            sf = pop.take(rows)
            sf.X[:] = self._correct_solution((B[rows].astype(float) @ X) / count[rows][:, None])
            self.evaluate(sf, 0, len(rows))
            sf_pos[rows], sf_fit[rows] = sf.X, sf.F
        ops.step(self, X + rng.random((n, 1)) * (sf_pos - rng.integers(1, 3, size=(n, 1)) * X) * np.sign(F - sf_fit)[:, None])
        # phase 3: local search
        X = pop.X
        ops.step(self, X + (-0.01 + rng.random((n, 1)) * 0.02) * X)

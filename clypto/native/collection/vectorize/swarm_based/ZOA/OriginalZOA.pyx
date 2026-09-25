#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalZOA(LegacyNativeOptimizer):
    """
    The original version of: Zebra Optimization Algorithm (ZOA)

    Links:
        1. https://ieeexplore.ieee.org/document/9768820
        2. https://www.mathworks.com/matlabcentral/fileexchange/122942-zebra-optimization-algorithm-zoa

    Notes:
                1. It's concerning that the author seems to be reusing the same algorithms with minor variations.
        2. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Pelican optimization algorithm (POA), Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA), Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO), Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO).
        3. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        4. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import ZOA    >>> import numpy as np
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
    >>> model = ZOA.OriginalZOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovská, E., Dehghani, M., & Trojovský, P. (2022). Zebra optimization algorithm: A new bio-inspired
    optimization algorithm for solving optimization algorithm. IEEE Access, 10, 49445-49473.
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
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        # PHASE1: Foraging Behaviour (one agent per row: the draws follow the agent order)
        for idx in range(0, self.pop_size):
            r1 = np.round(1 + self.generator.random())
            pos_new = Xp[idx] + self.generator.random(d) * (g_best - r1 * Xp[idx])  # Eq. 3
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand, old_first=False)

        # PHASE2: defense strategies against predators
        kk = self.generator.permutation(self.pop_size)[0]
        cand = pop.empty_like()
        Xp = pop.X
        for idx in range(0, self.pop_size):
            if self.generator.random() < 0.5:
                # S1: the lion attacks the zebra and thus the zebra chooses an escape strategy
                r2 = 0.1
                pos_new = Xp[idx] + r2 * (2 + self.generator.random(d) - 1) * (1 - epoch / self.epoch) * Xp[idx]
            else:
                # S2: other predators attack the zebra and the zebra will choose the offensive strategy
                r2 = self.generator.integers(1, 3)
                pos_new = Xp[idx] + self.generator.random(d) * (Xp[kk] - r2 * Xp[idx])
            # sequential mode: later agents read the rows already replaced (Xp[kk])
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

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
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalOOA(LegacyNativeOptimizer):
    """
    The original version of: Osprey Optimization Algorithm (OOA)

    Links:
        1. https://www.frontiersin.org/articles/10.3389/fmech.2022.1126450/full
        2. https://www.mathworks.com/matlabcentral/fileexchange/124555-osprey-optimization-algorithm

    Notes:
        1. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Pelican optimization algorithm (POA), Siberian Tiger Optimization (STO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA), Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Northern goshawk optimization (NGO), Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        2. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        3. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import OOA    >>> import numpy as np
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
    >>> model = OOA.OriginalOOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovský, P., & Dehghani, M. Osprey Optimization Algorithm: A new bio-inspired metaheuristic algorithm
    for solving engineering optimization problems. Frontiers in Mechanical Engineering, 8, 136.
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

    def get_indexes_better__(self, NativePopulation pop, idx):
        fits = np.array(pop.F)
        if self.problem.minmax == "min":
            idxs = np.where(fits < pop.F[idx])
        else:
            idxs = np.where(fits > pop.F[idx])
        return idxs[0]

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativeTarget tar
        cdef Py_ssize_t idx
        minmax = self.problem.minmax
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        for idx in range(0, self.pop_size):
            # Phase 1: : POSITION IDENTIFICATION AND HUNTING THE FISH (EXPLORATION)
            idxs = self.get_indexes_better__(pop, idx)
            if len(idxs) == 0:
                sf = g_best
            else:
                if self.generator.random() < 0.5:
                    sf = g_best
                else:
                    kk = self.generator.permutation(idxs)[0]
                    sf = Xp[kk]
            r1 = self.generator.integers(1, 3)
            pos_new = Xp[idx] + self.generator.normal(0, 1) * (sf - r1 * Xp[idx])  # Eq. 5
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)
            if self.compare_fitness(tar.fitness, pop.F[idx], minmax):
                ops.set_row(pop, idx, pos_new, tar)

            # PHASE 2: CARRYING THE FISH TO THE SUITABLE POSITION (EXPLOITATION)
            pos_new = Xp[idx] + self.problem.lb + self.generator.random() * (self.problem.ub - self.problem.lb)  # Eq. 7
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)
            if self.compare_fitness(tar.fitness, pop.F[idx], minmax):
                ops.set_row(pop, idx, pos_new, tar)

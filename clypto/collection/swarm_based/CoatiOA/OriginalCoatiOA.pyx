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
        cdef NativeTarget tar
        cdef Py_ssize_t idx
        minmax = self.problem.minmax
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        # Phase1: Hunting and attacking strategy on iguana (Exploration Phase)
        size2 = int(self.pop_size / 2)
        for idx in range(0, size2):
            pos_new = Xp[idx] + self.generator.random() * (g_best - self.generator.integers(1, 3) * Xp[idx])  # Eq. 4
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)
            if self.compare_fitness(tar.fitness, pop.F[idx], minmax):
                ops.set_row(pop, idx, pos_new, tar)

        for idx in range(size2, self.pop_size):
            iguana_x = self.problem.generate_solution(encoded=True)
            iguana_tar = self.get_target(iguana_x)
            if self.compare_fitness(iguana_tar.fitness, pop.F[idx], minmax):
                pos_new = Xp[idx] + self.generator.random() * (iguana_x - self.generator.integers(1, 3) * Xp[idx])  # Eq. 6
            else:
                pos_new = Xp[idx] + self.generator.random() * (Xp[idx] - iguana_x)  # Eq. 6
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)
            if self.compare_fitness(tar.fitness, pop.F[idx], minmax):
                ops.set_row(pop, idx, pos_new, tar)

        # Phase2: The process of escaping from predators (Exploitation Phase)
        for idx in range(0, self.pop_size):
            LO, HI = self.problem.lb / epoch, self.problem.ub / epoch
            pos_new = Xp[idx] + (1 - 2 * self.generator.random()) * (LO + self.generator.random() * (HI - LO))  # Eq. 8
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)
            if self.compare_fitness(tar.fitness, pop.F[idx], minmax):
                ops.set_row(pop, idx, pos_new, tar)

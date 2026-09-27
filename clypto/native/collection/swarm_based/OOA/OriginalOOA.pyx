#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalOOA(cy.Optimizer):
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
    >>> from clypto.native.collection.swarm_based import OOA    >>> import numpy as np
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
    >>> model = OOA.OriginalOOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovský, P., & Dehghani, M. Osprey Optimization Algorithm: A new bio-inspired metaheuristic algorithm
    for solving engineering optimization problems. Frontiers in Mechanical Engineering, 8, 136.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    def get_indexes_better__(self, pop, idx):
        fits = np.array([agent.fitness for agent in self.population])
        if self.problem.sense == "min":
            idxs = np.where(fits < pop[idx].fitness)
        else:
            idxs = np.where(fits > pop[idx].fitness)
        return idxs[0]

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        for idx, agent in enumerate(self.population.toarray()):
            # Phase 1: : POSITION IDENTIFICATION AND HUNTING THE FISH (EXPLORATION)
            idxs = self.get_indexes_better__(self.population, idx)
            if len(idxs) == 0:
                sf = self.g_best
            else:
                if self.generator.random() < 0.5:
                    sf = self.g_best
                else:
                    kk = self.generator.permutation(idxs)[0]
                    sf = self.population[kk]
            r1 = self.generator.integers(1, 3)
            x = agent.solution + self.generator.normal(0, 1) * (
                    sf.solution - r1 * agent.solution
            )  # Eq. 5
            x = cy.correct_solution(self.problem, x)
            child = self.population.generate_agent(x)
            if cy.is_better(child, agent, self.problem.sense):
                self.population[idx] = child

            # PHASE 2: CARRYING THE FISH TO THE SUITABLE POSITION (EXPLOITATION)
            x = (
                    self.population[idx].solution
                    + self.problem.bounds.low
                    + self.generator.random() * (self.problem.bounds.up - self.problem.bounds.low)
            )  # Eq. 7
            x = cy.correct_solution(self.problem, x)
            child = self.population.generate_agent(x)
            if cy.is_better(child, self.population[idx], self.problem.sense):
                self.population[idx] = child

#!/usr/bin/env python
# Created by "Thieu" at 22:00, 11/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalSTO(cy.Optimizer):
    """
    The original version of: Siberian Tiger Optimization (STO)

    Links:
        1. https://ieeexplore.ieee.org/abstract/document/9989374
        2. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9989374

    Notes:
        1. This is somewhat concerning, as there appears to be a high degree of similarity between the source code for this algorithm and the Osprey Optimization Algorithm (OOA)
        2. Algorithm design is similar to Zebra Optimization Algorithm (ZOA), Osprey Optimization Algorithm (OOA), Coati Optimization Algorithm (CoatiOA), Northern Goshawk Optimization (NGO), Language Education Optimization (LEO), Serval Optimization Algorithm (SOA), Walrus Optimization Algorithm (WOA), Fennec Fox Optimization (FFO), Three-periods optimization algorithm (TPOA), Teamwork optimization algorithm (TOA), Pelican Optimization Algorithm (POA), Tasmanian devil optimization (TDO), Archery algorithm (AA), Cat and mouse based optimizer (CMBO)
        3. It may be useful to compare the Matlab code of this algorithm with those of the similar algorithms to ensure its accuracy and completeness.
        4. The article may share some similarities with previous work by the same authors, further investigation may be warranted to verify the benchmark results reported in the papers and ensure their reliability and accuracy.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import STO    >>> import numpy as np
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
    >>> model = STO.OriginalSTO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovský, P., Dehghani, M., & Hanuš, P. (2022). Siberian Tiger Optimization: A New Bio-Inspired
    Metaheuristic Algorithm for Solving Engineering Optimization Problems. IEEE Access, 10, 132396-132431.
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        for idx in range(0, pop_size):
            # PHASE 1: PREY HUNTING
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
            pos_new = self.population[idx].solution + self.generator.random() * (
                    sf.solution - r1 * self.population[idx].solution
            )  # Eq. 5
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.generate_agent(pos_new)
            if cy.is_better(agent, self.population[idx], self.problem.sense):
                self.population[idx] = agent

            # PHASE 2: CARRYING THE FISH TO THE SUITABLE POSITION (EXPLOITATION)
            pos_new = (
                    self.population[idx].solution
                    + self.generator.random() * (self.problem.bounds.up - self.problem.bounds.low) / epoch
            )  # Eq. 7
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.generate_agent(pos_new)
            if cy.is_better(agent, self.population[idx], self.problem.sense):
                self.population[idx] = agent

#!/usr/bin/env python
# Created by "Thieu" at 18:22, 11/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalTOA(cy.Optimizer):
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
    >>> from clypto.native.collection.legacy.human_based import TOA    >>> import numpy as np
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
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Dehghani, M., & Trojovský, P. (2021). Teamwork optimization algorithm: A new optimization
    approach for function minimization/maximization. Sensors, 21(13), 4567.
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
            # Stage 1: Supervisor guidance
            pos_new = self.population[idx].solution + self.generator.random() * (
                    self.g_best.solution
                    - self.generator.integers(1, 3) * self.population[idx].solution
            )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.generate_agent(pos_new)
            if cy.is_better(agent, self.population[idx], self.problem.sense):
                self.population[idx] = agent
            # Stage 2: Information sharing
            idxs = self.get_indexes_better__(self.population, idx)
            if len(idxs) == 0:
                sf = self.g_best
            else:
                sf_pos = np.array([self.population[jdx].solution for jdx in idxs])
                sf_pos = self.population.correct_solution(np.mean(sf_pos, axis=0))
                sf = self.population.generate_agent(sf_pos)
            pos_new = self.population[idx].solution + self.generator.random() * (
                    sf.solution - self.generator.integers(1, 3) * self.population[idx].solution
            ) * np.sign(self.population[idx].fitness - sf.fitness)
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.generate_agent(pos_new)
            if cy.is_better(agent, self.population[idx], self.problem.sense):
                self.population[idx] = agent
            # Stage 3: Individual activity
            pos_new = (
                    self.population[idx].solution
                    + (-0.01 + self.generator.random() * 0.02) * self.population[idx].solution
            )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.generate_agent(pos_new)
            if cy.is_better(agent, self.population[idx], self.problem.sense):
                self.population[idx] = agent

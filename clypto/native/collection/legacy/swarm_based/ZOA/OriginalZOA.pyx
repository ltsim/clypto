#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalZOA(cy.Optimizer):
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
    >>> from clypto.native.collection.legacy.swarm_based import ZOA    >>> import numpy as np
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
    >>> model = ZOA.OriginalZOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Trojovská, E., Dehghani, M., & Trojovský, P. (2022). Zebra optimization algorithm: A new bio-inspired
    optimization algorithm for solving optimization algorithm. IEEE Access, 10, 49445-49473.
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # PHASE1: Foraging Behaviour
        pop_new = []
        for idx in range(0, pop_size):
            r1 = np.round(1 + self.generator.random())
            pos_new = self.population[idx].solution + self.generator.random(
                self.problem.n_dims
            ) * (
                              self.g_best.solution - r1 * self.population[idx].solution
                      )  # Eq. 3
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

        # PHASE2: defense strategies against predators
        kk = self.generator.permutation(pop_size)[0]
        pop_new = []
        for idx in range(0, pop_size):
            if self.generator.random() < 0.5:
                # S1: the lion attacks the zebra and thus the zebra chooses an escape strategy
                r2 = 0.1
                pos_new = (
                        self.population[idx].solution
                        + r2
                        * (2 + self.generator.random(self.problem.n_dims) - 1)
                        * (1 - epoch / self.epoch)
                        * self.population[idx].solution
                )
            else:
                # S2: other predators attack the zebra and the zebra will choose the offensive strategy
                r2 = self.generator.integers(1, 3)
                pos_new = self.population[idx].solution + self.generator.random(
                    self.problem.n_dims
                ) * (self.population[kk].solution - r2 * self.population[idx].solution)
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

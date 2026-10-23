#!/usr/bin/env python
# Created by "Thieu" at 18:37, 28/05/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalCSA(cy.Optimizer):
    """
    The original version of: Cuckoo Search Algorithm (CSA)

    Links:
        1. https://doi.org/10.1109/NABIC.2009.5393690

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_a (float): [0.1, 0.7], probability a, default=0.3

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import CSA    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "sense": "min",
    >>> }
    >>>
    >>> model = CSA.OriginalCSA(epoch=1000, pop_size=50, p_a = 0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S. and Deb, S., 2009, December. Cuckoo search via Lévy flights. In 2009 World
    congress on nature & biologically inspired computing (NaBIC) (pp. 210-214). Ieee.
    """

    cdef public double p_a

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            p_a: float = 0.3,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_a (float): probability a, default=0.3
        """
        super().__init__(parameters=["epoch", "pop_size", "p_a"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.p_a = cy.validator(float, p_a, (0, 1.0), "p_a")
        self.n_cut = int(self.p_a * self.population.size())

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_new = []
        for idx in range(0, pop_size):
            ## Generate levy-flight solution
            levy_step = cy.levy_flight(self.generator, beta=1.0, multiplier=0.001, size=None, case=-1)
            x = self.population[idx].solution + 1.0 / np.sqrt(epoch) * np.sign(
                self.generator.random() - 0.5
            ) * levy_step * (self.population[idx].solution - self.g_best.solution)
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop_new.append(agent)
        self.population = self.population.greedy(self.population.evaluate(pop_new, self.mode), self.mode)

        ## Abandoned some worst nests
        pop = self.population.sort()[:pop_size]
        pop_new = []
        for idx in range(0, self.n_cut):
            x = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            agent = self.population.create_agent(x)
            pop_new.append(agent)
        pop_new = self.population.evaluate(pop_new, self.mode)
        self.population = self.population.spawn(pop[: (pop_size - self.n_cut)] + pop_new)

#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class ExGWO(cy.Optimizer):
    """
    The original version of: Expanded Grey Wolf Optimizer (Ex-GWO)

    Notes:
        + When calling the solve() function, you need to set the mode to "swarm" to use this algorithm as original version.
        + They update the position of whole population before calculating the fitness of each agent.

    Links:
        1. https://doi.org/10.1007/s00366-019-00837-7

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.ExGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Seyyedabbasi, A., & Kiani, F. (2021). I-GWO and Ex-GWO: improved algorithms of the Grey Wolf Optimizer to solve global optimization problems. Engineering with Computers, 37(1), 509-532.
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

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # linearly decreased from 2 to 0
        a = 2 * (1.0 - epoch / self.epoch)
        pop_sorted = self.population.sort()
        list_best = [cy.duplicate_agent(agent) for agent in pop_sorted[:3]]
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            if idx == 0:
                A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C1 = 2 * self.generator.random(self.problem.n_dims)
                x = list_best[0].solution - A1 * np.abs(
                    C1 * list_best[0].solution - agent.solution
                )
            elif idx == 1:
                A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C2 = 2 * self.generator.random(self.problem.n_dims)
                x = list_best[1].solution - A2 * np.abs(
                    C2 * list_best[1].solution - agent.solution
                )
            elif idx == 2:
                A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C3 = 2 * self.generator.random(self.problem.n_dims)
                x = list_best[2].solution - A3 * np.abs(
                    C3 * list_best[2].solution - agent.solution
                )
            else:
                # Other wolves update based on first three + previous wolves (Equation 15)
                # Average of first three wolves + previously updated wolves
                x = np.mean(
                    [child.solution for child in pop_sorted[:idx]], axis=0
                )
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)

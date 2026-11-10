#!/usr/bin/env python
# Created by "Thieu" at 22:46, 26/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalARO(cy.Optimizer):
    """
    The original version of: Artificial Rabbits Optimization (ARO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2022.105082
        2. https://www.mathworks.com/matlabcentral/fileexchange/110250-artificial-rabbits-optimization-aro

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import ARO    >>> import numpy as np
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
    >>> model = ARO.OriginalARO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, L., Cao, Q., Zhang, Z., Mirjalili, S., & Zhao, W. (2022). Artificial rabbits optimization: A new bio-inspired
    meta-heuristic algorithm for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 114, 105082.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
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
        theta = 2 * (1 - epoch / self.epoch)
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            L = (np.exp(1) - np.exp((epoch / self.epoch) ** 2)) * (
                np.sin(2 * np.pi * self.generator.random())
            )
            temp = np.zeros(self.problem.n_dims)
            rd_index = self.generator.choice(
                np.arange(0, self.problem.n_dims),
                int(np.ceil(self.generator.random() * self.problem.n_dims)),
                replace=False,
            )
            temp[rd_index] = 1
            R = L * temp  # Eq 2
            A = 2 * np.log(1.0 / self.generator.random()) * theta  # Eq. 15
            if A > 1:  # detour foraging strategy
                rand_idx = self.generator.integers(0, pop_size)
                x = (
                        self.population[rand_idx].solution
                        + R * (agent.solution - self.population[rand_idx].solution)
                        + np.round(0.5 * (0.05 + self.generator.random()))
                        * self.generator.normal(0, 1)
                )  # Eq. 1
            else:  # Random hiding stage
                gr = np.zeros(self.problem.n_dims)
                rd_index = self.generator.choice(
                    np.arange(0, self.problem.n_dims),
                    int(np.ceil(self.generator.random() * self.problem.n_dims)),
                    replace=False,
                )
                gr[rd_index] = 1  # Eq. 12
                H = self.generator.normal(0, 1) * (epoch / self.epoch)  # Eq. 8
                b = agent.solution + H * gr * agent.solution  # Eq. 13
                x = agent.solution + R * (
                        self.generator.random() * b - agent.solution
                )  # Eq. 11
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

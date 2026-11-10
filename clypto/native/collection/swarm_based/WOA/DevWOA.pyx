#!/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevWOA(cy.Optimizer):
    """
    The developed version of: Whale Optimization Algorithm (WOA)

    Notes:
        + Hanlding simple vector instead of loop through whole dimensions
        + Using greedy to update position

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import WOA    >>> import numpy as np
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
    >>> model = WOA.DevWOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm. Advances in engineering software, 95, pp.51-67.
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
        a = 2 - 2 * epoch / self.epoch  # linearly decreased from 2 to 0
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            r = self.generator.random()
            A = 2 * a * r - a
            C = 2 * r
            l = self.generator.uniform(-1, 1)
            p = 0.5
            b = 1

            # Get pos1
            pos1 = self.g_best.solution - A * np.abs(
                C * self.g_best.solution - agent.solution
            )

            # Get pos2
            id_r2 = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
            pos2 = self.population[id_r2].solution - A * np.abs(
                C * self.population[id_r2].solution - agent.solution
            )

            # Get pos3
            D1 = np.abs(self.g_best.solution - agent.solution)
            pos3 = self.g_best.solution + np.exp(b * l) * np.cos(2 * np.pi * l) * D1

            # Get final x
            x = pos1 if np.abs(A) < 1 else pos2
            x = np.where(
                self.generator.random(size=self.problem.n_dims) < p, x, pos3
            )

            # Correct solution
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

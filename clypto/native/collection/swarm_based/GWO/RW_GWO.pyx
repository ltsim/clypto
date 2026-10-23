#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class RW_GWO(cy.Optimizer):
    """
    The original version of: Random Walk Grey Wolf Optimizer (RW-GWO)

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
    >>> model = GWO.RW_GWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Gupta, S. and Deep, K., 2019. A novel random walk grey wolf optimizer. Swarm and evolutionary computation, 44, pp.101-112.
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
        # linearly decreased from 2 to 0, Eq. 5
        b = 2.0 - 2.0 * epoch / self.epoch
        # linearly decreased from 2 to 0
        a = 2.0 - 2.0 * epoch / self.epoch
        ranked = self.population.sort()
        leaders = [cy.duplicate_agent(agent) for agent in ranked[:3]]

        ## Random walk here
        leaders_new = []
        for idx in range(0, len(leaders)):
            x = leaders[idx].solution + a * self.generator.standard_cauchy(
                self.problem.n_dims
            )
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            leaders_new.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                leaders[idx] = cy.get_better_agent(agent, leaders[idx], self.problem.sense)
        if self.mode != "sequential":
            leaders_new = self.population.evaluate(leaders_new, self.mode)
            leaders = cy.greedy_agents(leaders, leaders_new, self.problem.sense)

        ## Update other wolfs
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            # Eq. 3 and 4
            miu1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            miu2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            miu3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            c1 = 2 * self.generator.random(self.problem.n_dims)
            c2 = 2 * self.generator.random(self.problem.n_dims)
            c3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = leaders[0].solution - miu1 * np.abs(
                c1 * self.g_best.solution - agent.solution
            )
            X2 = leaders[1].solution - miu2 * np.abs(
                c2 * self.g_best.solution - agent.solution
            )
            X3 = leaders[2].solution - miu3 * np.abs(
                c3 * self.g_best.solution - agent.solution
            )
            x = (X1 + X2 + X3) / 3.0
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)
        self.population = self.population.spawn(cy.sort_agents(self.population + leaders, self.problem.sense)[:pop_size])

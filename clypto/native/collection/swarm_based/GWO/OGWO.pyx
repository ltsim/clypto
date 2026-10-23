#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OGWO(cy.Optimizer):
    """
    The original version of: Opposition-based learning Grey Wolf Optimizer (OGWO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2021.107139

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
    >>> model = GWO.OGWO(epoch=1000, pop_size=50, miu_factor=2.0, jumping_rate=0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Yu, X., Xu, W., & Li, C. (2021). Opposition-based learning grey wolf optimizer for global optimization. Knowledge-Based Systems, 226, 107139.
    """

    cdef public double jumping_rate
    cdef public double miu_factor

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        miu_factor: float = 2.0,
        jumping_rate: float = 0.05,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_factor (float): nonlinear coefficient for equation (11), default = 2.0
            jumping_rate (float):  jumping rate for OBL, default = 0.05
        """
        super().__init__(parameters=["epoch", "pop_size", "miu_factor", "jumping_rate"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.miu_factor = cy.validator(float, miu_factor, [0.0, 10.0], "miu_factor")
        self.jumping_rate = cy.validator(float, jumping_rate, [0.0, 1.0], "jumping_rate")

    def initialization(self) -> None:
        """Initialize population with opposition-based learning"""
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)

        # Generate opposition population using equation (12)
        pop_opposite = []
        for agent in self.population:
            pos_opposite = self.problem.bounds.low + self.problem.bounds.up - agent.solution
            agent_opposite = self.population.create_agent(pos_opposite)
            agent_opposite.evaluate(self.problem)
            pop_opposite.append(agent_opposite)
        # Combine original and opposite populations
        self.population = self.population.spawn(cy.sort_agents(self.population + pop_opposite, self.problem.sense)[:pop_size])

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # linearly decreased from 2 to 0
        a = 2.0 * (1 - (epoch / self.epoch) ** self.miu_factor)
        ranked = self.population.sort()
        list_best = [cy.duplicate_agent(agent) for agent in ranked[:3]]
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx in range(0, pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(
                C1 * list_best[0].solution - self.population[idx].solution
            )
            X2 = list_best[1].solution - A2 * np.abs(
                C2 * list_best[1].solution - self.population[idx].solution
            )
            X3 = list_best[2].solution - A3 * np.abs(
                C3 * list_best[2].solution - self.population[idx].solution
            )
            x = (X1 + X2 + X3) / 3.0
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            n_population.append(agent)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)

        # Apply opposition-based learning
        if self.generator.random() < self.jumping_rate:
            # Generate opposition population using equation (12)
            pop_opposite = []
            for agent in self.population:
                pos_opposite = self.problem.bounds.low + self.problem.bounds.up - agent.solution
                agent_opposite = self.population.create_agent(pos_opposite)
                agent_opposite.evaluate(self.problem)
                pop_opposite.append(agent_opposite)
            # Combine original and opposite populations
            self.population = self.population.spawn(cy.sort_agents(self.population + pop_opposite, self.problem.sense)[:pop_size])

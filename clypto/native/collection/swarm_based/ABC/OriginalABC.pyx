#!/usr/bin/env python
# Created by "Thieu" at 09:57, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalABC(cy.Optimizer):
    """
    The original version of: Artificial Bee Colony (ABC)

    Links:
        1. https://www.sciencedirect.com/topics/computer-science/artificial-bee-colony

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_limits (int): Limit of trials before abandoning a food source, default=25

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import ABC    >>> import numpy as np
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
    >>> model = ABC.OriginalABC(epoch=1000, pop_size=50, n_limits = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] B. Basturk, D. Karaboga, An artificial bee colony (ABC) algorithm for numeric function optimization,
    in: IEEE Swarm Intelligence Symposium 2006, May 12–14, Indianapolis, IN, USA, 2006.
    """

    cdef public int n_limits

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            n_limits: int = 25,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size = onlooker bees = employed bees, default = 100
            n_limits: Limit of trials before abandoning a food source, default=25
        """
        super().__init__(parameters=["epoch", "pop_size", "n_limits"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.n_limits = cy.validator(int, n_limits, [1, 1000], "n_limits")

    def initialize_variables(self):
        pop_size = self.population.size()
        self.trials = np.zeros(pop_size)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        for idx in range(0, pop_size):
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            x = self.population[idx].solution + phi * (
                    self.population[rdx].solution - self.population[idx].solution
            )
            x = cy.correct_solution(self.problem, x)
            agent = self.population.generate_agent(x)
            if cy.is_better(agent, self.population[idx], self.problem.sense):
                self.population[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1
        # Onlooker bees phase
        # Calculate the probabilities of each employed bee
        employed_fits = np.array([agent.fitness for agent in self.population])
        # probabilities = employed_fits / np.sum(employed_fits)
        for idx in range(0, pop_size):
            # Select an employed bee using roulette wheel selection
            selected_bee = cy.roulette_wheel(self.generator, self.problem.sense, employed_fits)
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(
                list(set(range(0, pop_size)) - {idx, selected_bee})
            )
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            x = self.population[selected_bee].solution + phi * (
                    self.population[rdx].solution - self.population[selected_bee].solution
            )
            x = cy.correct_solution(self.problem, x)
            agent = self.population.generate_agent(x)
            if cy.is_better(agent, self.population[selected_bee], self.problem.sense):
                self.population[selected_bee] = agent
                self.trials[selected_bee] = 0
            else:
                self.trials[selected_bee] += 1
        # Scout bees phase
        # Check the number of trials for each employed bee and abandon the food source if the limit is exceeded
        abandoned = np.where(self.trials >= self.n_limits)[0]
        for idx in abandoned:
            self.population[idx] = self.population.generate_agent()
            self.trials[idx] = 0

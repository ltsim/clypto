#!/usr/bin/env python
# Created by "Thieu" at 13:42, 06/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalSeaHO(cy.Optimizer):
    """
    The original version of: Sea-Horse Optimization (SeaHO)

    Links:
        1. https://link.springer.com/article/10.1007/s10489-022-03994-3
        2. https://www.mathworks.com/matlabcentral/fileexchange/115945-sea-horse-optimizer

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import SeaHO    >>> import numpy as np
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
    >>> model = SeaHO.OriginalSeaHO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, S., Zhang, T., Ma, S., & Wang, M. (2022). Sea-horse optimizer: a novel nature-inspired
    meta-heuristic for global optimization problems. Applied Intelligence, 1-28.
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

    def initialize_variables(self):
        self.uu = 0.05
        self.vv = 0.05
        self.ll = 0.05

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # The motor behavior of sea horses
        step_length = cy.levy_flight(self.generator, beta=1.5, multiplier=0.01, size=(pop_size, self.problem.n_dims), case=-1)
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            beta = self.generator.normal(0, 1, self.problem.n_dims)
            theta = 2 * np.pi * self.generator.random(self.problem.n_dims)
            row = self.uu * np.exp(theta * self.vv)
            xx, yy, zz = row * np.cos(theta), row * np.sin(theta), row * theta
            if self.generator.normal(0, 1) > 0:  # Eq. 4
                x = agent.solution + step_length[idx] * (
                        (self.g_best.solution - agent.solution) * xx * yy * zz
                        + self.g_best.solution
                )
            else:  # Eq. 7
                x = agent.solution + self.generator.random(
                    self.problem.n_dims
                ) * self.ll * beta * (
                                  self.g_best.solution - beta * self.g_best.solution
                          )
            x = cy.correct_solution(self.problem, x)
            n_population.append(x)

        # The predation behavior of sea horses
        pop_child = []
        alpha = (1 - epoch / self.epoch) ** (2 * epoch / self.epoch)
        for idx in range(0, pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            if self.generator.random() >= 0.1:
                x = (
                        alpha * (self.g_best.solution - r1 * n_population[idx])
                        + (1 - alpha) * self.g_best.solution
                )  # Eq. 10
            else:
                x = (1 - alpha) * (
                        n_population[idx] - r1 * self.g_best.solution
                ) + alpha * n_population[
                              idx
                          ]  # Eq. 11
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop_child.append(agent)
        pop_child = self.population.evaluate(pop_child, self.mode)
        pop_child = cy.sort_agents(pop_child, self.problem.sense)  # Sorted population

        # The reproductive behavior of sea horses
        dads = pop_child[: int(pop_size / 2)]
        moms = pop_child[int(pop_size / 2):]
        pop_offspring = []
        for kdx in range(0, int(pop_size / 2)):
            r3 = self.generator.random()
            x = r3 * dads[kdx].solution + (1 - r3) * moms[kdx].solution  # Eq. 13
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop_offspring.append(agent)
        pop_offspring = self.population.evaluate(pop_offspring, self.mode)
        # Sea horses selection
        self.population = self.population.spawn(cy.sort_agents(pop_child + pop_offspring, self.problem.sense)[:pop_size])

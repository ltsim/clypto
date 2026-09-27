#!/usr/bin/env python
# Created by "Thieu" at 22:07, 11/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevVCS(cy.Optimizer):
    """
    The developed version: Virus Colony Search (VCS)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2015.11.004

    Notes:
        + In Immune response process, updates the whole position instead of updating each variable in position

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): (0, 1.0) -> better [0.2, 0.5], Percentage of the number of the best will keep, default = 0.5
        + sigma (float): (0, 5.0) -> better [0.1, 2.0], Weight factor

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.bio_based import VCS    >>> import numpy as np
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
    >>> model = VCS.DevVCS(epoch=1000, pop_size=50, lamda = 0.5, sigma = 0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            lamda: float = 0.5,
            sigma: float = 1.5,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            lamda (float): Percentage of the number of the best will keep, default = 0.5
            sigma (float): Weight factor, default = 1.5
        """
        super().__init__(parameters=["epoch", "pop_size", "lamda", "sigma"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.lamda = cy.validator(float, lamda, (0, 1.0), "lamda")
        self.sigma = cy.validator(float, sigma, (0, 5.0), "sigma")
        self.n_best = int(self.lamda * self.population.size())

    def calculate_xmean__(self, pop):
        """
        Calculate the mean position of list of solutions (population)

        Args:
            pop (list): List of solutions (population)

        Returns:
            list: Mean position
        """
        ## Calculate the weighted mean of the λ best individuals by
        pop = cy.sort_agents(pop, self.problem.sense)
        pos_list = [agent.solution for agent in pop[: self.n_best]]
        factor_down = self.n_best * np.log1p(self.n_best + 1) - np.log1p(
            np.prod(range(1, self.n_best + 1))
        )
        weight = np.log1p(self.n_best + 1) / factor_down
        weight = weight / self.n_best
        x_mean = weight * np.sum(pos_list, axis=0)
        return x_mean

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Viruses diffusion
        pop = []
        for idx in range(0, pop_size):
            sigma = (np.log1p(epoch + 1) / self.epoch) * (
                    self.population[idx].solution - self.g_best.solution
            )
            gauss = self.generator.normal(
                self.generator.normal(self.g_best.solution, np.abs(sigma))
            )
            pos_new = (
                    gauss
                    + self.generator.uniform() * self.g_best.solution
                    - self.generator.uniform() * self.population[idx].solution
            )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.population.evaluate(pop, self.mode)
            self.population = self.population.greedy(pop)
        ## Host cells infection
        x_mean = self.calculate_xmean__(self.population)
        sigma = self.sigma * (1 - epoch / self.epoch)
        pop = []
        for idx in range(0, pop_size):
            ## Basic / simple version, not the original version in the paper
            pos_new = x_mean + sigma * self.generator.normal(0, 1, self.problem.n_dims)
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.population.evaluate(pop, self.mode)
            self.population = self.population.greedy(pop)
        ## Calculate the weighted mean of the λ best individuals by
        self.population = self.population.sort()
        ## Immune response
        pop = []
        for idx in range(0, pop_size):
            pr = (self.problem.n_dims - idx + 1) / self.problem.n_dims
            id1, id2 = self.generator.choice(
                list(set(range(0, pop_size)) - {idx}), 2, replace=False
            )
            temp = (
                    self.population[id1].solution
                    - (self.population[id2].solution - self.population[idx].solution)
                    * self.generator.uniform()
            )
            condition = self.generator.random(self.problem.n_dims) < pr
            pos_new = np.where(condition, self.population[idx].solution, temp)
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.population.evaluate(pop, self.mode)
            self.population = self.population.greedy(pop)

#!/usr/bin/env python
# Created by "Thieu" at 22:07, 11/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.bio_based.VCS.DevVCS cimport DevVCS


cdef object repair(object problem, object generator, object solution):
    """MEALPY's VCS repair, kept as published: where the clipped value is non-zero the raw value stays,
    elsewhere a uniform draw; then the problem's own correction."""
    condition = np.clip(solution, problem.bounds.low, problem.bounds.up)
    rand_pos = generator.uniform(problem.bounds.low, problem.bounds.up)
    return problem.correct_solution(np.where(condition, solution, rand_pos))


cdef class OriginalVCS(DevVCS):
    """
    The original version of: Virus Colony Search (VCS)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2015.11.004

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): (0, 1.0) -> better [0.2, 0.5], Percentage of the number of the best will keep, default = 0.5
        + sigma (float): (0, 5.0) -> better [0.1, 2.0], Weight factor

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.bio_based import VCS    >>> import numpy as np
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
    >>> model = VCS.OriginalVCS(epoch=1000, pop_size=50, lamda = 0.5, sigma = 0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Li, M.D., Zhao, H., Weng, X.W. and Han, T., 2016. A novel nature-inspired algorithm
    for optimization: Virus colony search. Advances in Engineering Software, 92, pp.65-88.
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
            lamda (float): Number of the best will keep, default = 0.5
            sigma (float): Weight factor, default = 1.5
        """
        super().__init__(epoch, pop_size, lamda, sigma, **kwargs)
        self.population = cy.population(pop_size, range=[5, 10000])

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Viruses diffusion
        pop = []
        for idx in range(0, pop_size):
            sigma = (np.log1p(epoch) / self.epoch) * (
                    self.population[idx].solution - self.g_best.solution
            )
            gauss = np.array(
                [
                    self.generator.normal(self.g_best.solution[j], np.abs(sigma[j]))
                    for j in range(0, self.problem.n_dims)
                ]
            )
            x = (
                    gauss
                    + self.generator.uniform() * self.g_best.solution
                    - self.generator.uniform() * self.population[idx].solution
            )
            x = repair(self.problem, self.generator, x)
            agent = self.population.create_agent(x)
            pop.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop = self.population.evaluate(pop, self.mode)
            self.population = self.population.greedy(pop)
        ## Host cells infection
        x_mean = self.calculate_xmean__(self.population)
        sigma = self.sigma * (1 - epoch / self.epoch)
        pop = []
        for idx in range(0, pop_size):
            ## Basic / simple version, not the original version in the paper
            x = x_mean + sigma * self.generator.normal(0, 1, self.problem.n_dims)
            x = repair(self.problem, self.generator, x)
            agent = self.population.create_agent(x)
            pop.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop = self.population.evaluate(pop, self.mode)
            self.population = self.population.greedy(pop)
        ## Immune response
        pop = []
        for idx, agent in enumerate(self.population.toarray()):
            pr = (self.problem.n_dims - idx + 1) / self.problem.n_dims
            x = agent.solution.copy()
            for j in range(0, self.problem.n_dims):
                if self.generator.uniform() > pr:
                    id1, id2 = self.generator.choice(
                        list(set(range(0, pop_size)) - {idx}), 2, replace=False
                    )
                    x[j] = (
                            self.population[id1].solution[j]
                            - (self.population[id2].solution[j] - agent.solution[j])
                            * self.generator.uniform()
                    )
            x = repair(self.problem, self.generator, x)
            child = self.population.create_agent(x)
            pop.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop = self.population.evaluate(pop, self.mode)
            self.population = self.population.greedy(pop)

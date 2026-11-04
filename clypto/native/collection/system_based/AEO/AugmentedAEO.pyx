#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class AugmentedAEO(cy.Optimizer):
    """
    The original version of: Augmented Artificial Ecosystem Optimization (AAEO)

    Notes:
        + Used linear weight factor reduce from 2 to 0 through time
        + Applied Levy-flight technique and the global best solution

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.system_based import AEO    >>> import numpy as np
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
    >>> model = AEO.AugmentedAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Van Thieu, N., Barma, S. D., Van Lam, T., Kisi, O., & Mahesha, A. (2022). Groundwater level modeling
    using Augmented Artificial Ecosystem Optimization. Journal of Hydrology, 129034.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Production - Update the worst agent
        # Eq. 2, 3, 1
        wf = 2 * (1 - epoch / self.epoch)  # Weight factor
        a = (1.0 - epoch / self.epoch) * self.generator.random()
        x1 = (1 - a) * self.population[-1].solution + a * self.generator.uniform(
            self.problem.bounds.low, self.problem.bounds.up
        )
        x = cy.correct_solution(self.problem, x1)
        agent = self.population.generate_agent(x)
        self.population[-1] = agent
        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, pop_size - 1):
            if self.generator.random() < 0.5:
                rand = self.generator.random()
                # Eq. 4, 5, 6
                c = (
                        0.5
                        * self.generator.normal(0, 1)
                        / np.abs(self.generator.normal(0, 1))
                )  # Consumption factor
                j = 1 if idx == 0 else self.generator.integers(0, idx)
                ### Herbivore
                if rand < 1.0 / 3:
                    x = self.population[idx].solution + wf * c * (
                            self.population[idx].solution - self.population[0].solution
                    )  # Eq. 6
                ### Omnivore
                elif 1.0 / 3 <= rand <= 2.0 / 3:
                    x = self.population[idx].solution + wf * c * (
                            self.population[idx].solution - self.population[j].solution
                    )  # Eq. 7
                ### Carnivore
                else:
                    r2 = self.generator.uniform()
                    x = self.population[idx].solution + wf * c * (
                            r2 * (self.population[idx].solution - self.population[0].solution)
                            + (1 - r2) * (self.population[idx].solution - self.population[j].solution)
                    )
            else:
                x = self.population[idx].solution + cy.levy_flight(self.generator, 1.0, 0.001, size=None, case=-1) * (1.0 / np.sqrt(epoch)) * np.sign(self.generator.random() - 0.5) * (
                                  self.population[idx].solution - self.g_best.solution
                          )
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop_new.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population[:-1] = cy.greedy_agents(self.population[:-1], pop_new, self.problem.sense)
        ## find current best used in decomposition
        best = cy.duplicate_agent(self.population.sort()[0])
        ## Decomposition
        ### Eq. 10, 11, 12, 9   idx, pop, g_best, local_best
        pop_child = []
        for idx, agent in enumerate(self.population.toarray()):
            if self.generator.random() < 0.5:
                x = best.solution + self.generator.normal(
                    0, 1, self.problem.n_dims
                ) * (best.solution - agent.solution)
            else:
                beta = self.generator.uniform(0.01, 1.0)
                x = best.solution + cy.levy_flight(self.generator, beta=beta, multiplier=0.01, size=self.problem.n_dims, case=0) * (best.solution - agent.solution)
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            pop_child.append(child)
        self.population = self.population.greedy(self.population.evaluate(pop_child, self.mode), self.mode)

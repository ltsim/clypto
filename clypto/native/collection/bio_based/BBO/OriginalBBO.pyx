#!/usr/bin/env python
# Created by "Thieu" at 12:24, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalBBO(cy.Optimizer):
    """
    The original version of: Biogeography-Based Optimization (BBO)

    Links:
        1. https://ieeexplore.ieee.org/abstract/document/4475427

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_m (float): (0, 1) -> better [0.01, 0.2], Mutation probability
        + n_elites (int): (2, pop_size/2) -> better [2, 5], Number of elites will be keep for next generation

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.bio_based import BBO    >>> import numpy as np
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
    >>> model = BBO.OriginalBBO(epoch=1000, pop_size=50, p_m=0.01, n_elites=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Simon, D., 2008. Biogeography-based optimization. IEEE transactions on evolutionary computation, 12(6), pp.702-713.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            p_m: float = 0.01,
            n_elites: int = 2,
            **kwargs: object
    ) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 100
            p_m: Mutation probability, default=0.01
            n_elites: Number of elites will be keep for next generation, default=2
        """
        super().__init__(parameters=["epoch", "pop_size", "p_m", "n_elites"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.p_m = cy.validator(float, p_m, (0.0, 1.0), "p_m")
        self.n_elites = cy.validator(int, n_elites, [2, int(self.population.size() / 2)], "n_elites")
        self.mu = (self.population.size() + 1 - np.array(range(1, self.population.size() + 1))) / (
                self.population.size() + 1
        )
        self.mr = 1 - self.mu

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch: The current iteration
        """
        pop_size = self.population.size()
        ranked = self.population.sort()
        pop_elites = [cy.duplicate_agent(agent) for agent in ranked[:self.n_elites]]
        pop = []
        for idx, agent in enumerate(self.population.toarray()):
            # Probabilistic migration to the i-th position
            x = agent.solution.copy()
            for j in range(self.problem.n_dims):
                if self.generator.random() < self.mr[idx]:  # Should we immigrate?
                    # Pick a position from which to emigrate (roulette wheel selection)
                    random_number = self.generator.random() * np.sum(self.mu)
                    select = self.mu[0]
                    select_index = 0
                    while (random_number > select) and (
                            select_index < pop_size - 1
                    ):
                        select_index += 1
                        select += self.mu[select_index]
                    # this is the migration step
                    x[j] = self.population[select_index].solution[j]
            noise = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            condition = self.generator.random(self.problem.n_dims) < self.p_m
            x = np.where(condition, noise, x)
            x = cy.correct_solution(self.problem, x)
            agent_new = self.population.create_agent(x)
            pop.append(agent_new)
            if self.mode == "sequential":
                agent_new.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent_new, sense=self.problem.sense)
        if self.mode != "sequential":
            pop = self.population.evaluate(pop, self.mode)
            self.population = self.population.greedy(pop)
        # replace the solutions with their new migrated and mutated versions then Merge Populations
        self.population = self.population.spawn(cy.sort_agents(self.population + pop_elites, self.problem.sense)[:pop_size])

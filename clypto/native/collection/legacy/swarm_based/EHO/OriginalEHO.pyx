#!/usr/bin/env python
# Created by "Thieu" at 18:41, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalEHO(cy.Optimizer):
    """
    The original version of: Elephant Herding Optimization (EHO)

    Links:
        1. https://doi.org/10.1109/ISCBI.2015.8

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.3, 0.8], a factor that determines the influence of the best in each clan, default=0.5
        + beta (float): [0.3, 0.8], a factor that determines the influence of the x_center, default=0.5
        + n_clans (int): [3, 10], the number of clans, default=5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import EHO    >>> import numpy as np
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
    >>> model = EHO.OriginalEHO(epoch=1000, pop_size=50, alpha = 0.5, beta = 0.5, n_clans = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2015, December. Elephant herding optimization.
    In 2015 3rd international symposium on computational and business intelligence (ISCBI) (pp. 1-5). IEEE.
    """

    cdef public double alpha
    cdef public double beta
    cdef public int n_clans

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            alpha: float = 0.5,
            beta: float = 0.5,
            n_clans: int = 5,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): a factor that determines the influence of the best in each clan, default=0.5
            beta (float): a factor that determines the influence of the x_center, default=0.5
            n_clans (int): the number of clans, default=5
        """
        super().__init__(parameters=["epoch", "pop_size", "alpha", "beta", "n_clans"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.alpha = cy.validator(float, alpha, (0, 3.0), "alpha")
        self.beta = cy.validator(float, beta, (0, 1.0), "beta")
        self.n_clans = cy.validator(int, n_clans, [2, int(self.population.size() / 5)], "n_clans")
        self.n_individuals = int(self.population.size() / self.n_clans)

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        self.pop_group = cy.split_groups(self.population, self.n_clans, self.n_individuals)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Clan updating operator
        pop_new = []
        for idx in range(0, pop_size):
            clan_idx = int(idx / self.n_individuals)
            pos_clan_idx = int(idx % self.n_individuals)
            if (
                    pos_clan_idx == 0
            ):  # The best in clan, because all clans are sorted based on fitness
                center = np.mean(
                    np.array([agent.solution for agent in self.pop_group[clan_idx]]),
                    axis=0,
                )
                pos_new = self.beta * center
            else:
                pos_new = self.pop_group[clan_idx][
                              pos_clan_idx
                          ].solution + self.alpha * self.generator.random() * (
                                  self.pop_group[clan_idx][0].solution
                                  - self.pop_group[clan_idx][pos_clan_idx].solution
                          )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        self.pop_group = cy.split_groups(self.population, self.n_clans, self.n_individuals)
        # Separating operator
        for idx in range(0, self.n_clans):
            self.pop_group[idx] = cy.sort_agents(self.pop_group[idx], self.problem.sense)
            self.pop_group[idx][-1] = self.population.generate_agent()
        self.population = [agent for pack in self.pop_group for agent in pack]

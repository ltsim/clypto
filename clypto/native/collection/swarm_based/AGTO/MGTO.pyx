#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class MGTO(cy.Optimizer):
    """
    The original version of: Modified Gorilla Troops Optimization (mGTO)

    Notes (parameters):
        1. pp (float): the probability of transition in exploration phase (p in the paper), default = 0.03

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import AGTO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "sense": "min",
    >>> }
    >>>
    >>> model = AGTO.MGTO(epoch=1000, pop_size=50, pp=0.03)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mostafa, R. R., Gaheen, M. A., Abd ElAziz, M., Al-Betar, M. A., & Ewees, A. A. (2023). An improved gorilla
    troops optimizer for global optimization problems and feature selection. Knowledge-Based Systems, 110462.
    """

    cdef public double pp

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            pp: float = 0.03,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pp (float): the probability of transition in exploration phase (p in the paper), default = 0.03
        """
        super().__init__(parameters=["epoch", "pop_size", "pp"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=cy.Population)
        self.pp = cy.validator(float, pp, (0, 1), "p1")  # p in the paper

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        F = 1 + np.cos(2 * self.generator.random())
        C = F * (1 - epoch / self.epoch)
        L = C * self.generator.choice([-1, 1])

        ## Elite opposition-based learning
        pos_list = np.array([agent.solution for agent in self.population])
        d_lb, d_ub = np.min(pos_list, axis=0), np.max(pos_list, axis=0)
        pos_list = d_lb + d_ub - pos_list
        pop_new = []
        for idx in range(0, pop_size):
            x = cy.reset_solution(self.problem, self.generator, pos_list[idx])
            agent = self.population.create_agent(x)
            pop_new.append(agent)
            if self.mode == "sequential":
                pop_new[-1].evaluate(self.problem)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
        self.population = self.population.spawn(pop_new)
        ranked = self.population.sort()
        self.g_best = ranked[0]

        ## Exploration
        pop_new = []
        for idx in range(0, pop_size):
            if self.generator.random() < self.pp:
                x = self.problem.generate_solution()
            else:
                if self.generator.random() >= 0.5:
                    rand_idx = self.generator.integers(0, pop_size)
                    x = (self.generator.random() - C) * self.population[
                        rand_idx
                    ].solution + L * self.generator.uniform(-C, C) * self.population[
                                  idx
                              ].solution
                else:
                    id1, id2 = self.generator.choice(
                        list(set(range(0, pop_size)) - {idx}), 2, replace=False
                    )
                    x = (
                            self.population[idx].solution
                            - L * (L * self.population[idx].solution - self.population[id1].solution)
                            + self.generator.random()
                            * (self.population[idx].solution - self.population[id2].solution)
                    )
            x = cy.reset_solution(self.problem, self.generator, x)
            agent = self.population.create_agent(x)
            pop_new.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        ranked = self.population.sort()
        self.g_best = ranked[0]

        pos_list = np.array([agent.solution for agent in self.population])
        ## Exploitation
        pop_new = []
        for idx, agent in enumerate(self.population.toarray()):
            if np.abs(C) >= 1:
                g = self.generator.choice([-0.5, 2.0])
                M = (np.abs(np.mean(pos_list, axis=0)) ** g) ** (1.0 / g)
                # print(M)
                p = self.generator.uniform(0, 1, self.problem.n_dims)
                x = (
                        L
                        * M
                        * (agent.solution - self.g_best.solution)
                        * (0.01 * np.tan(np.pi * (p - 0.5)))
                )
            else:
                Q = 2 * self.generator.random() - 1
                v = self.generator.uniform(0, 1)
                x = self.g_best.solution - Q * (
                        self.g_best.solution - agent.solution
                ) * np.tan(v * np.pi / 2)
            x = cy.reset_solution(self.problem, self.generator, x)
            child = self.population.create_agent(x)
            pop_new.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

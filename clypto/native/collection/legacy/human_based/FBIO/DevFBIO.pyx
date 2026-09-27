#!/usr/bin/env python
# Created by "Thieu" at 08:57, 14/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevFBIO(cy.Optimizer):
    """
    The developed : Forensic-Based Investigation Optimization (FBIO)

    Notes:
        + Third loop is removed, the flowand a few equations is improved

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import FBIO    >>> import numpy as np
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
    >>> model = FBIO.DevFBIO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
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

    def probability__(
            self, list_fitness=None
    ):  # Eq.(3) in FBI Inspired Meta-Optimization
        max1 = np.max(list_fitness)
        min1 = np.min(list_fitness)
        return (max1 - list_fitness) / (max1 - min1 + self.EPSILON)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Investigation team - team A
        # Step A1
        pop_new = []
        for idx in range(0, pop_size):
            n_change = self.generator.integers(0, self.problem.n_dims)
            nb1, nb2 = self.generator.choice(
                list(set(range(0, pop_size)) - {idx}), 2, replace=False
            )
            # Eq.(2) in FBI Inspired Meta - Optimization
            pos_a = self.population[idx].solution.copy()
            pos_a[n_change] = self.population[idx].solution[
                                  n_change
                              ] + self.generator.normal() * (
                                      self.population[idx].solution[n_change]
                                      - (self.population[nb1].solution[n_change] + self.population[nb2].solution[n_change])
                                      / 2
                              )
            pos_a = self.population.correct_solution(pos_a)
            agent = self.population.create_agent(pos_a)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        list_fitness = np.array([agent.fitness for agent in self.population])
        prob = self.probability__(list_fitness)

        # Step A2
        pop_child = []
        for idx in range(0, pop_size):
            if self.generator.random() > prob[idx]:
                r1, r2, r3 = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx}), 3, replace=False
                )
                ## Remove third loop here, the condition also not good, need to remove also. No need Rnd variable
                temp = (
                        self.g_best.solution
                        + self.population[r1].solution
                        + self.generator.uniform()
                        * (self.population[r2].solution - self.population[r3].solution)
                )
                condition = self.generator.random(self.problem.n_dims) < 0.5
                pos_new = np.where(condition, temp, self.population[idx].solution)
            else:
                pos_new = self.problem.generate_solution()
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.population.evaluate(pop_child, self.mode)
            self.population = cy.greedy_agents(pop_child, self.population, self.problem.sense)
        ## Persuing team - team B
        ## Step B1
        pop_new = []
        for idx in range(0, pop_size):
            ### Remove third loop here also
            ### Eq.(6) in FBI Inspired Meta-Optimization
            pos_b = self.generator.uniform(0, 1, self.problem.n_dims) * self.population[
                idx
            ].solution + self.generator.uniform(0, 1, self.problem.n_dims) * (
                            self.g_best.solution - self.population[idx].solution
                    )
            pos_b = self.population.correct_solution(pos_b)
            agent = self.population.create_agent(pos_b)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        ## Step B2
        pop_child = []
        for idx in range(0, pop_size):
            rr = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
            if cy.is_better(self.population[idx], self.population[rr], self.problem.sense):
                ## Eq.(7) in FBI Inspired Meta-Optimization
                pos_b = (
                        self.population[idx].solution
                        + self.generator.uniform(0, 1, self.problem.n_dims)
                        * (self.population[rr].solution - self.population[idx].solution)
                        + self.generator.uniform()
                        * (self.g_best.solution - self.population[rr].solution)
                )
            else:
                ## Eq.(8) in FBI Inspired Meta-Optimization
                pos_b = (
                        self.population[idx].solution
                        + self.generator.uniform(0, 1, self.problem.n_dims)
                        * (self.population[idx].solution - self.population[rr].solution)
                        + self.generator.uniform()
                        * (self.g_best.solution - self.population[idx].solution)
                )
            pos_b = self.population.correct_solution(pos_b)
            agent = self.population.create_agent(pos_b)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.population.evaluate(pop_child, self.mode)
            self.population = cy.greedy_agents(pop_child, self.population, self.problem.sense)

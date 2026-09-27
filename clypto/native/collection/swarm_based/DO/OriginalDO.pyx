#!/usr/bin/env python
# Created by "Thieu" at 04:43, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalDO(cy.Optimizer):
    """
    The original version of: Dragonfly Optimization (DO)

    Links:
        1. https://link.springer.com/article/10.1007/s00521-015-1920-1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import DO    >>> import numpy as np
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
    >>> model = DO.OriginalDO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2016. Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective,
    discrete, and multi-objective problems. Neural computing and applications, 27(4), pp.1053-1073.
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

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        self.pop_delta = self.population.generate(pop_size)
        # Initial radius of dragonflies' neighborhoods
        self.radius = (self.problem.bounds.up - self.problem.bounds.low) / 10
        self.delta_max = (self.problem.bounds.up - self.problem.bounds.low) / 10

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ranked = self.population.sort()
        (self.g_best,) = [cy.duplicate_agent(agent) for agent in ranked[:1]]
        (self.g_worst,) = [cy.duplicate_agent(agent) for agent in ranked[::-1][:1]]

        r = (self.problem.bounds.up - self.problem.bounds.low) / 4 + (
                (self.problem.bounds.up - self.problem.bounds.low) * (2 * epoch / self.epoch)
        )
        w = 0.9 - epoch * ((0.9 - 0.4) / self.epoch)
        my_c = 0.1 - epoch * ((0.1 - 0) / (self.epoch / 2))
        my_c = 0 if my_c < 0 else my_c

        s = 2 * self.generator.random() * my_c  # Seperation weight
        a = 2 * self.generator.random() * my_c  # Alignment weight
        c = 2 * self.generator.random() * my_c  # Cohesion weight
        f = 2 * self.generator.random()  # Food attraction weight
        e = my_c  # Enemy distraction weight

        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        pop_delta_new = []
        for idx, agent in enumerate(self.population.toarray()):
            pos_neighbours = []
            pos_neighbours_delta = []
            neighbours_num = 0
            # Find the neighbouring solutions
            for j in range(0, pop_size):
                dist = np.abs(agent.solution - self.population[j].solution)
                if np.all(dist <= r) and np.all(dist != 0):
                    neighbours_num += 1
                    pos_neighbours.append(self.population[j].solution)
                    pos_neighbours_delta.append(self.pop_delta[j].solution)
            pos_neighbours = np.array(pos_neighbours)
            pos_neighbours_delta = np.array(pos_neighbours_delta)

            # Separation: Eq 3.1, Alignment: Eq 3.2, Cohesion: Eq 3.3
            if neighbours_num > 1:
                S = (
                        np.sum(pos_neighbours, axis=0)
                        - neighbours_num * agent.solution
                )
                A = np.sum(pos_neighbours_delta, axis=0) / neighbours_num
                C_temp = np.sum(pos_neighbours, axis=0) / neighbours_num
            else:
                S = np.zeros(self.problem.n_dims)
                A = self.pop_delta[idx].solution.copy()
                C_temp = agent.solution.copy()
            C = C_temp - agent.solution

            # Attraction to food: Eq 3.4
            dist_to_food = np.abs(agent.solution - self.g_best.solution)
            if np.all(dist_to_food <= r):
                F = self.g_best.solution - agent.solution
            else:
                F = np.zeros(self.problem.n_dims)

            # Distraction from enemy: Eq 3.5
            dist_to_enemy = np.abs(agent.solution - self.g_worst.solution)
            if np.all(dist_to_enemy <= r):
                enemy = self.g_worst.solution + agent.solution
            else:
                enemy = np.zeros(self.problem.n_dims)

            x = agent.solution.copy().astype(float)
            pos_delta_new = self.pop_delta[idx].solution.copy().astype(float)
            if np.any(dist_to_food > r):
                if neighbours_num > 1:
                    temp = (
                            w * self.pop_delta[idx].solution
                            + self.generator.uniform(0, 1, self.problem.n_dims) * A
                            + self.generator.uniform(0, 1, self.problem.n_dims) * C
                            + self.generator.uniform(0, 1, self.problem.n_dims) * S
                    )
                    temp = np.clip(temp, -1 * self.delta_max, self.delta_max)
                    pos_delta_new = temp.copy()
                    x += temp
                else:  # Eq. 3.8
                    x += (
                            cy.levy_flight(self.generator, beta=1.5, multiplier=0.01, size=None, case=-1)
                            * agent.solution
                    )
                    pos_delta_new = np.zeros(self.problem.n_dims)
            else:
                # Eq. 3.6
                temp = (a * A + c * C + s * S + f * F + e * enemy) + w * self.pop_delta[
                    idx
                ].solution
                temp = np.clip(temp, -1 * self.delta_max, self.delta_max)
                pos_delta_new = temp
                x += temp

            # Amend solution
            x = cy.correct_solution(self.problem, x)
            pos_delta_new = cy.correct_solution(self.problem, pos_delta_new)
            child = self.population.create_agent(x)
            agent_delta = self.population.create_agent(pos_delta_new)
            n_population.append(child)
            pop_delta_new.append(agent_delta)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                agent_delta.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
                self.pop_delta[idx] = cy.get_better_agent(agent_delta, self.pop_delta[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            pop_delta_new = self.population.evaluate(pop_delta_new, self.mode)
            self.population = self.population.greedy(n_population)
            self.pop_delta = cy.greedy_agents(self.pop_delta, pop_delta_new, self.problem.sense)

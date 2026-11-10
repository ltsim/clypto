#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class ImprovedNMRA(cy.Optimizer):
    """
    The developed version of: Improved Naked Mole-Rat Algorithm (I-NMRA)

    Notes:
        + Use mutation probability idea
        + Use crossover operator
        + Use Levy-flight technique

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.5, 0.95], probability of breeding, default = 0.75
        + pm (float): [0.01, 0.1], probability of mutation, default = 0.01

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import NMRA    >>> import numpy as np
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
    >>> model = NMRA.ImprovedNMRA(epoch=1000, pop_size=50, pb = 0.75, pm = 0.01)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    cdef public double pb
    cdef public double pm

    def __init__(self, epoch=10000, pop_size=100, pb=0.75, pm=0.01, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pb (float): breeding probability, default = 0.75
            pm (float): probability of mutation, default = 0.01
        """
        super().__init__(parameters=["epoch", "pop_size", "pb", "pm"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.pb = cy.validator(float, pb, (0, 1.0), "pb")
        self.pm = cy.validator(float, pm, (0, 1.0), "pm")
        self.size_b = int(self.population.size() / 5)

    def crossover_random__(self, pop, g_best):
        pop_size = self.population.size()
        start_point = self.generator.integers(0, self.problem.n_dims / 2)
        id1 = start_point
        id2 = int(start_point + self.problem.n_dims / 3)
        id3 = int(self.problem.n_dims)

        partner = pop[self.generator.integers(0, pop_size)].solution
        new_temp = g_best.solution.copy()
        new_temp[0:id1] = g_best.solution[0:id1]
        new_temp[id1:id2] = partner[id1:id2]
        new_temp[id2:id3] = g_best.solution[id2:id3]
        return new_temp

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            # Exploration
            if idx < self.size_b:  # breeding operators
                if self.generator.uniform() < self.pb:
                    x = agent.solution + self.generator.normal(
                        0, 1, self.problem.n_dims
                    ) * (self.g_best.solution - agent.solution)
                else:
                    levy_step = cy.levy_flight(self.generator, beta=1, multiplier=0.001, size=None, case=-1)
                    x = agent.solution + 1.0 / np.sqrt(epoch) * np.sign(
                        self.generator.random() - 0.5
                    ) * levy_step * (agent.solution - self.g_best.solution)
            # Exploitation
            else:  # working operators
                if self.generator.uniform() < 0.5:
                    t1, t2 = self.generator.choice(
                        range(self.size_b, pop_size), 2, replace=False
                    )
                    x = agent.solution + self.generator.normal(
                        0, 1, self.problem.n_dims
                    ) * (self.population[t1].solution - self.population[t2].solution)
                else:
                    x = self.crossover_random__(self.population, self.g_best)
            # Mutation
            temp = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            condition = self.generator.uniform(0, 1, self.problem.n_dims) < self.pm
            x = np.where(condition, temp, x)
            x = cy.correct_solution(self.problem, x)
            child = self.population.generate_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], child, self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

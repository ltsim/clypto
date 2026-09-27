#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalAGTO(cy.Optimizer):
    """
    The original version of: Artificial Gorilla Troops Optimization (AGTO)

    Links:
        1. https://doi.org/10.1002/int.22535
        2. https://www.mathworks.com/matlabcentral/fileexchange/95953-artificial-gorilla-troops-optimizer

    Notes (parameters):
        1. p1 (float): the probability of transition in exploration phase (p in the paper), default = 0.03
        2. p2 (float): the probability of transition in exploitation phase (w in the paper), default = 0.8
        3. beta (float): coefficient in updating equation, should be in [-5.0, 5.0], default = 3.0

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
    >>> model = AGTO.OriginalAGTO(epoch=1000, pop_size=50, p1=0.03, p2=0.8, beta=3.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021). Artificial gorilla troops optimizer: a new
    nature‐inspired metaheuristic algorithm for global optimization problems. International Journal of Intelligent Systems, 36(10), 5887-5958.
    """

    cdef public double beta
    cdef public double p1
    cdef public double p2

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            p1: float = 0.03,
            p2: float = 0.8,
            beta: float = 3.0,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size", "p1", "p2", "beta"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.p1 = cy.validator(float, p1, (0, 1), "p1")  # p in the paper
        self.p2 = cy.validator(float, p2, (0, 1), "p2")  # w in the paper
        self.beta = cy.validator(float, beta, [-10.0, 10.0], "beta")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        a = (np.cos(2 * self.generator.random()) + 1) * (1 - epoch / self.epoch)
        c = a * (2 * self.generator.random() - 1)
        ## Exploration
        pop_new = []
        for idx in range(0, pop_size):
            if self.generator.random() < self.p1:
                x = self.problem.generate_solution()
            else:
                if self.generator.random() >= 0.5:
                    z = self.generator.uniform(-a, a, self.problem.n_dims)
                    rand_idx = self.generator.integers(0, pop_size)
                    x = (self.generator.random() - a) * self.population[
                        rand_idx
                    ].solution + c * z * self.population[idx].solution
                else:
                    id1, id2 = self.generator.choice(
                        list(set(range(0, pop_size)) - {idx}), 2, replace=False
                    )
                    x = (
                            self.population[idx].solution
                            - c * (c * self.population[idx].solution - self.population[id1].solution)
                            + self.generator.random()
                            * (self.population[idx].solution - self.population[id2].solution)
                    )
            x = cy.correct_solution(self.problem, x)
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
            if a >= self.p2:
                g = 2 ** c
                delta = (np.abs(np.mean(pos_list, axis=0)) ** g) ** (1.0 / g)
                x = (
                        c * delta * (agent.solution - self.g_best.solution)
                        + agent.solution
                )
            else:
                if self.generator.random() >= 0.5:
                    h = self.generator.normal(0, 1, self.problem.n_dims)
                else:
                    h = self.generator.normal(0, 1)
                r1 = self.generator.random()
                x = self.g_best.solution - (2 * r1 - 1) * (
                        self.g_best.solution - agent.solution
                ) * (self.beta * h)
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            pop_new.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

#!/usr/bin/env python
# Created by "Thieu" at 17:48, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevDMOA(cy.Optimizer):
    """
    The developed version of: Dwarf Mongoose Optimization Algorithm (DMOA)

    Notes:
        1. Removed the parameter n_baby_sitter
        2. Changed in section # Next Mongoose position
        3. Removed the meaningless variable tau

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import DMOA    >>> import numpy as np
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
    >>> model = DMOA.DevDMOA(epoch=1000, pop_size=50, peep = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    cdef public double peep

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, peep: float = 2, **kwargs: object
    ) -> None:
        super().__init__(parameters=["epoch", "pop_size", "peep"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[10, 10000])
        self.peep = cy.validator(float, peep, [1, 10.0], "peep")

    def initialize_variables(self):
        pop_size = self.population.size()
        self.C = np.zeros(pop_size)
        self.L = np.round(0.6 * self.epoch)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Abandonment Counter
        CF = (1.0 - epoch / self.epoch) ** (2.0 * epoch / self.epoch)
        fit_list = np.array([agent.fitness for agent in self.population])
        mean_cost = np.mean(fit_list)
        fi = np.exp(-fit_list / mean_cost)

        ## Foraging led by Alpha female
        for idx in range(0, pop_size):
            alpha = cy.roulette_wheel(self.generator, self.problem.sense, fi)
            k = self.generator.choice(list(set(range(0, pop_size)) - {idx, alpha}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.population[alpha].solution + phi * (
                    self.population[alpha].solution - self.population[k].solution
            )
            new_pos = cy.correct_solution(self.problem, new_pos)
            agent = self.population.generate_agent(new_pos)
            if cy.is_better(agent, self.population[idx], self.problem.sense):
                self.population[idx] = agent
            else:
                self.C[idx] += 1

        ## Scout group
        SM = np.zeros(pop_size)
        for idx in range(0, pop_size):
            k = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.population[idx].solution + phi * (
                    self.population[idx].solution - self.population[k].solution
            )
            new_pos = cy.correct_solution(self.problem, new_pos)
            agent = self.population.generate_agent(new_pos)
            ## Sleeping mould
            SM[idx] = (agent.fitness - self.population[idx].fitness) / (
                    np.max([agent.fitness, self.population[idx].fitness])
                    + self.EPSILON
            )
            if cy.is_better(agent, self.population[idx], self.problem.sense):
                self.population[idx] = agent
            else:
                self.C[idx] += 1

        ## Baby sitters
        for idx, agent in enumerate(self.population.toarray()):
            if self.C[idx] >= self.L:
                self.population[idx] = self.population.generate_agent()
                self.C[idx] = 0

        ## Next Mongoose position
        new_tau = np.mean(SM)
        for idx, agent in enumerate(self.population.toarray()):
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            if new_tau > SM[idx]:
                new_pos = self.g_best.solution - CF * phi * (
                        self.g_best.solution - SM[idx] * agent.solution
                )
            else:
                new_pos = agent.solution + CF * phi * (
                        self.g_best.solution - SM[idx] * agent.solution
                )
            new_pos = cy.correct_solution(self.problem, new_pos)
            child = self.population.generate_agent(new_pos)
            if cy.is_better(child, agent, self.problem.sense):
                self.population[idx] = child

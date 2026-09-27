#!/usr/bin/env python
# Created by "Thieu" at 17:48, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalDMOA(cy.Optimizer):
    """
    The original version of: Dwarf Mongoose Optimization Algorithm (DMOA)

    Links:
        1. https://doi.org/10.1016/j.cma.2022.114570
        2. https://www.mathworks.com/matlabcentral/fileexchange/105125-dwarf-mongoose-optimization-algorithm

    Notes:
        1. The Matlab code differs slightly from the original paper
        2. There are some parameters and equations in the Matlab code that don't seem to have any meaningful purpose.
        3. The algorithm seems to be weak on solving several problems.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import DMOA    >>> import numpy as np
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
    >>> model = DMOA.OriginalDMOA(epoch=1000, pop_size=50, n_baby_sitter = 3, peep = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Agushaka, J. O., Ezugwu, A. E., & Abualigah, L. (2022). Dwarf mongoose optimization algorithm.
    Computer methods in applied mechanics and engineering, 391, 114570.
    """

    cdef public int n_baby_sitter
    cdef public double peep

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            n_baby_sitter: int = 3,
            peep: float = 2,
            **kwargs: object
    ) -> None:
        super().__init__(parameters=["epoch", "pop_size", "n_baby_sitter", "peep"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[10, 10000])
        self.n_baby_sitter = cy.validator(int, n_baby_sitter, [2, 10], "n_baby_sitter")
        self.peep = cy.validator(float, peep, [1, 10.0], "peep")
        self.n_scout = self.population.size() - self.n_baby_sitter

    def initialize_variables(self):
        pop_size = self.population.size()
        self.C = np.zeros(pop_size)
        self.tau = -np.inf
        self.L = np.round(0.6 * self.problem.n_dims * self.n_baby_sitter)

    def evolve(self, epoch):
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
        for idx in range(0, pop_size):
            alpha = cy.roulette_wheel(self.generator, self.problem.sense, fi)
            k = self.generator.choice(list(set(range(0, pop_size)) - {idx, alpha}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.population[alpha].solution + phi * (
                    self.population[alpha].solution - self.population[k].solution
            )
            new_pos = self.population.correct_solution(new_pos)
            agent = self.population.generate_agent(new_pos)
            if cy.is_better(agent, self.population[idx], self.problem.sense):
                self.population[idx] = agent
            else:
                self.C[idx] += 1
        SM = np.zeros(pop_size)
        for idx in range(0, pop_size):
            k = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.population[idx].solution + phi * (
                    self.population[idx].solution - self.population[k].solution
            )
            new_pos = self.population.correct_solution(new_pos)
            agent = self.population.generate_agent(new_pos)
            ## Sleeping mould
            SM[idx] = (agent.fitness - self.population[idx].fitness) / np.max(
                [agent.fitness, self.population[idx].fitness]
            )
            if cy.is_better(agent, self.population[idx], self.problem.sense):
                self.population[idx] = agent
            else:
                self.C[idx] += 1
        ## Baby sitters
        for idx in range(0, self.n_baby_sitter):
            if self.C[idx] >= self.L:
                self.population[idx] = self.population.generate_agent()
                self.C[idx] = 0
        ## Next Mongoose position
        new_tau = np.mean(SM)
        for idx in range(0, pop_size):
            M = SM[idx] * self.population[idx].solution / self.population[idx].solution
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            if new_tau > self.tau:
                new_pos = self.population[
                              idx
                          ].solution - CF * phi * self.generator.random() * (
                                  self.population[idx].solution - M
                          )
            else:
                new_pos = self.population[
                              idx
                          ].solution + CF * phi * self.generator.random() * (
                                  self.population[idx].solution - M
                          )
            self.tau = new_tau
            new_pos = self.population.correct_solution(new_pos)
            self.population[idx] = self.population.generate_agent(new_pos)

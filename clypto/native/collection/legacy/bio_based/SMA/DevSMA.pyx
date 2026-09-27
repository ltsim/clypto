#!/usr/bin/env python
# Created by "Thieu" at 20:22, 12/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevSMA(cy.Optimizer):
    """
    The developed version: Slime Mould Algorithm (SMA)

    Notes:
        + Selected 2 unique and random solution to create new solution (not to create variable)
        + Check bound and compare old position with new position to get the best one

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_t (float): (0, 1.0) -> better [0.01, 0.1], probability threshold (z in the paper)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.bio_based import SMA    >>> import numpy as np
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
    >>> model = SMA.DevSMA(epoch=1000, pop_size=50, p_t = 0.03)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            p_t: float = 0.03,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_t (float): probability threshold (z in the paper), default = 0.03
        """
        super().__init__(parameters=["epoch", "pop_size", "p_t"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.p_t = cy.validator(float, p_t, (0, 1.0), "p_t")

    def initialize_variables(self):
        pop_size = self.population.size()
        self.weights = np.zeros((pop_size, self.problem.n_dims))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # plus eps to avoid denominator zero
        ss = self.g_best.fitness - self.population[-1].fitness + self.EPSILON
        # calculate the fitness weight of each slime mold
        for idx in range(0, pop_size):
            # Eq.(2.5)
            if idx <= int(pop_size / 2):
                self.weights[idx] = 1 + self.generator.uniform(
                    0, 1, self.problem.n_dims
                ) * np.log10(
                    (self.g_best.fitness - self.population[idx].fitness) / ss + 1
                )
            else:
                self.weights[idx] = 1 - self.generator.uniform(
                    0, 1, self.problem.n_dims
                ) * np.log10(
                    (self.g_best.fitness - self.population[idx].fitness) / ss + 1
                )
        a = np.arctanh(1 - epoch / self.epoch)  # Eq.(2.4)
        b = 1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, pop_size):
            # Update the Position of search agent
            if self.generator.uniform() < self.p_t:  # Eq.(2.7)
                pos_new = self.problem.generate_solution()
            else:
                p = np.tanh(
                    np.abs(self.population[idx].fitness - self.g_best.fitness)
                )  # Eq.(2.2)
                vb = self.generator.uniform(-a, a, self.problem.n_dims)  # Eq.(2.3)
                vc = self.generator.uniform(-b, b, self.problem.n_dims)
                # two positions randomly selected from population, apply for the whole problem size instead of 1 variable
                id_a, id_b = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx}), 2, replace=False
                )
                pos_1 = self.g_best.solution + vb * (
                        self.weights[idx] * self.population[id_a].solution
                        - self.population[id_b].solution
                )
                pos_2 = vc * self.population[idx].solution
                condition = self.generator.random(self.problem.n_dims) < p
                pos_new = np.where(condition, pos_1, pos_2)
            # Check bound and re-calculate fitness after each individual move
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

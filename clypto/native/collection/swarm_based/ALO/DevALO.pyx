#!/usr/bin/env python
# Created by "Thieu" at 12:01, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.swarm_based.ALO.OriginalALO cimport OriginalALO


cdef class DevALO(OriginalALO):
    """
    The developed version: Ant Lion Optimizer (ALO)

    Notes:
        + Improved performance by removing the for loop when creating n random walks

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import ALO    >>> import numpy as np
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
    >>> model = ALO.DevALO(epoch=1000, pop_size=50)
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
        super().__init__(epoch, pop_size, **kwargs)

    def random_walk_antlion__(self, solution, current_epoch):
        pop_size = self.population.size()
        I = 1  # I is the ratio in Equations (2.10) and (2.11)
        if current_epoch > self.epoch / 10:
            I = 1 + 100 * (current_epoch / self.epoch)
        if current_epoch > self.epoch / 2:
            I = 1 + 1000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * (3 / 4):
            I = 1 + 10000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.9:
            I = 1 + 100000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.95:
            I = 1 + 1000000 * (current_epoch / self.epoch)
        # Decrease boundaries to converge towards antlion
        lb = self.problem.bounds.low / I  # Equation (2.10) in the paper
        ub = self.problem.bounds.up / I  # Equation (2.10) in the paper
        # Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]. Eq 2.8, 2.9
        lb = lb + solution if self.generator.random() < 0.5 else -lb + solution
        ub = ub + solution if self.generator.random() < 0.5 else -ub + solution
        # This function creates n random walks and normalize according to lb and ub vectors,
        ## Using matrix and vector for better performance
        X = np.array(
            [
                np.cumsum(2 * (self.generator.random(pop_size) > 0.5) - 1)
                for _ in range(0, self.problem.n_dims)
            ]
        )
        a = np.min(X, axis=1)
        b = np.max(X, axis=1)
        temp1 = np.reshape((ub - lb) / (b - a), (self.problem.n_dims, 1))
        temp0 = X - np.reshape(a, (self.problem.n_dims, 1))
        X_norm = temp0 * temp1 + np.reshape(lb, (self.problem.n_dims, 1))
        return X_norm

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        list_fitness = np.array([item.fitness for item in self.population])
        # This for loop simulate random walks
        pop_new = []
        for idx in range(0, pop_size):
            # Select ant lions based on their fitness (the better anlion the higher chance of catching ant)
            rolette_index = cy.roulette_wheel(self.generator, self.problem.sense, list_fitness)
            # RA is the random walk around the selected antlion by rolette wheel
            RA = self.random_walk_antlion__(self.population[rolette_index].solution, epoch)
            # RE is the random walk around the elite (the best antlion so far)
            RE = self.random_walk_antlion__(self.g_best.solution, epoch)
            temp = (RA[:, idx] + RE[:, idx]) / 2  # Equation(2.13) in the paper
            # Bound checking (bring back the antlions of ants inside search space if they go beyonds the boundaries
            x = cy.correct_solution(self.problem, temp)
            agent = self.population.create_agent(x)
            pop_new.append(agent)
        pop_new = self.population.evaluate(pop_new, self.mode)
        # Update antlion positions and fitnesses based on the ants (if an ant becomes fitter than an antlion
        # we assume it was caught by the antlion and the antlion update goes to its position to build the trap)
        self.population = self.population.spawn(cy.sort_agents(self.population + pop_new, self.problem.sense)[:pop_size])
        # Keep the elite in the population
        self.population[-1] = cy.duplicate_agent(self.g_best)

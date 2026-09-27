#!/usr/bin/env python
# Created by "Thieu" at 22:46, 26/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class LARO(cy.Optimizer):
    """
    The improved version of:  Lévy flight, and the selective opposition version of the artificial rabbit algorithm (LARO)

    Links:
        1. https://doi.org/10.3390/sym14112282

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import ARO    >>> import numpy as np
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
    >>> model = ARO.LARO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, Y., Huang, L., Zhong, J., & Hu, G. (2022). LARO: Opposition-based learning boosted
    artificial rabbits-inspired optimization algorithm with Lévy flight. Symmetry, 14(11), 2282.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        theta = 2 * (1 - (epoch + 1) / self.epoch)
        pop_new = []
        for idx in range(0, pop_size):
            L = (np.exp(1) - np.exp((epoch / self.epoch) ** 2)) * (
                np.sin(2 * np.pi * self.generator.random())
            )
            temp = np.zeros(self.problem.n_dims)
            rd_index = self.generator.choice(
                np.arange(0, self.problem.n_dims),
                int(np.ceil(self.generator.random() * self.problem.n_dims)),
                replace=False,
            )
            temp[rd_index] = 1
            R = L * temp  # Eq 2
            A = 2 * np.log(1.0 / self.generator.random()) * theta  # Eq. 15
            if A > 1:  # # detour foraging strategy
                rand_idx = self.generator.integers(0, pop_size)
                pos_new = (
                        self.population[rand_idx].solution
                        + R * (self.population[idx].solution - self.population[rand_idx].solution)
                        + np.round(0.5 * (0.05 + self.generator.random()))
                        * self.generator.normal(0, 1)
                )  # Eq. 1
            else:  # Random hiding stage
                gr = np.zeros(self.problem.n_dims)
                rd_index = self.generator.choice(
                    np.arange(0, self.problem.n_dims),
                    int(np.ceil(self.generator.random() * self.problem.n_dims)),
                    replace=False,
                )
                gr[rd_index] = 1  # Eq. 12
                H = self.generator.normal(0, 1) * (epoch / self.epoch)  # Eq. 8
                b = self.population[idx].solution + H * gr * self.population[idx].solution  # Eq. 13
                levy = cy.levy_flight(self.generator, beta=1.5, multiplier=0.1)
                pos_new = self.population[idx].solution + R * (
                        levy * b - self.population[idx].solution
                )  # Eq. 11
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        # Selective Opposition (SO) Strategy
        TS = 2 - (2 * epoch / self.epoch)
        for idx in range(0, pop_size):
            if self.population[idx].fitness != self.g_best.fitness:
                dd = np.abs(self.g_best.solution - self.population[idx].solution)
                idx_far = np.sign(dd - TS) < 0
                n_df = np.sum(idx_far)
                n_dc = np.sum(np.sign(dd - TS) > 0)
                src = 1 - 6 * np.sum(dd ** 2) / np.dot(dd, (dd ** 2 - 1))
                if len(dd[idx_far]) == 0:
                    df_lb, df_ub = np.min(dd), np.max(dd)
                else:
                    df_lb, df_ub = np.min(dd[idx_far]), np.max(dd[idx_far])
                if src <= 0 and n_df > n_dc:
                    pos_new = df_lb + df_ub - self.population[idx].solution
                    pos_new = self.population.correct_solution(pos_new)
                    target = self.population.evaluate_solution(pos_new)
                    if cy.is_better(target, self.population[idx], self.problem.sense):
                        self.population[idx].update_solution(target, pos_new)

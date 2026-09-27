#!/usr/bin/env python
# Created by "Thieu" at 10:14, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevTLO(cy.Optimizer):
    """
    The developed version: Teaching Learning-based Optimization (TLO)

    Links:
       1. https://doi.org/10.5267/j.ijiec.2012.03.007

    Notes:
        + Use numpy np.array to make operations faster
        + The global best solution is used

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import TLO    >>> import numpy as np
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
    >>> model = TLO.DevTLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R. and Patel, V., 2012. An elitist teaching-learning-based optimization algorithm for solving
    complex constrained optimization problems. international journal of industrial engineering computations, 3(4), pp.535-560.
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_new = []
        for idx in range(0, pop_size):
            ## Teaching Phrase
            TF = self.generator.integers(1, 3)  # 1 or 2 (never 3)
            list_pos = np.array([agent.solution for agent in self.population])
            DIFF_MEAN = self.generator.random(self.problem.n_dims) * (
                    self.g_best.solution - TF * np.mean(list_pos, axis=0)
            )
            pos_new = self.population[idx].solution + DIFF_MEAN
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        pop_child = []
        for idx in range(0, pop_size):
            ## Learning Phrase
            pos_new = self.population[idx].solution.copy().astype(float)
            id_partner = self.generator.choice(
                np.setxor1d(np.array(range(pop_size)), np.array([idx]))
            )
            if cy.is_better(self.population[idx], self.population[id_partner], self.problem.sense):
                pos_new += self.generator.random(self.problem.n_dims) * (
                        self.population[idx].solution - self.population[id_partner].solution
                )
            else:
                pos_new += self.generator.random(self.problem.n_dims) * (
                        self.population[id_partner].solution - self.population[idx].solution
                )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.population.evaluate(pop_child, self.mode)
            self.population = self.population.greedy(pop_child)

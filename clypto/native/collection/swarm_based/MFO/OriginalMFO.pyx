#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalMFO(cy.Optimizer):
    """
    The developed version: Moth-Flame Optimization (MFO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import MFO    >>> import numpy as np
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
    >>> model = MFO.OriginalMFO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2015. Moth-flame optimization algorithm: A novel nature-inspired
    heuristic paradigm. Knowledge-based systems, 89, pp.228-249.
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

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Number of flames Eq.(3.14) in the paper (linearly decreased)
        num_flame = round(pop_size - epoch * ((pop_size - 1) / self.epoch))
        # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1.0 + epoch * (-1.0 / self.epoch)
        pop_flames = self.population.sort()
        g_best = cy.duplicate_agent(pop_flames[0])
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            #   D in Eq.(3.13)
            distance_to_flame = np.abs(
                pop_flames[idx].solution - agent.solution
            )
            t = (a - 1) * self.generator.uniform() + 1
            b = 1
            # Update the position of the moth with respect to its corresponding flame, Eq.(3.12).
            temp_1 = (
                    distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi)
                    + pop_flames[idx].solution
            )
            # Update the position of the moth with respect to one flame Eq.(3.12).
            temp_2 = (
                    distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi)
                    + g_best.solution
            )
            list_idx = idx * np.ones(self.problem.n_dims)
            x = np.where(list_idx < num_flame, temp_1, temp_2)
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], child, self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

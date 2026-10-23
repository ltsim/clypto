#!/usr/bin/env python
# Created by "Thieu" at 17:38, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalCircleSA(cy.Optimizer):
    """
    The original version of: Circle Search Algorithm (CircleSA)

    Links:
        1. https://doi.org/10.3390/math10101626
        2. https://www.mdpi.com/2227-7390/10/10/1626

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.math_based import CircleSA    >>> import numpy as np
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
    >>> model = CircleSA.OriginalCircleSA(epoch=1000, pop_size=50, c_factor=0.8)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Qais, M. H., Hasanien, H. M., Turky, R. A., Alghuwainem, S., Tostado-Véliz, M., & Jurado, F. (2022).
    Circle Search Algorithm: A Geometry-Based Metaheuristic Optimization Algorithm. Mathematics, 10(10), 1626.
    """

    cdef public double c_factor

    def __init__(self, epoch=10000, pop_size=100, c_factor=0.8, **kwargs):
        super().__init__(parameters=["epoch", "pop_size", "c_factor"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.c_factor = cy.validator(float, c_factor, (0, 1.0), "c_factor")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        a = np.pi - np.pi * (epoch / self.epoch) ** 2  # Eq. 8
        p = 1 - 0.9 * (epoch / self.epoch) ** 0.5
        threshold = self.c_factor * self.epoch
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            w = a * self.generator.random() - a
            if epoch > threshold:
                x_new = self.g_best.solution + (
                        self.g_best.solution - agent.solution
                ) * np.tan(w * self.generator.random())
            else:
                x_new = self.g_best.solution - (
                        self.g_best.solution - agent.solution
                ) * np.tan(w * p)
            x = cy.correct_solution(self.problem, x_new)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                n_population[-1].evaluate(self.problem)
        self.population = self.population.spawn(self.population.evaluate(n_population, self.mode))

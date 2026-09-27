#!/usr/bin/env python
# Created by "Thieu" at 17:21, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevSOA(cy.Optimizer):
    """
    The developed version: Seagull Optimization Algorithm (SOA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0950705118305768

    Notes:
        1. The original one will not work because their operators always make the solution out of bound.
        2. I added the normal random number in Eq. 14 to make its work
        3. Besides, I will check keep the better one and remove the worst

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + fc (float): [1.0, 10.0] -> better [1, 5], freequency of employing variable A (A linear decreased from fc to 0), default = 2

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.bio_based import SOA    >>> import numpy as np
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
    >>> model = SOA.DevSOA(epoch=1000, pop_size=50, fc = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    cdef public double fc

    def __init__(self, epoch=10000, pop_size=100, fc=2, **kwargs):
        super().__init__(parameters=["epoch", "pop_size", "fc"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.fc = cy.validator(float, fc, [1.0, 10.0], "fc")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        A = self.fc - epoch * self.fc / self.epoch  # Eq. 6
        uu = vv = 1
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            B = 2 * A ** 2 * self.generator.random()  # Eq. 8
            M = B * (self.g_best.solution - agent.solution)  # Eq. 7
            C = A * agent.solution  # Eq. 5
            D = np.abs(C + M)  # Eq. 9
            k = self.generator.uniform(0, 2 * np.pi)
            r = uu * np.exp(k * vv)
            xx = r * np.cos(k)
            yy = r * np.sin(k)
            zz = r * k
            x = (
                    xx * yy * zz * D + self.generator.normal(0, 1) * self.g_best.solution
            )  # Eq. 14
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)

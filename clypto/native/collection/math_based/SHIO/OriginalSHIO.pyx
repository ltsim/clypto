#!/usr/bin/env python
# Created by "Thieu" at 18:09, 13/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalSHIO(cy.Optimizer):
    """
    The original version of: Success History Intelligent Optimizer (SHIO)

    Links:
        1. https://link.springer.com/article/10.1007/s11227-021-04093-9
        2. https://www.mathworks.com/matlabcentral/fileexchange/122157-success-history-intelligent-optimizer-shio

    Notes:
        1. The algorithm is designed with simplicity and ease of implementation in mind, utilizing basic operators.
        2. This algorithm has several limitations and weak when dealing with several problems
        3. The algorithm's convergence is slow. The Matlab code has many errors and unnecessary things.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.math_based import SHIO    >>> import numpy as np
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
    >>> model = SHIO.OriginalSHIO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Fakhouri, H. N., Hamad, F., & Alawamrah, A. (2022). Success history intelligent optimizer. The Journal of Supercomputing, 1-42.
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
        ranked = self.population.sort()
        (b1, b2, b3) = [cy.duplicate_agent(agent) for agent in ranked[:3]]
        a = 1.5
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            a = a - 0.04
            x1 = b1.solution + (
                    a * 2 * self.generator.random(self.problem.n_dims) - a
            ) * np.abs(
                self.generator.random(self.problem.n_dims) * b1.solution
                - agent.solution
            )
            x2 = b2.solution + (
                    a * 2 * self.generator.random(self.problem.n_dims) - a
            ) * np.abs(
                self.generator.random(self.problem.n_dims) * b2.solution
                - agent.solution
            )
            x3 = b3.solution + (
                    a * 2 * self.generator.random(self.problem.n_dims) - a
            ) * np.abs(
                self.generator.random(self.problem.n_dims) * b3.solution
                - agent.solution
            )
            x = (x1 + x2 + x3) / 3
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
        n_population = self.population.evaluate(n_population, self.mode)
        self.population = self.population.spawn(n_population)

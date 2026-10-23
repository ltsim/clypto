#!/usr/bin/env python
# Created by "Thieu" at 17:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class DevSCA(cy.Optimizer):
    """
    The developed version: Sine Cosine Algorithm (SCA)

    Notes:
        + The flow and few equations are changed
        + Third loops are removed faster computational time

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.math_based import SCA    >>> import numpy as np
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
    >>> model = SCA.DevSCA(epoch=1000, pop_size=50)
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
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            # Eq 3.4, r1 decreases linearly from a to 0
            a = 2.0
            r1 = a * (1.0 - epoch / self.epoch)
            # Update r2, r3, and r4 for Eq. (3.3), remove third loop here
            r2 = 2 * np.pi * self.generator.uniform(0, 1, self.problem.n_dims)
            r3 = 2 * self.generator.uniform(0, 1, self.problem.n_dims)
            # Eq. 3.3, 3.1 and 3.2
            pos_new1 = agent.solution + r1 * np.sin(r2) * np.abs(
                r3 * self.g_best.solution - agent.solution
            )
            pos_new2 = agent.solution + r1 * np.cos(r2) * np.abs(
                r3 * self.g_best.solution - agent.solution
            )
            x = np.where(
                self.generator.random(self.problem.n_dims) < 0.5, pos_new1, pos_new2
            )
            # Check the bound
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)

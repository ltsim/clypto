#!/usr/bin/env python
# Created by "Thieu" at 21:45, 13/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalCDO(cy.Optimizer):
    """
    The original version of: Chernobyl Disaster Optimizer (CDO)

    Links:
        1. https://link.springer.com/article/10.1007/s00521-023-08261-1
        2. https://www.mathworks.com/matlabcentral/fileexchange/124351-chernobyl-disaster-optimizer-cdo

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.physics_based import CDO    >>> import numpy as np
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
    >>> model = CDO.OriginalCDO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Shehadeh, H. A. (2023). Chernobyl disaster optimizer (CDO): a novel meta-heuristic method
    for global optimization. Neural Computing and Applications, 1-17.
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
        ranked = self.population.sort()
        (b1, b2, b3) = [agent.copy() for agent in ranked[:3]]
        a = 3.0 - 3.0 * epoch / self.epoch
        a1 = np.log10((16000 - 1) * self.generator.random() + 16000)
        a2 = np.log10((270000 - 1) * self.generator.random() + 270000)
        a3 = np.log10((300000 - 1) * self.generator.random() + 300000)
        pop_new = []
        for idx in range(0, pop_size):
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            pa = np.pi * r1 * r1 / (0.25 * a1) - a * self.generator.random(
                self.problem.n_dims
            )
            c1 = r2 * r2 * np.pi
            alpha = np.abs(c1 * b1.solution - self.population[idx].solution)
            pos_a = 0.25 * (b1.solution - pa * alpha)

            r3 = self.generator.random(self.problem.n_dims)
            r4 = self.generator.random(self.problem.n_dims)
            pb = np.pi * r3 * r3 / (0.5 * a2) - a * self.generator.random(
                self.problem.n_dims
            )
            c2 = r4 * r4 * np.pi
            beta = np.abs(c2 * b2.solution - self.population[idx].solution)
            pos_b = 0.5 * (b2.solution - pb * beta)

            r5 = self.generator.random(self.problem.n_dims)
            r6 = self.generator.random(self.problem.n_dims)
            pc = np.pi * r5 * r5 / a3 - a * self.generator.random(self.problem.n_dims)
            c3 = r6 * r6 * np.pi
            gama = np.abs(c3 * b3.solution - self.population[idx].solution)
            pos_c = b3.solution - pc * gama

            pos_new = (pos_a + pos_b + pos_c) / 3
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].evaluate(self.problem)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
        self.population = pop_new

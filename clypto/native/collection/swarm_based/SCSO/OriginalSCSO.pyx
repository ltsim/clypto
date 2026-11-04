#!/usr/bin/env python
# Created by "Thieu" at 17:36, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalSCSO(cy.Optimizer):
    """
    The original version of: Sand Cat Swarm Optimization (SCSO)

    Links:
        1. https://link.springer.com/article/10.1007/s00366-022-01604-x
        2. https://www.mathworks.com/matlabcentral/fileexchange/110185-sand-cat-swarm-optimization

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import SCSO    >>> import numpy as np
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
    >>> model = SCSO.OriginalSCSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Seyyedabbasi, A., & Kiani, F. (2022). Sand Cat swarm optimization: a nature-inspired algorithm to
    solve global optimization problems. Engineering with Computers, 1-25.
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

    def initialize_variables(self):
        self.ss = 2  # maximum Sensitivity range
        self.pp = np.arange(1, 361)

    def get_index_roulette_wheel_selection__(self, p):
        p = p / np.sum(p)
        c = np.cumsum(p)
        return np.argwhere(self.generator.random() < c)[0][0]

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        guides_r = self.ss - (self.ss * epoch / self.epoch)
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            r = self.generator.random() * guides_r
            R = (
                        2 * guides_r
                ) * self.generator.random() - guides_r  # controls to transition phases
            x = agent.solution.copy()
            for jdx in range(0, self.problem.n_dims):
                teta = self.get_index_roulette_wheel_selection__(self.pp)
                if -1 <= R <= 1:
                    rand_pos = np.abs(
                        self.generator.random() * self.g_best.solution[jdx]
                        - agent.solution[jdx]
                    )
                    x[jdx] = self.g_best.solution[jdx] - r * rand_pos * np.cos(
                        teta
                    )
                else:
                    cp = int(self.generator.random() * pop_size)
                    x[jdx] = r * (
                            self.population[cp].solution[jdx]
                            - self.generator.random() * agent.solution[jdx]
                    )
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                n_population[-1].evaluate(self.problem)
        self.population = self.population.spawn(self.population.evaluate(n_population, self.mode))

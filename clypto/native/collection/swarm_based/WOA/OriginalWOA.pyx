#!/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalWOA(cy.Optimizer):
    """
    The original version of: Whale Optimization Algorithm (WOA)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2016.01.008
        2. https://mathworks.com/matlabcentral/fileexchange/55667-the-whale-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import WOA    >>> import numpy as np
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
    >>> model = WOA.OriginalWOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm. Advances in engineering software, 95, pp.51-67.
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
        a = 2 - 2 * epoch / self.epoch  # linearly decreased from 2 to 0
        a2 = -1 + epoch * ((-1) / self.epoch)
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            r1, r2 = self.generator.random(size=2)
            A = a * (2 * r1 - a)
            C = 2 * r2
            b = 1
            l = (a2 - 1) * self.generator.random() + 1
            p = self.generator.random()

            x = agent.solution.copy()
            for jdx in range(0, self.problem.n_dims):
                if p < 0.5:
                    if np.abs(A) >= 1:
                        id_r = self.generator.choice(
                            list(set(range(0, pop_size)) - {idx})
                        )
                        D_X_rand = abs(
                            C * self.population[id_r].solution[jdx]
                            - agent.solution[jdx]
                        )
                        x[jdx] = self.population[id_r].solution[jdx] - A * D_X_rand
                    else:
                        D_Leader = abs(
                            C * self.g_best.solution[jdx] - agent.solution[jdx]
                        )
                        x[jdx] = self.g_best.solution[jdx] - A * D_Leader
                else:
                    D1 = abs(self.g_best.solution[jdx] - agent.solution[jdx])
                    x[jdx] = (
                            D1 * np.exp(b * l) * np.cos(l * 2 * np.pi)
                            + self.g_best.solution[jdx]
                    )
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                # the classic code evaluates x, not agent.solution (MEALPY behaviour, kept)
                agent.update_solution(self.population.evaluate_solution(x), agent.solution)
        if self.mode != "sequential":
            self.population = self.population.spawn(self.population.evaluate(n_population, self.mode))

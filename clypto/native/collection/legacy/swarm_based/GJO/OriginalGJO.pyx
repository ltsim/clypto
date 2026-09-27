#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalGJO(cy.Optimizer):
    """
    The original version of: Golden jackal optimization (GJO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S095741742200358X
        2. https://www.mathworks.com/matlabcentral/fileexchange/108889-golden-jackal-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import GJO    >>> import numpy as np
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
    >>> model = GJO.OriginalGJO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Chopra, N., & Ansari, M. M. (2022). Golden jackal optimization: A novel nature-inspired
    optimizer for engineering applications. Expert Systems with Applications, 198, 116924.
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
        E1 = 1.5 * (1.0 - (epoch / self.epoch))
        RL = cy.levy_flight(self.generator, beta=1.5, multiplier=0.05, size=(pop_size, self.problem.n_dims), case=-1)
        ranked = self.population.sort()
        (male, female) = [agent.copy() for agent in ranked[:2]]
        pop_new = []
        for idx in range(0, pop_size):
            male_pos = male.solution.copy()
            female_pos = female.solution.copy()
            for jdx in range(0, self.problem.n_dims):
                r1 = self.generator.random()
                E0 = 2 * r1 - 1
                E = E1 * E0
                if np.abs(E) < 1:  # EXPLOITATION
                    t1 = np.abs(
                        (
                                RL[idx, jdx] * male.solution[jdx]
                                - self.population[idx].solution[jdx]
                        )
                    )
                    male_pos[jdx] = male.solution[jdx] - E * t1
                    t2 = np.abs(
                        (
                                RL[idx, jdx] * female.solution[jdx]
                                - self.population[idx].solution[jdx]
                        )
                    )
                    female_pos[jdx] = female.solution[jdx] - E * t2
                else:  # EXPLORATION
                    t1 = np.abs(
                        (
                                male.solution[jdx]
                                - RL[idx, jdx] * self.population[idx].solution[jdx]
                        )
                    )
                    male_pos[jdx] = male.solution[jdx] - E * t1
                    t2 = np.abs(
                        (
                                female.solution[jdx]
                                - RL[idx, jdx] * self.population[idx].solution[jdx]
                        )
                    )
                    female_pos[jdx] = female.solution[jdx] - E * t2
            pos_new = (male_pos + female_pos) / 2
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = agent
        if self.mode in self.AVAILABLE_MODES:
            self.population = self.population.evaluate(pop_new, self.mode)

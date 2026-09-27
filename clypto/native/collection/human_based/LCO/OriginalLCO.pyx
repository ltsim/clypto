#!/usr/bin/env python
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalLCO(cy.Optimizer):
    """
    The original version of: Life Choice-based Optimization (LCO)

    Links:
        1. https://doi.org/10.1007/s00500-019-04443-z

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r1 (float): [1.5, 4], coefficient factor, default = 2.35

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import LCO    >>> import numpy as np
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
    >>> model = LCO.OriginalLCO(epoch=1000, pop_size=50, r1 = 2.35)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Khatri, A., Gaba, A., Rana, K.P.S. and Kumar, V., 2020. A novel life choice-based optimizer. Soft Computing, 24(12), pp.9121-9141.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            r1: float = 2.35,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r1 (float): coefficient factor
        """
        super().__init__(parameters=["epoch", "pop_size", "r1"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.r1 = cy.validator(float, r1, [1.0, 3.0], "r1")
        self.n_agents = int(np.ceil(np.sqrt(self.population.size())))

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            prob = self.generator.random()
            if prob > 0.875:  # Update using Eq. 1, update from n best position
                temp = np.array(
                    [
                        self.generator.random() * self.population[j].solution
                        for j in range(0, self.n_agents)
                    ]
                )
                temp = np.mean(temp, axis=0)
            elif prob < 0.7:  # Update using Eq. 2-6
                f1 = 1 - epoch / self.epoch
                f2 = 1 - f1
                prev_pos = (
                    self.g_best.solution if idx == 0 else self.population[idx - 1].solution
                )
                best_diff = (
                        f1 * self.r1 * (self.g_best.solution - agent.solution)
                )
                better_diff = f2 * self.r1 * (prev_pos - agent.solution)
                temp = (
                        agent.solution
                        + self.generator.random() * better_diff
                        + self.generator.random() * best_diff
                )
            else:
                temp = (
                        self.problem.bounds.up
                        - (agent.solution - self.problem.bounds.low)
                        * self.generator.random()
                )
            x = cy.correct_solution(self.problem, temp)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

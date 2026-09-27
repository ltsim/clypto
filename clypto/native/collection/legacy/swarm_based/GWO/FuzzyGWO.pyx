#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native.fuzzy import FuzzySystem as FS
cimport clypto.core as cy



cdef class FuzzyGWO(cy.Optimizer):
    """
    The original version of: Fuzzy Hierarchical Operator - Grey Wolf Optimizer (FHO-GWO or FuzzyGWO or F-GWO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2017.03.048

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.FuzzyGWO(epoch=1000, pop_size=50, fuzzy_name="increase")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Rodríguez, Luis, Oscar Castillo, José Soria, Patricia Melin, Fevrier Valdez, Claudia I. Gonzalez, Gabriela E. Martinez, and Jesus Soto. "A fuzzy hierarchical operator in the grey wolf optimizer algorithm." Applied Soft Computing 57 (2017): 315-328.
    """

    cdef public str fuzzy_name

    FUZZY_OPERATORS = ["increase", "decrease"]

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        fuzzy_name: str = "increase",
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            fuzzy_name (str): type of fuzzy operator to use, default = "increase"
        """
        super().__init__(parameters=["epoch", "pop_size", "fuzzy_name"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.fuzzy_name = cy.validator(str, fuzzy_name, FuzzyGWO.FUZZY_OPERATORS, "fuzzy_name")

    def initialize_variables(self) -> None:
        self.fuzzy_system = FS(self.fuzzy_name)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # linearly decreased from 2 to 0
        a = 2 - 2.0 * epoch / self.epoch
        ranked = self.population.sort()
        list_best = [agent.copy() for agent in ranked[:3]]
        pop_new = []
        for idx in range(0, pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(
                C1 * list_best[0].solution - self.population[idx].solution
            )
            X2 = list_best[1].solution - A2 * np.abs(
                C2 * list_best[1].solution - self.population[idx].solution
            )
            X3 = list_best[2].solution - A3 * np.abs(
                C3 * list_best[2].solution - self.population[idx].solution
            )

            # Get fuzzy weights
            FW_alpha, FW_beta, FW_delta = self.fuzzy_system.get_fuzzy_weights(
                epoch, self.epoch
            )
            total_weight = FW_alpha + FW_beta + FW_delta
            pos_new = (X1 * FW_alpha + X2 * FW_beta + X3 * FW_delta) / total_weight
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

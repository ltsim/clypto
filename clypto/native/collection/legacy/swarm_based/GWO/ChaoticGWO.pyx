#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native.chaotic import ChaoticMap as CM
cimport clypto.core as cy



cdef class ChaoticGWO(cy.Optimizer):
    """
    The original version of: Chaotic-based Grey Wolf Optimizer (Chaotic-GWO or C-GWO)

    Links:
        1. https://doi.org/10.1016/j.jcde.2017.02.005

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
    >>> model = GWO.ChaoticGWO(epoch=1000, pop_size=50, chaotic_name="chebyshev", initial_chaotic_value=0.7)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Kohli, M., & Arora, S. (2018). Chaotic grey wolf optimization algorithm for constrained optimization problems. Journal of computational design and engineering, 5(4), 458-472.
    """

    cdef public str chaotic_name
    cdef public double initial_chaotic_value

    CHAOTIC_MAPS = {
        "bernoulli": CM.bernoulli_map,
        "logistic": CM.logistic_map,
        "chebyshev": CM.chebyshev_map,
        "circle": CM.circle_map,
        "cubic": CM.cubic_map,
        "icmic": CM.icmic_map,
        "piecewise": CM.piecewise_map,
        "singer": CM.singer_map,
        "sinusoidal": CM.sinusoidal_map,
        "tent": CM.tent_map,
    }

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        chaotic_name: str = "chebyshev",
        initial_chaotic_value: float = 0.7,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            chaotic_name (str): name of chaotic map to use, default = "chebyshev"
            initial_chaotic_value (float): initial value for chaotic map, default = 0.7
        """
        super().__init__(parameters=["epoch", "pop_size", "chaotic_name", "initial_chaotic_value"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.chaotic_name = cy.validator(str, chaotic_name, ChaoticGWO.CHAOTIC_MAPS.keys(), "chaotic_name")
        self.initial_chaotic_value = cy.validator(float, initial_chaotic_value, [0.0, 1.0], "initial_chaotic_value")

    def initialize_variables(self) -> None:
        self.chao_value = self.initial_chaotic_value
        self.chao_func = ChaoticGWO.CHAOTIC_MAPS[self.chaotic_name]

    def _update_chao_value(self):
        """Update chaotic value using selected chaotic map"""
        chao_value = self.chao_func(self.chao_value)
        # Ensure chaotic value stays in [0, 1]
        self.chao_value = np.clip(chao_value, 0, 1)

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
            self._update_chao_value()
            A1 = a * (
                2 * self.generator.random(self.problem.n_dims) * self.chao_value - 1
            )
            A2 = a * (
                2 * self.generator.random(self.problem.n_dims) * self.chao_value - 1
            )
            A3 = a * (
                2 * self.generator.random(self.problem.n_dims) * self.chao_value - 1
            )
            C1 = 2 * self.generator.random(self.problem.n_dims) * self.chao_value
            C2 = 2 * self.generator.random(self.problem.n_dims) * self.chao_value
            C3 = 2 * self.generator.random(self.problem.n_dims) * self.chao_value
            X1 = list_best[0].solution - A1 * np.abs(
                C1 * list_best[0].solution - self.population[idx].solution
            )
            X2 = list_best[1].solution - A2 * np.abs(
                C2 * list_best[1].solution - self.population[idx].solution
            )
            X3 = list_best[2].solution - A3 * np.abs(
                C3 * list_best[2].solution - self.population[idx].solution
            )
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

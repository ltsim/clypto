#!/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevMVO(cy.Optimizer):
    """
    The developed version: Multi-Verse Optimizer (MVO)

    Notes:
        + New routtele wheel selection can handle negative values
        + Removed condition when self.generator.normalize fitness. So the chance to choose while whole higher --> better

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wep_min (float): [0.05, 0.3], Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
        + wep_max (float: [0.75, 1.0], Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.physics_based import MVO    >>> import numpy as np
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
    >>> model = MVO.DevMVO(epoch=1000, pop_size=50, wep_min = 0.2, wep_max = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            wep_min: float = 0.2,
            wep_max: float = 1.0,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wep_min (float): Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
            wep_max (float: Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0
        """
        super().__init__(parameters=["epoch", "pop_size", "wep_min", "wep_max"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.wep_min = cy.validator(float, wep_min, (0, 0.5), "wep_min")
        self.wep_max = cy.validator(float, wep_max, [0.5, 3.0], "wep_max")

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Eq. (3.3) in the paper
        wep = self.wep_max - epoch * ((self.wep_max - self.wep_min) / self.epoch)
        # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
        tdr = 1 - epoch ** (1.0 / 6) / self.epoch ** (1.0 / 6)
        pop_new = []
        for idx in range(0, pop_size):
            if self.generator.uniform() < wep:
                list_fitness = np.array([agent.fitness for agent in self.population])
                white_hole_id = cy.roulette_wheel(self.generator, self.problem.sense, list_fitness)
                black_hole_pos_1 = self.population[idx].solution + tdr * self.generator.normal(
                    0, 1
                ) * (self.population[white_hole_id].solution - self.population[idx].solution)
                black_hole_pos_2 = self.g_best.solution + tdr * self.generator.normal(
                    0, 1
                ) * (self.g_best.solution - self.population[idx].solution)
                black_hole_pos = np.where(
                    self.generator.random(self.problem.n_dims) < 0.5,
                    black_hole_pos_1,
                    black_hole_pos_2,
                )
            else:
                black_hole_pos = self.problem.generate_solution()
            pos_new = self.population.correct_solution(black_hole_pos)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

#!/usr/bin/env python
# Created by "Thieu" at 17:48, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevHS(cy.Optimizer):
    """
    The developed version: Harmony Search (HS)

    Links:
        1. https://doi.org/10.1177/003754970107600201

    Notes:
        - Used the global best in the harmony memories
        - Removed all third for loops

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c_r (float): [0.1, 0.5], Harmony Memory Consideration Rate), default = 0.15
        + pa_r (float): [0.3, 0.8], Pitch Adjustment Rate, default=0.5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.music_based import HS    >>> import numpy as np
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
    >>> model = HS.DevHS(epoch=1000, pop_size=50, c_r = 0.95, pa_r = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            c_r: float = 0.95,
            pa_r: float = 0.05,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c_r (float): Harmony Memory Consideration Rate, default = 0.15
            pa_r (float): Pitch Adjustment Rate, default=0.5
        """
        super().__init__(parameters=["epoch", "pop_size", "c_r", "pa_r"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.c_r = cy.validator(float, c_r, (0, 1.0), "c_r")
        self.pa_r = cy.validator(float, pa_r, (0, 1.0), "pa_r")

    def initialize_variables(self):
        self.fw = 0.0001 * (self.problem.bounds.up - self.problem.bounds.low)  # Fret Width (Bandwidth)
        self.fw_damp = 0.9995  # Fret Width Damp Ratio
        self.dyn_fw = self.fw

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_new = []
        for idx in range(0, pop_size):
            # Create New Harmony Position
            pos_new = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            delta = self.dyn_fw * self.generator.normal(
                self.problem.bounds.low, self.problem.bounds.up
            )
            # Use Harmony Memory
            pos_new = np.where(
                self.generator.random(self.problem.n_dims) < self.c_r,
                self.g_best.solution,
                pos_new,
            )
            # Pitch Adjustment
            x_new = pos_new + delta
            pos_new = np.where(
                self.generator.random(self.problem.n_dims) < self.pa_r, x_new, pos_new
            )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].evaluate(self.problem)
        pop_new = self.population.evaluate(pop_new, self.mode)
        # Update Damp Fret Width
        self.dyn_fw = self.dyn_fw * self.fw_damp
        # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
        self.population = cy.sort_agents(self.population + pop_new, self.problem.sense)[:pop_size]

#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer._native.legacy cimport _LegacyOptimizer


cdef class OGWO(_LegacyOptimizer):
    """
    The original version of: Opposition-based learning Grey Wolf Optimizer (OGWO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2021.107139

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GWO    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GWO.OGWO(epoch=1000, pop_size=50, miu_factor=2.0, jumping_rate=0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yu, X., Xu, W., & Li, C. (2021). Opposition-based learning grey wolf optimizer for global optimization. Knowledge-Based Systems, 226, 107139.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        miu_factor: float = 2.0,
        jumping_rate: float = 0.05,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_factor (float): nonlinear coefficient for equation (11), default = 2.0
            jumping_rate (float):  jumping rate for OBL, default = 0.05
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.miu_factor = self.validator.check_float(
            "miu_factor", miu_factor, [0.0, 10.0]
        )
        self.jumping_rate = self.validator.check_float(
            "jumping_rate", jumping_rate, [0.0, 1.0]
        )
        self.set_parameters(["epoch", "pop_size", "miu_factor", "jumping_rate"])
        self.sort_flag = False

    def initialization(self) -> None:
        """Initialize population with opposition-based learning"""
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

        # Generate opposition population using equation (12)
        pop_opposite = []
        for agent in self.pop:
            pos_opposite = self.problem.lb + self.problem.ub - agent.solution
            agent_opposite = self.generate_empty_agent(pos_opposite)
            agent_opposite.target = self.get_target(pos_opposite)
            pop_opposite.append(agent_opposite)
        # Combine original and opposite populations
        self.pop = self.get_sorted_and_trimmed_population(
            self.pop + pop_opposite, self.pop_size, minmax=self.problem.minmax
        )

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2.0 * (1 - (epoch / self.epoch) ** self.miu_factor)
        _, list_best, _ = self.get_special_agents(
            self.pop, n_best=3, minmax=self.problem.minmax
        )
        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(
                C1 * list_best[0].solution - self.pop[idx].solution
            )
            X2 = list_best[1].solution - A2 * np.abs(
                C2 * list_best[1].solution - self.pop[idx].solution
            )
            X3 = list_best[2].solution - A3 * np.abs(
                C3 * list_best[2].solution - self.pop[idx].solution
            )
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )

        # Apply opposition-based learning
        if self.generator.random() < self.jumping_rate:
            # Generate opposition population using equation (12)
            pop_opposite = []
            for agent in self.pop:
                pos_opposite = self.problem.lb + self.problem.ub - agent.solution
                agent_opposite = self.generate_empty_agent(pos_opposite)
                agent_opposite.target = self.get_target(pos_opposite)
                pop_opposite.append(agent_opposite)
            # Combine original and opposite populations
            self.pop = self.get_sorted_and_trimmed_population(
                self.pop + pop_opposite, self.pop_size, minmax=self.problem.minmax
            )

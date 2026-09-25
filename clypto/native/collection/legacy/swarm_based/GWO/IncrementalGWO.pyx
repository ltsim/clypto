#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer._native.legacy cimport _LegacyOptimizer


cdef class IncrementalGWO(_LegacyOptimizer):
    """
    The original version of: Incremental model-based Grey Wolf Optimizer (IncrementalGWO)

    Notes:
        + When calling the solve() function, you need to set the mode to "swarm" to use this algorithm as original version.
        + They update the position of whole population before calculating the fitness of each agent.

    Links:
        1. https://doi.org/10.1007/s00366-019-00837-7

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
    >>> model = GWO.IncrementalGWO(epoch=1000, pop_size=50, explore_factor=1.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Seyyedabbasi, A., & Kiani, F. (2021). I-GWO and Ex-GWO: improved algorithms of the Grey Wolf Optimizer to solve global optimization problems. Engineering with Computers, 37(1), 509-532.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        explore_factor: float = 1.5,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            explore_factor (float): factor to control exploration, default = 1.5
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.explore_factor = self.validator.check_float(
            "explore_factor", explore_factor, [0.0, 5.0]
        )
        self.set_parameters(["epoch", "pop_size", "explore_factor"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 * (1.0 - (epoch / self.epoch) ** self.explore_factor)
        pop_sorted, list_best, _ = self.get_special_agents(
            self.pop, n_best=3, minmax=self.problem.minmax
        )
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx == 0:
                # Alpha wolf updates based on hunting mechanism
                A = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C = 2 * self.generator.random(self.problem.n_dims)
                pos_new = list_best[0].solution - A * np.abs(
                    C * list_best[0].solution - self.pop[idx].solution
                )
            else:
                # Other wolves update based on all previous wolves (Equation 19)
                # Average position of all previous wolves (n-1 wolves)
                p_temp = np.array([agent.solution for agent in pop_sorted])
                mask = np.arange(p_temp.shape[0]) != idx
                pos_new = p_temp[mask].mean(axis=0)
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

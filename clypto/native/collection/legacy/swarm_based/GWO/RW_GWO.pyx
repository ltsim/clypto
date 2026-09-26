#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class RW_GWO(_LegacyOptimizer):
    """
    The original version of: Random Walk Grey Wolf Optimizer (RW-GWO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.RW_GWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Gupta, S. and Deep, K., 2019. A novel random walk grey wolf optimizer. Swarm and evolutionary computation, 44, pp.101-112.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0, Eq. 5
        b = 2.0 - 2.0 * epoch / self.epoch
        # linearly decreased from 2 to 0
        a = 2.0 - 2.0 * epoch / self.epoch
        _, leaders, _ = self.get_special_agents(
            self.pop, n_best=3, minmax=self.problem.minmax
        )

        ## Random walk here
        leaders_new = []
        for idx in range(0, len(leaders)):
            pos_new = leaders[idx].solution + a * self.generator.standard_cauchy(
                self.problem.n_dims
            )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            leaders_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                leaders[idx] = self.get_better_agent(
                    agent, leaders[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            leaders_new = self.update_target_for_population(leaders_new)
            leaders = self.greedy_selection_population(
                leaders, leaders_new, self.problem.minmax
            )

        ## Update other wolfs
        pop_new = []
        for idx in range(0, self.pop_size):
            # Eq. 3 and 4
            miu1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            miu2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            miu3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            c1 = 2 * self.generator.random(self.problem.n_dims)
            c2 = 2 * self.generator.random(self.problem.n_dims)
            c3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = leaders[0].solution - miu1 * np.abs(
                c1 * self.g_best.solution - self.pop[idx].solution
            )
            X2 = leaders[1].solution - miu2 * np.abs(
                c2 * self.g_best.solution - self.pop[idx].solution
            )
            X3 = leaders[2].solution - miu3 * np.abs(
                c3 * self.g_best.solution - self.pop[idx].solution
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
        self.pop = self.get_sorted_and_trimmed_population(
            self.pop + leaders, self.pop_size, self.problem.minmax
        )

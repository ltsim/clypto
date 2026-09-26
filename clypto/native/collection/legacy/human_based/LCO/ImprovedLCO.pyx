#!/usr/bin/env python
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class ImprovedLCO(LegacyOptimizer):
    """
    The improved version: Life Choice-based Optimization (ILCO)

    Notes:
        + The flow of the original LCO is kept.
        + Gaussian distribution and mutation mechanism are added
        + R1 parameter is removed

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import LCO    >>> import numpy as np
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
    >>> model = LCO.ImprovedLCO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self._set_parameters(["epoch", "pop_size"])
        self.pop_len = int(self.pop_size / 2)
        self.sort_flag = True

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # epoch: current chance, self.epoch: number of chances
        pop_new = []
        for idx in range(0, self.pop_size):
            rand = self.generator.random()
            if rand > 0.875:  # Update using Eq. 1, update from n best position
                n = int(np.ceil(np.sqrt(self.pop_size)))
                pos_new = np.array(
                    [
                        self.generator.random() * self.pop[j].solution
                        for j in range(0, n)
                    ]
                )
                pos_new = np.mean(pos_new, axis=0)
            elif rand < 0.7:  # Update using Eq. 2-6
                f = epoch / self.epoch
                if idx != 0:
                    better_diff = (
                            f
                            * self.generator.random()
                            * (self.pop[idx - 1].solution - self.pop[idx].solution)
                    )
                else:
                    better_diff = (
                            f
                            * self.generator.random()
                            * (self.g_best.solution - self.pop[idx].solution)
                    )
                best_diff = (
                        (1 - f)
                        * self.generator.random()
                        * (self.pop[0].solution - self.pop[idx].solution)
                )
                pos_new = self.pop[idx].solution + better_diff + best_diff
            else:
                pos_new = (
                        self.problem.bounds.up
                        - (self.pop[idx].solution - self.problem.bounds.low)
                        * self.generator.random()
                )
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.pop[idx] = self._get_better_agent(
                    agent, self.pop[idx], self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            self.pop = self._greedy_selection_population(
                self.pop, pop_new, self.problem.sense
            )

        ## Sort the updated population based on fitness
        pop = self._get_sorted_population(self.pop, self.problem.sense)
        local_best = pop[0].copy()
        pop_s1 = [agent.copy() for agent in pop[: self.pop_len]]
        pop_s2 = [agent.copy() for agent in pop[self.pop_len:]]
        ## Mutation scheme
        pop_child1 = []
        for idx in range(0, self.pop_len):
            pos_new = (
                    pop_s1[idx].solution
                    + self.generator.normal(0, 1, self.problem.n_dims)
                    * pop_s1[idx].solution
            )
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_child1.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                pop_s1[idx] = self._get_better_agent(
                    agent, pop_s1[idx], self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_child1 = self._update_target_for_population(pop_child1)
            pop_s1 = self._greedy_selection_population(
                pop_s1, pop_child1, self.problem.sense
            )

        ## Search Mechanism
        pos_s1_list = [agent.solution for agent in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        pop_child2 = []
        for idx in range(0, self.pop_len):
            pos_new = local_best.solution + self.generator.uniform(
                0, 1
            ) * pos_s1_mean * (epoch / self.epoch)
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_child2.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                pop_s2[idx] = self._get_better_agent(
                    pop_s2[idx], agent, self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_child2 = self._update_target_for_population(pop_s2)
            pop_s2 = self._greedy_selection_population(
                pop_s2, pop_child2, self.problem.sense
            )
        ## Construct a new population
        self.pop = pop_s1 + pop_s2

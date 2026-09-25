#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 17:19, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.human_based.SPBO.OriginalSPBO cimport OriginalSPBO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class DevSPBO(OriginalSPBO):
    """
    The developed version of: Student Psychology Based Optimization (SPBO)

    Notes:
        1. Replace uniform random number by normal random number
        2. Sort the population and select 1/3 pop size for each category

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import SPBO    >>> import numpy as np
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
    >>> model = SPBO.DevSPBO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        super().__init__(epoch, pop_size, name=name, mode=mode)
        self.sort_flag = True

    def evolve_agents(self, epoch):
        good = int(self.pop_size / 3)
        average = 2 * int(self.pop_size / 3)
        x_mean = np.mean([agent.solution for agent in self.objs], axis=0)
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx == 0:
                j = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                new_pos = self.g_best.solution + self.generator.normal(
                    0, 1, self.problem.n_dims
                ) * (self.g_best.solution - self.objs[j].solution)
            elif idx < good:  ## Good Student
                if self.generator.random() > self.generator.random():
                    new_pos = self.g_best.solution + self.generator.normal(
                        0, 1, self.problem.n_dims
                    ) * (self.g_best.solution - self.objs[idx].solution)
                else:
                    ra = self.generator.random(self.problem.n_dims)
                    new_pos = (
                            self.objs[idx].solution
                            + ra * (self.g_best.solution - self.objs[idx].solution)
                            + (1 - ra) * (self.objs[idx].solution - x_mean)
                    )
            elif idx < average:  ## Average Student
                new_pos = self.objs[idx].solution + self.generator.normal(
                    0, 1, self.problem.n_dims
                ) * (x_mean - self.objs[idx].solution)
            else:
                new_pos = self.problem.generate_solution()
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_empty_agent(new_pos)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(new_pos)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )

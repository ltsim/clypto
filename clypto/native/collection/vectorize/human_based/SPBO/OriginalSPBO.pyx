#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 17:19, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class OriginalSPBO(AgentListOptimizer):
    """
    The original version of: Student Psychology Based Optimization (SPBO)

    Notes:
        1. This algorithm is a weak algorithm in solving several problems
        2. It also consumes too much time because of ndim * pop_size updating times.

    Links:
       1. https://www.sciencedirect.com/science/article/abs/pii/S0965997820301484
       2. https://www.mathworks.com/matlabcentral/fileexchange/80991-student-psycology-based-optimization-spbo-algorithm

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
    >>> model = SPBO.OriginalSPBO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Das, B., Mukherjee, V., & Das, D. (2020). Student psychology based optimization algorithm: A new population based
    optimization algorithm for solving optimization problems. Advances in Engineering software, 146, 102804.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def evolve_agents(self, epoch):
        for jdx in range(0, self.problem.n_dims):
            idx_best = self.get_index_best(self.objs, self.problem.minmax)
            mid = self.generator.integers(1, self.pop_size - 1)
            x_mean = np.mean([agent.solution for agent in self.objs], axis=0)
            pop_new = []
            for idx in range(0, self.pop_size):
                if idx == idx_best:
                    k = self.generator.choice([1, 2])
                    j = self.generator.choice(
                        list(set(range(0, self.pop_size)) - {idx})
                    )
                    new_pos = self.g_best.solution + (-1) ** k * self.generator.random(
                        self.problem.n_dims
                    ) * (self.g_best.solution - self.objs[j].solution)
                elif idx < mid:
                    ## Good Student
                    if self.generator.random() > self.generator.random():
                        new_pos = self.g_best.solution + self.generator.random(
                            self.problem.n_dims
                        ) * (self.g_best.solution - self.objs[idx].solution)
                    else:
                        new_pos = (
                                self.objs[idx].solution
                                + self.generator.random(self.problem.n_dims)
                                * (self.g_best.solution - self.objs[idx].solution)
                                + self.generator.random()
                                * (self.objs[idx].solution - x_mean)
                        )
                else:
                    ## Average Student
                    if self.generator.random() > self.generator.random():
                        new_pos = self.objs[idx].solution + self.generator.random(
                            self.problem.n_dims
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

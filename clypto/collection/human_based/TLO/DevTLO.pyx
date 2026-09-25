#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 10:14, 18/03/2020 ----------%
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


cdef class DevTLO(AgentListOptimizer):
    """
    The developed version: Teaching Learning-based Optimization (TLO)

    Links:
       1. https://doi.org/10.5267/j.ijiec.2012.03.007

    Notes:
        + Use numpy np.array to make operations faster
        + The global best solution is used

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import TLO    >>> import numpy as np
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
    >>> model = TLO.DevTLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R. and Patel, V., 2012. An elitist teaching-learning-based optimization algorithm for solving
    complex constrained optimization problems. international journal of industrial engineering computations, 3(4), pp.535-560.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
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
        pop_new = []
        for idx in range(0, self.pop_size):
            ## Teaching Phrase
            TF = self.generator.integers(1, 3)  # 1 or 2 (never 3)
            list_pos = np.array([agent.solution for agent in self.objs])
            DIFF_MEAN = self.generator.random(self.problem.n_dims) * (
                    self.g_best.solution - TF * np.mean(list_pos, axis=0)
            )
            pos_new = self.objs[idx].solution + DIFF_MEAN
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )
        pop_child = []
        for idx in range(0, self.pop_size):
            ## Learning Phrase
            pos_new = self.objs[idx].solution.copy().astype(float)
            id_partner = self.generator.choice(
                np.setxor1d(np.array(range(self.pop_size)), np.array([idx]))
            )
            if self.compare_target(
                    self.objs[idx].target, self.objs[id_partner].target, self.problem.minmax
            ):
                pos_new += self.generator.random(self.problem.n_dims) * (
                        self.objs[idx].solution - self.objs[id_partner].solution
                )
            else:
                pos_new += self.generator.random(self.problem.n_dims) * (
                        self.objs[id_partner].solution - self.objs[idx].solution
                )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.objs = self.greedy_selection_population(
                self.objs, pop_child, self.problem.minmax
            )

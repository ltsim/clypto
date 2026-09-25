#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
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


cdef class MGTO(AgentListOptimizer):
    """
    The original version of: Modified Gorilla Troops Optimization (mGTO)

    Notes (parameters):
        1. pp (float): the probability of transition in exploration phase (p in the paper), default = 0.03

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import AGTO    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = AGTO.MGTO(epoch=1000, pop_size=50, pp=0.03)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mostafa, R. R., Gaheen, M. A., Abd ElAziz, M., Al-Betar, M. A., & Ewees, A. A. (2023). An improved gorilla
    troops optimizer for global optimization problems and feature selection. Knowledge-Based Systems, 110462.
    """

    cdef public object pp

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pp: float = 0.03,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pp (float): the probability of transition in exploration phase (p in the paper), default = 0.03
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pp"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pp = cy.validator(float, pp, (0, 1), "p1")

    cdef object amend_solution(self, object solution):
        condition = np.logical_and(
            self.problem.lb <= solution, solution <= self.problem.ub
        )
        random_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, random_pos)

    def evolve_agents(self, epoch):
        F = 1 + np.cos(2 * self.generator.random())
        C = F * (1 - epoch / self.epoch)
        L = C * self.generator.choice([-1, 1])

        ## Elite opposition-based learning
        pos_list = np.array([agent.solution for agent in self.objs])
        d_lb, d_ub = np.min(pos_list, axis=0), np.max(pos_list, axis=0)
        pos_list = d_lb + d_ub - pos_list
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.correct_solution(pos_list[idx])
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.objs = pop_new
        _, self.g_best = self.update_global_best_agent(self.objs, save=False)

        ## Exploration
        pop_new = []
        for idx in range(0, self.pop_size):
            if self.generator.random() < self.pp:
                pos_new = self.problem.generate_solution()
            else:
                if self.generator.random() >= 0.5:
                    rand_idx = self.generator.integers(0, self.pop_size)
                    pos_new = (self.generator.random() - C) * self.objs[
                        rand_idx
                    ].solution + L * self.generator.uniform(-C, C) * self.objs[
                                  idx
                              ].solution
                else:
                    id1, id2 = self.generator.choice(
                        list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
                    )
                    pos_new = (
                            self.objs[idx].solution
                            - L * (L * self.objs[idx].solution - self.objs[id1].solution)
                            + self.generator.random()
                            * (self.objs[idx].solution - self.objs[id2].solution)
                    )
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
        _, self.g_best = self.update_global_best_agent(self.objs, save=False)

        pos_list = np.array([agent.solution for agent in self.objs])
        ## Exploitation
        pop_new = []
        for idx in range(0, self.pop_size):
            if np.abs(C) >= 1:
                g = self.generator.choice([-0.5, 2.0])
                M = (np.abs(np.mean(pos_list, axis=0)) ** g) ** (1.0 / g)
                # print(M)
                p = self.generator.uniform(0, 1, self.problem.n_dims)
                pos_new = (
                        L
                        * M
                        * (self.objs[idx].solution - self.g_best.solution)
                        * (0.01 * np.tan(np.pi * (p - 0.5)))
                )
            else:
                Q = 2 * self.generator.random() - 1
                v = self.generator.uniform(0, 1)
                pos_new = self.g_best.solution - Q * (
                        self.g_best.solution - self.objs[idx].solution
                ) * np.tan(v * np.pi / 2)
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

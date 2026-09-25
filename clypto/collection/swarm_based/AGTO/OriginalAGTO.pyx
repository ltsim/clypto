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


cdef class OriginalAGTO(AgentListOptimizer):
    """
    The original version of: Artificial Gorilla Troops Optimization (AGTO)

    Links:
        1. https://doi.org/10.1002/int.22535
        2. https://www.mathworks.com/matlabcentral/fileexchange/95953-artificial-gorilla-troops-optimizer

    Notes (parameters):
        1. p1 (float): the probability of transition in exploration phase (p in the paper), default = 0.03
        2. p2 (float): the probability of transition in exploitation phase (w in the paper), default = 0.8
        3. beta (float): coefficient in updating equation, should be in [-5.0, 5.0], default = 3.0

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
    >>> model = AGTO.OriginalAGTO(epoch=1000, pop_size=50, p1=0.03, p2=0.8, beta=3.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021). Artificial gorilla troops optimizer: a new
    nature‐inspired metaheuristic algorithm for global optimization problems. International Journal of Intelligent Systems, 36(10), 5887-5958.
    """

    cdef public object p1
    cdef public object p2
    cdef public object beta

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        p1: float = 0.03,
        p2: float = 0.8,
        beta: float = 3.0,
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
            parameters=["epoch", "pop_size", "p1", "p2", "beta"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.p1 = cy.validator(float, p1, (0, 1), "p1")
        self.p2 = cy.validator(float, p2, (0, 1), "p2")
        self.beta = cy.validator(float, beta, [-10.0, 10.0], "beta")

    def evolve_agents(self, epoch):
        a = (np.cos(2 * self.generator.random()) + 1) * (1 - epoch / self.epoch)
        c = a * (2 * self.generator.random() - 1)
        ## Exploration
        pop_new = []
        for idx in range(0, self.pop_size):
            if self.generator.random() < self.p1:
                pos_new = self.problem.generate_solution()
            else:
                if self.generator.random() >= 0.5:
                    z = self.generator.uniform(-a, a, self.problem.n_dims)
                    rand_idx = self.generator.integers(0, self.pop_size)
                    pos_new = (self.generator.random() - a) * self.objs[
                        rand_idx
                    ].solution + c * z * self.objs[idx].solution
                else:
                    id1, id2 = self.generator.choice(
                        list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
                    )
                    pos_new = (
                            self.objs[idx].solution
                            - c * (c * self.objs[idx].solution - self.objs[id1].solution)
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
            if a >= self.p2:
                g = 2 ** c
                delta = (np.abs(np.mean(pos_list, axis=0)) ** g) ** (1.0 / g)
                pos_new = (
                        c * delta * (self.objs[idx].solution - self.g_best.solution)
                        + self.objs[idx].solution
                )
            else:
                if self.generator.random() >= 0.5:
                    h = self.generator.normal(0, 1, self.problem.n_dims)
                else:
                    h = self.generator.normal(0, 1)
                r1 = self.generator.random()
                pos_new = self.g_best.solution - (2 * r1 - 1) * (
                        self.g_best.solution - self.objs[idx].solution
                ) * (self.beta * h)
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

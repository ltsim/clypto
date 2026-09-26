#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class DevSARO(AgentListOptimizer):
    """
    The developed version: Search And Rescue Optimization (SARO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + se (float): [0.3, 0.8], social effect, default = 0.5
        + mu (int): maximum unsuccessful search number, belongs to range: [2, 2+int(self.pop_size/2)], default = 15

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import SARO    >>> import numpy as np
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
    >>> model = SARO.DevSARO(epoch=1000, pop_size=50, se = 0.5, mu = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        se: float = 0.5,
        mu: int = 10,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 15
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "se", "mu"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.se = cy.validator(float, se, (0, 1.0), "se")
        self.mu = cy.validator(int, mu, [2, 2 + int(self.pop_size / 2)], "mu")

    cdef void initialize_variables(self):
        self.dyn_USN = np.zeros(self.pop_size)

    cdef void initialization(self):
        AgentListOptimizer.initialization(self)
        if self.objs is None:
            self.objs = self.generate_agents(2 * self.pop_size)
        else:
            self.objs = self.objs + self.generate_agents(self.pop_size)
        self.pop = self.mirror__()

    cdef object amend_solution(self, object solution):
        condition = np.logical_and(
            self.problem.lb <= solution, solution <= self.problem.ub
        )
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, rand_pos)

    def evolve_agents(self, epoch):
        pop_x = [agent.copy() for agent in self.objs[: self.pop_size]]
        pop_m = [agent.copy() for agent in self.objs[self.pop_size:]]
        pop_new = []
        for idx in range(self.pop_size):
            ## Social Phase
            k = self.generator.choice(list(set(range(0, 2 * self.pop_size)) - {idx}))
            sd = pop_x[idx].solution - self.objs[k].solution
            #### Remove third loop here, also using random flight back when out of bound
            pos_new_1 = self.objs[k].solution + self.generator.uniform() * sd
            pos_new_2 = pop_x[idx].solution + self.generator.uniform() * sd
            condition = np.logical_and(
                self.generator.uniform(0, 1, self.problem.n_dims) < self.se,
                self.objs[k].target.fitness < pop_x[idx].target.fitness,
            )
            pos_new = np.where(condition, pos_new_1, pos_new_2)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(self.pop_size):
            if self.compare_target(
                    pop_new[idx].target, pop_x[idx].target, self.problem.minmax
            ):
                pop_m[self.generator.integers(0, self.pop_size)] = pop_x[idx].copy()
                pop_x[idx] = pop_new[idx].copy()
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1
        pop = pop_x.copy() + pop_m.copy()
        pop_new = []
        for idx in range(self.pop_size):
            ## Individual phase
            k1, k2 = self.generator.choice(
                list(set(range(0, 2 * self.pop_size)) - {idx}), 2, replace=False
            )
            #### Remove third loop here, and flight back strategy now be a random
            pos_new = self.g_best.solution + self.generator.uniform() * (
                    pop[k1].solution - pop[k2].solution
            )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_target(
                    pop_new[idx].target, pop_x[idx].target, self.problem.minmax
            ):
                pop_m[self.generator.integers(0, self.pop_size)] = pop_x[idx].copy()
                pop_x[idx] = pop_new[idx].copy()
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1
            if self.dyn_USN[idx] > self.mu:
                pop_x[idx] = self.generate_agent()
                self.dyn_USN[idx] = 0
        self.objs = pop_x + pop_m

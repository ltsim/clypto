#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 08:57, 14/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.human_based.FBIO.DevFBIO cimport DevFBIO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class OriginalFBIO(DevFBIO):
    """
    The original version of: Forensic-Based Investigation Optimization (FBIO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2020.106339
        2. https://ww2.mathworks.cn/matlabcentral/fileexchange/76299-forensic-based-investigation-algorithm-fbi

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import FBIO    >>> import numpy as np
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
    >>> model = FBIO.OriginalFBIO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Chou, J.S. and Nguyen, N.M., 2020. FBI inspired meta-optimization. Applied Soft Computing, 93, p.106339.
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
        super().__init__(epoch, pop_size, name=name, mode=mode)

    cdef object amend_solution(self, object solution):
        rd = self.generator.uniform(self.problem.lb, self.problem.ub)
        condition = np.logical_and(
            self.problem.lb <= solution, solution <= self.problem.ub
        )
        return np.where(condition, solution, rd)

    def evolve_agents(self, epoch):
        # Investigation team - team A
        # Step A1
        pop_new = []
        for idx in range(0, self.pop_size):
            n_change = self.generator.integers(0, self.problem.n_dims)
            nb1, nb2 = self.generator.choice(
                list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
            )
            # Eq.(2) in FBI Inspired Meta - Optimization
            pos_a = self.objs[idx].solution.copy()
            pos_a[n_change] = self.objs[idx].solution[n_change] + (
                    self.generator.uniform() - 0.5
            ) * 2 * (
                                      self.objs[idx].solution[n_change]
                                      - (self.objs[nb1].solution[n_change] + self.objs[nb2].solution[n_change])
                                      / 2
                              )
            ## Not good move here, change only 1 variable but check bound of all variable in solution
            pos_a = self.correct_solution(pos_a)
            agent = self.generate_empty_agent(pos_a)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_a)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )

        # Step A2
        list_fitness = np.array([agent.target.fitness for agent in self.objs])
        prob = self.probability__(list_fitness)
        pop_child = []
        for idx in range(0, self.pop_size):
            if self.generator.uniform() > prob[idx]:
                r1, r2, r3 = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx}), 3, replace=False
                )
                pos_a = self.objs[idx].solution.copy()
                Rnd = np.floor(self.generator.uniform() * self.problem.n_dims) + 1
                for j in range(0, self.problem.n_dims):
                    if self.generator.uniform() < self.generator.uniform() or Rnd == j:
                        pos_a[j] = (
                                self.g_best.solution[j]
                                + self.objs[r1].solution[j]
                                + self.generator.uniform()
                                * (self.objs[r2].solution[j] - self.objs[r3].solution[j])
                        )
                    ## In the original matlab code they do the else condition here, not good again because no need else here
                ## Same here, they do check the bound of all variable in solution
                ## pos_a = self.amend_position(pos_a, self.problem.lb, self.problem.ub)
            else:
                pos_a = self.generator.uniform(self.problem.lb, self.problem.ub)
            pos_a = self.correct_solution(pos_a)
            agent = self.generate_empty_agent(pos_a)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_a)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.objs = self.greedy_selection_population(
                pop_child, self.objs, self.problem.minmax
            )
        ## Persuing team - team B
        ## Step B1
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_b = self.objs[idx].solution.copy()
            for j in range(0, self.problem.n_dims):
                ### Eq.(6) in FBI Inspired Meta-Optimization
                pos_b[j] = self.generator.uniform() * self.objs[idx].solution[
                    j
                ] + self.generator.uniform() * (
                                   self.g_best.solution[j] - self.objs[idx].solution[j]
                           )
            pos_b = self.correct_solution(pos_b)
            agent = self.generate_empty_agent(pos_b)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_b)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )
        ## Step B2
        pop_child = []
        for idx in range(0, self.pop_size):
            rr = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            if self.compare_target(
                    self.objs[idx].target, self.objs[rr].target, self.problem.minmax
            ):
                ## Eq.(7) in FBI Inspired Meta-Optimization
                pos_b = (
                        self.objs[idx].solution
                        + self.generator.uniform(0, 1, self.problem.n_dims)
                        * (self.objs[rr].solution - self.objs[idx].solution)
                        + self.generator.uniform()
                        * (self.g_best.solution - self.objs[rr].solution)
                )
            else:
                ## Eq.(8) in FBI Inspired Meta-Optimization
                pos_b = (
                        self.objs[idx].solution
                        + self.generator.uniform(0, 1, self.problem.n_dims)
                        * (self.objs[idx].solution - self.objs[rr].solution)
                        + self.generator.uniform()
                        * (self.g_best.solution - self.objs[idx].solution)
                )
            pos_b = self.correct_solution(pos_b)
            agent = self.generate_empty_agent(pos_b)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_b)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.objs = self.greedy_selection_population(
                pop_child, self.objs, self.problem.minmax
            )

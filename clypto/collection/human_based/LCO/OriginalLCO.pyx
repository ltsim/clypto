#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
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


cdef class OriginalLCO(AgentListOptimizer):
    """
    The original version of: Life Choice-based Optimization (LCO)

    Links:
        1. https://doi.org/10.1007/s00500-019-04443-z

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r1 (float): [1.5, 4], coefficient factor, default = 2.35

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import LCO    >>> import numpy as np
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
    >>> model = LCO.OriginalLCO(epoch=1000, pop_size=50, r1 = 2.35)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Khatri, A., Gaba, A., Rana, K.P.S. and Kumar, V., 2020. A novel life choice-based optimizer. Soft Computing, 24(12), pp.9121-9141.
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        r1: float = 2.35,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r1 (float): coefficient factor
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "r1"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.r1 = cy.validator(float, r1, [1.0, 3.0], "r1")
        self.n_agents = int(np.ceil(np.sqrt(self.pop_size)))

    def evolve_agents(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            prob = self.generator.random()
            if prob > 0.875:  # Update using Eq. 1, update from n best position
                temp = np.array(
                    [
                        self.generator.random() * self.objs[j].solution
                        for j in range(0, self.n_agents)
                    ]
                )
                temp = np.mean(temp, axis=0)
            elif prob < 0.7:  # Update using Eq. 2-6
                f1 = 1 - epoch / self.epoch
                f2 = 1 - f1
                prev_pos = (
                    self.g_best.solution if idx == 0 else self.objs[idx - 1].solution
                )
                best_diff = (
                        f1 * self.r1 * (self.g_best.solution - self.objs[idx].solution)
                )
                better_diff = f2 * self.r1 * (prev_pos - self.objs[idx].solution)
                temp = (
                        self.objs[idx].solution
                        + self.generator.random() * better_diff
                        + self.generator.random() * best_diff
                )
            else:
                temp = (
                        self.problem.ub
                        - (self.objs[idx].solution - self.problem.lb)
                        * self.generator.random()
                )
            pos_new = self.correct_solution(temp)
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

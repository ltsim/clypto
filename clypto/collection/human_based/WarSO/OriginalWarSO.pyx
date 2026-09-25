#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 17:41, 21/05/2022 ----------%
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


cdef class OriginalWarSO(AgentListOptimizer):
    """
    The original version of: War Strategy Optimization (WarSO) algorithm

    Links:
       1. https://www.researchgate.net/publication/358806739_War_Strategy_Optimization_Algorithm_A_New_Effective_Metaheuristic_Algorithm_for_Global_Optimization

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + rr (float): [0.1, 0.9], the probability of switching position updating, default=0.1

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import WarSO    >>> import numpy as np
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
    >>> model = WarSO.OriginalWarSO(epoch=1000, pop_size=50, rr=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ayyarao, Tummala SLV, and Polamarasetty P. Kumar. "Parameter estimation of solar PV models with a new proposed
    war strategy optimization algorithm." International Journal of Energy Research (2022).
    """

    cdef public object rr
    cdef public object wl
    cdef public object wg

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        rr: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            rr (float): the probability of switching position updating, default=0.1
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "rr"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.rr = cy.validator(float, rr, (0.0, 1.0), "rr")

    cdef void initialize_variables(self):
        self.wl = 2 * np.ones(self.pop_size)
        self.wg = np.zeros(self.pop_size)

    def evolve_agents(self, epoch):
        pop_sorted, indices = self.get_sorted_indices_population(
            self.objs, self.problem.minmax
        )
        self.wl = self.wl[indices]
        self.wg = self.wg[indices]
        com = self.generator.permutation(self.pop_size)
        for idx in range(0, self.pop_size):
            r1 = self.generator.random()
            if r1 < self.rr:
                pos_new = 2 * r1 * (
                    self.g_best.solution - self.objs[com[idx]].solution
                ) + self.wl[idx] * self.generator.random() * (
                    pop_sorted[idx].solution - self.objs[idx].solution
                )
            else:
                pos_new = 2 * r1 * (
                    pop_sorted[idx].solution - self.g_best.solution
                ) + self.generator.random() * (
                    self.wl[idx] * self.g_best.solution - self.objs[idx].solution
                )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(
                agent.target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx] = agent
                self.wg[idx] += 1
                self.wl[idx] = 1 * self.wl[idx] * (1 - self.wg[idx] / self.epoch) ** 2

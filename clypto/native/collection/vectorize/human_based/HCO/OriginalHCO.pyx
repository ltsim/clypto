#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 08:57, 12/03/2023 ----------%
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


cdef class OriginalHCO(AgentListOptimizer):
    """
    The original version of: Human Conception Optimizer (HCO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/124200-human-conception-optimizer-hco
        2. https://www.nature.com/articles/s41598-022-25031-6

    Notes:
        1. This algorithm shares some similarities with the PSO algorithm (equations)
        2. The implementation of Matlab code is kinda different to the paper

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wfp (float): (0, 1.) - weight factor for probability of fitness selection, default=0.65
        + wfv (float): (0, 1.0) - weight factor for velocity update stage, default=0.05
        + c1 (float): (0., 3.0) - acceleration coefficient, same as PSO, default=1.4
        + c2 (float): (0., 3.0) - acceleration coefficient, same as PSO, default=1.4

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import HCO    >>> import numpy as np
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
    >>> model = HCO.OriginalHCO(epoch=1000, pop_size=50, wfp=0.65, wfv=0.05, c1=1.4, c2=1.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Acharya, D., & Das, D. K. (2022). A novel Human Conception Optimizer for solving optimization problems. Scientific Reports, 12(1), 21631.
    """

    cdef public object wfp
    cdef public object wfv
    cdef public object c1
    cdef public object c2
    cdef public object vec
    cdef public object pop_p

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        wfp: float = 0.65,
        wfv: float = 0.05,
        c1: float = 1.4,
        c2: float = 1.4,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wfp (float): weight factor for probability of fitness selection, default=0.65
            wfv (float): weight factor for velocity update stage, default=0.05
            c1 (float): acceleration coefficient, same as PSO, default=1.4
            c2 (float): acceleration coefficient, same as PSO, default=1.4
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "wfp", "wfv", "c1", "c2"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.wfp = cy.validator(float, wfp, [0, 1.0], "wfp")
        self.wfv = cy.validator(float, wfv, [0, 1.0], "wfv")
        self.c1 = cy.validator(float, c1, [0.0, 100.0], "c1")
        self.c2 = cy.validator(float, c2, [1.0, 100.0], "c2")

    cdef void initialization(self):
        AgentListOptimizer.initialization(self)
        pop_op = []
        for idx in range(0, self.pop_size):
            pos_new = self.problem.ub + self.problem.lb - self.objs[idx].solution
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_op.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_op = self.update_target_for_population(pop_op)
            self.objs = self.greedy_selection_population(
                self.objs, pop_op, self.problem.minmax
            )
        _, (best,), (worst,) = self.get_special_agents(
            self.objs, n_best=1, n_worst=1, minmax=self.problem.minmax
        )
        pfit = (
                       worst.target.fitness - best.target.fitness
               ) * self.wfp + best.target.fitness
        for idx in range(0, self.pop_size):
            if self.compare_fitness(
                    pfit, self.objs[idx].target.fitness, self.problem.minmax
            ):
                while True:
                    agent = self.generate_agent()
                    if self.compare_fitness(
                            agent.target.fitness, pfit, self.problem.minmax
                    ):
                        self.objs[idx] = agent
                        break
        self.vec = self.generator.uniform(
            self.problem.lb, self.problem.ub, (self.pop_size, self.problem.n_dims)
        )
        self.pop_p = [agent.copy() for agent in self.objs]
        self.pop = self.mirror__()

    def evolve_agents(self, epoch):
        lamda = self.generator.random()
        neu = 2
        fits = np.array([agent.target.fitness for agent in self.objs])
        fit_mean = np.mean(fits)
        RR = (self.g_best.target.fitness - fits) ** 2
        rr = (fit_mean - fits) ** 2
        ll = RR - rr
        LL = self.g_best.target.fitness - fit_mean
        VV = lamda * (ll / (4 * neu * LL))
        pop_new = []
        for idx in range(0, self.pop_size):
            a1 = self.pop_p[idx].solution - self.objs[idx].solution
            a2 = self.g_best.solution - self.objs[idx].solution
            self.vec[idx] = (
                    self.wfv * (VV[idx] + self.vec[idx])
                    + self.c1 * a1 * np.sin(2 * np.pi * epoch / self.epoch)
                    + self.c2 * a2 * np.sin(2 * np.pi * epoch / self.epoch)
            )
            pos_new = self.objs[idx].solution + self.vec[idx]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        for idx in range(0, self.pop_size):
            if self.compare_target(
                    pop_new[idx].target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx] = pop_new[idx].copy()
                if self.compare_target(
                        pop_new[idx].target, self.pop_p[idx].target, self.problem.minmax
                ):
                    self.pop_p[idx] = pop_new[idx].copy()

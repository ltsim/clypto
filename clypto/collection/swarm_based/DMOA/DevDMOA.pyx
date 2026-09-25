#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 17:48, 21/05/2022 ----------%
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


cdef class DevDMOA(AgentListOptimizer):
    """
    The developed version of: Dwarf Mongoose Optimization Algorithm (DMOA)

    Notes:
        1. Removed the parameter n_baby_sitter
        2. Changed in section # Next Mongoose position
        3. Removed the meaningless variable tau

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import DMOA    >>> import numpy as np
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
    >>> model = DMOA.DevDMOA(epoch=1000, pop_size=50, peep = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    cdef public object peep
    cdef public object C
    cdef public object L

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        peep: float = 2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "peep"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.peep = cy.validator(float, peep, [1, 10.0], "peep")

    cdef void initialize_variables(self):
        self.C = np.zeros(self.pop_size)
        self.L = np.round(0.6 * self.epoch)

    def evolve_agents(self, epoch):
        ## Abandonment Counter
        CF = (1.0 - epoch / self.epoch) ** (2.0 * epoch / self.epoch)
        fit_list = np.array([agent.target.fitness for agent in self.objs])
        mean_cost = np.mean(fit_list)
        fi = np.exp(-fit_list / mean_cost)

        ## Foraging led by Alpha female
        for idx in range(0, self.pop_size):
            alpha = self.get_index_roulette_wheel_selection(fi)
            k = self.generator.choice(list(set(range(0, self.pop_size)) - {idx, alpha}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.objs[alpha].solution + phi * (
                    self.objs[alpha].solution - self.objs[k].solution
            )
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            if self.compare_target(
                    agent.target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx] = agent
            else:
                self.C[idx] += 1

        ## Scout group
        SM = np.zeros(self.pop_size)
        for idx in range(0, self.pop_size):
            k = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.objs[idx].solution + phi * (
                    self.objs[idx].solution - self.objs[k].solution
            )
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            ## Sleeping mould
            SM[idx] = (agent.target.fitness - self.objs[idx].target.fitness) / (
                    np.max([agent.target.fitness, self.objs[idx].target.fitness])
                    + self.EPSILON
            )
            if self.compare_target(
                    agent.target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx] = agent
            else:
                self.C[idx] += 1

        ## Baby sitters
        for idx in range(0, self.pop_size):
            if self.C[idx] >= self.L:
                self.objs[idx] = self.generate_agent()
                self.C[idx] = 0

        ## Next Mongoose position
        new_tau = np.mean(SM)
        for idx in range(0, self.pop_size):
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            if new_tau > SM[idx]:
                new_pos = self.g_best.solution - CF * phi * (
                        self.g_best.solution - SM[idx] * self.objs[idx].solution
                )
            else:
                new_pos = self.objs[idx].solution + CF * phi * (
                        self.g_best.solution - SM[idx] * self.objs[idx].solution
                )
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            if self.compare_target(
                    agent.target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx] = agent

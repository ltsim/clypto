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


cdef class OriginalDMOA(AgentListOptimizer):
    """
    The original version of: Dwarf Mongoose Optimization Algorithm (DMOA)

    Links:
        1. https://doi.org/10.1016/j.cma.2022.114570
        2. https://www.mathworks.com/matlabcentral/fileexchange/105125-dwarf-mongoose-optimization-algorithm

    Notes:
        1. The Matlab code differs slightly from the original paper
        2. There are some parameters and equations in the Matlab code that don't seem to have any meaningful purpose.
        3. The algorithm seems to be weak on solving several problems.

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
    >>> model = DMOA.OriginalDMOA(epoch=1000, pop_size=50, n_baby_sitter = 3, peep = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Agushaka, J. O., Ezugwu, A. E., & Abualigah, L. (2022). Dwarf mongoose optimization algorithm.
    Computer methods in applied mechanics and engineering, 391, 114570.
    """

    cdef public object n_baby_sitter
    cdef public object peep
    cdef public object n_scout
    cdef public object C
    cdef public object tau
    cdef public object L

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_baby_sitter: int = 3,
        peep: float = 2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "n_baby_sitter", "peep"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.n_baby_sitter = cy.validator(int, n_baby_sitter, [2, 10], "n_baby_sitter")
        self.peep = cy.validator(float, peep, [1, 10.0], "peep")
        self.n_scout = self.pop_size - self.n_baby_sitter

    cdef void initialize_variables(self):
        self.C = np.zeros(self.pop_size)
        self.tau = -np.inf
        self.L = np.round(0.6 * self.problem.n_dims * self.n_baby_sitter)

    def evolve_agents(self, epoch):
        ## Abandonment Counter
        CF = (1.0 - epoch / self.epoch) ** (2.0 * epoch / self.epoch)
        fit_list = np.array([agent.target.fitness for agent in self.objs])
        mean_cost = np.mean(fit_list)
        fi = np.exp(-fit_list / mean_cost)
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
            SM[idx] = (agent.target.fitness - self.objs[idx].target.fitness) / np.max(
                [agent.target.fitness, self.objs[idx].target.fitness]
            )
            if self.compare_target(
                    agent.target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx] = agent
            else:
                self.C[idx] += 1
        ## Baby sitters
        for idx in range(0, self.n_baby_sitter):
            if self.C[idx] >= self.L:
                self.objs[idx] = self.generate_agent()
                self.C[idx] = 0
        ## Next Mongoose position
        new_tau = np.mean(SM)
        for idx in range(0, self.pop_size):
            M = SM[idx] * self.objs[idx].solution / self.objs[idx].solution
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            if new_tau > self.tau:
                new_pos = self.objs[
                              idx
                          ].solution - CF * phi * self.generator.random() * (
                                  self.objs[idx].solution - M
                          )
            else:
                new_pos = self.objs[
                              idx
                          ].solution + CF * phi * self.generator.random() * (
                                  self.objs[idx].solution - M
                          )
            self.tau = new_tau
            new_pos = self.correct_solution(new_pos)
            self.objs[idx] = self.generate_agent(new_pos)

#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 22:46, 26/10/2022 ----------%
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


cdef class IARO(AgentListOptimizer):
    """
    The improved version of: Improved Artificial Rabbits Optimization (IARO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2022.105082
        2. https://www.mathworks.com/matlabcentral/fileexchange/110250-artificial-rabbits-optimization-aro

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import ARO    >>> import numpy as np
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
    >>> model = ARO.IARO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, L., Cao, Q., Zhang, Z., Mirjalili, S., & Zhao, W. (2022). Artificial rabbits optimization: A new bio-inspired
    meta-heuristic algorithm for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 114, 105082.
    """

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
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
        theta = 2 * (1 - (epoch + 1) / self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            L = (np.exp(1) - np.exp((epoch / self.epoch) ** 2)) * (
                np.sin(2 * np.pi * self.generator.random())
            )
            temp = np.zeros(self.problem.n_dims)
            rd_index = self.generator.choice(
                np.arange(0, self.problem.n_dims),
                int(np.ceil(self.generator.random() * self.problem.n_dims)),
                replace=False,
            )
            temp[rd_index] = 1
            R = L * temp  # Eq 2
            A = 2 * np.log(1.0 / self.generator.random()) * theta  # Eq. 15
            if A > 1:  # # detour foraging strategy
                rand_idx = self.generator.integers(0, self.pop_size)
                pos_new = (
                        self.objs[rand_idx].solution
                        + R * (self.objs[idx].solution - self.objs[rand_idx].solution)
                        + np.round(0.5 * (0.05 + self.generator.random()))
                        * self.generator.normal(0, 1)
                )  # Eq. 1
            else:  # Random hiding stage
                gr = np.zeros(self.problem.n_dims)
                rd_index = self.generator.choice(
                    np.arange(0, self.problem.n_dims),
                    int(np.ceil(self.generator.random() * self.problem.n_dims)),
                    replace=False,
                )
                gr[rd_index] = 1  # Eq. 12
                H = self.generator.normal(0, 1) * (epoch / self.epoch)  # Eq. 8
                b = self.objs[idx].solution + H * gr * self.objs[idx].solution  # Eq. 13
                pos_new = self.objs[idx].solution + R * (
                        self.generator.random() * b - self.objs[idx].solution
                )  # Eq. 11
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
                self.objs, pop_new, minmax=self.problem.minmax
            )

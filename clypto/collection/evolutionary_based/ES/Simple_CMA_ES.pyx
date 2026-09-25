#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 18:14, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent
from clypto.optimizer._native.population cimport NativePopulation


cdef class Simple_CMA_ES(AgentListOptimizer):
    """
    The simple version of: Covariance Matrix Adaptation Evolution Strategy (Simple-CMA-ES)

    Links:
        1. Inspired from this version: https://github.com/jenkspt/CMA-ES
        2. https://ieeexplore.ieee.org/abstract/document/6790628/

    Examples
    ~~~~~~~~
    >>> from clypto.collection.evolutionary_based import ES    >>> import numpy as np
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
    >>> model = ES.Simple_CMA_ES(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. Evolutionary computation, 9(2), 159-195.
    """

    cdef public object mu

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
            pop_size (int): number of population size (miu in the paper), default = 100
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

    cdef void before_main_loop(self):
        self.mu = int(np.round(self.pop_size / 2))

    def evolve_agents(self, epoch):
        pos_list = np.array([agent.solution for agent in self.objs]).T
        pop_sorted = self.get_sorted_population(self.objs, self.problem.minmax)
        pos_topk = np.array([agent.solution for agent in pop_sorted[: self.mu]]).T
        # Covariance of top k but using mean of entire population
        centered = pos_list - pos_topk.mean(1, keepdims=True)
        C = (centered @ centered.T) / (self.mu - 1)
        # Eigenvalue decomposition
        w, E = np.linalg.eigh(C)
        if np.any(np.diag(w) < 0):
            w[w < 0] = 0
        # Generate new population
        # Sample from multivariate gaussian with mean of topk
        N = self.generator.normal(size=(self.problem.n_dims, self.pop_size))
        X = pos_topk.mean(1, keepdims=True) + (E @ np.diag(np.sqrt(w)) @ N)
        X = X.T
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.correct_solution(X[idx])
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    pop_new[-1], self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )

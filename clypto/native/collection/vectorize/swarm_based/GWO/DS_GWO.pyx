#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DS_GWO(LegacyNativeOptimizer):
    """
    The original version of: Diversity enhanced Strategy based Grey Wolf Optimizer (DS-GWO)

    This implementation includes:
        1. Group-stage competition mechanism
        2. Exploration-exploitation balance mechanism

    Links:
        1. https://doi.org/10.1016/j.knosys.2022.109100

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.DS_GWO(epoch=1000, pop_size=50, explore_ratio=0.4, n_groups=5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Jiang, Jianhua, Ziying Zhao, Yutong Liu, Weihua Li, and Huan Wang. "DSGWO: An improved grey wolf optimizer with diversity enhanced strategy based on group-stage competition and balance mechanisms." Knowledge-Based Systems 250 (2022): 109100.
    """

    cdef public double explore_ratio
    cdef public int n_groups
    cdef public object explore_epoch
    cdef public object delta_candidates
    cdef public object alpha
    cdef public object beta

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        explore_ratio: float = 0.4,
        n_groups: int = 5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            explore_ratio (float): ratio to control exploration, default = 0.4
            n_groups (int): number of groups for group-stage competition, default = 5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "explore_ratio", "n_groups"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.explore_ratio = cy.validator(float, explore_ratio, [0.0, 1.0], "explore_ratio")
        self.n_groups = cy.validator(int, n_groups, [5, 100], "n_groups")

    cdef void initialize_variables(self):
        self.explore_epoch = int(self.epoch * self.explore_ratio)

    cdef void before_main_loop(self):
        self.group_stage_competition()

    def get_coefficients(self, a: float) -> tuple:
        A = a * (2 * self.generator.random(self.problem.n_dims) - 1)
        C = 2 * self.generator.random(self.problem.n_dims)
        return A, C

    def group_stage_competition(self):
        cdef NativePopulation pop = self.pop
        # Divide population into n_groups
        group_size = self.pop_size // self.n_groups
        self.delta_candidates = []

        for idx in range(self.n_groups):
            start_idx = idx * group_size
            if idx == self.n_groups - 1:  # Last group takes remaining wolves
                end_idx = self.pop_size
            else:
                end_idx = (idx + 1) * group_size

            # Best wolf in the group
            group = pop.take(np.arange(start_idx, end_idx))
            self.delta_candidates.append(group.agent(self.sorted_order(group)[0]))

        # Set alpha wolf (best among all delta candidates)
        fits = [agent.target.fitness for agent in self.delta_candidates]
        order = np.argsort(fits)
        if self.problem.minmax == "max":
            order = order[::-1]
        self.alpha = self.delta_candidates[order[0]].copy()

        # Set beta wolf (delta candidate farthest from alpha)
        delta_pos = np.array([agent.solution for agent in self.delta_candidates])
        distances = np.linalg.norm(delta_pos - self.alpha.solution, axis=1)
        beta_idx = np.argmax(distances)
        self.beta = self.delta_candidates[beta_idx].copy()

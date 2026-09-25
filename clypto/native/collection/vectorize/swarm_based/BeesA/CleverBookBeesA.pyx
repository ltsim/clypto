#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
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


cdef class CleverBookBeesA(AgentListOptimizer):
    """
    The original version of: Bees Algorithm (CB-BeesA)

    Notes:
        + This version is based on ABC in the book Clever Algorithms
        + Improved the function search_neighborhood__

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_elites (int): number of employed bees which provided for good location
        + n_others (int): number of employed bees which provided for other location
        + patch_size (float): patch_variables = patch_variables * patch_reduction
        + patch_reduction (float): the reduction factor
        + n_sites (int): 3 bees (employed bees, onlookers and scouts),
        + n_elite_sites (int): 1 good partition

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import BeesA    >>> import numpy as np
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
    >>> model = BeesA.CleverBookBeesA(epoch=1000, pop_size=50, n_elites = 16, n_others = 4,
    >>>             patch_size = 5.0, patch_reduction = 0.985, n_sites = 3, n_elite_sites = 1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] D. T. Pham, Ghanbarzadeh A., Koc E., Otri S., Rahim S., and M.Zaidi. The bees algorithm - a novel tool
    for complex optimisation problems. In Proceedings of IPROMS 2006 Conference, pages 454–461, 2006.
    """

    cdef public object n_elites
    cdef public object n_others
    cdef public object patch_size
    cdef public object patch_reduction
    cdef public object n_sites
    cdef public object n_elite_sites

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_elites: int = 16,
        n_others: int = 4,
        patch_size: float = 5.0,
        patch_reduction: float = 0.985,
        n_sites: int = 3,
        n_elite_sites: int = 1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_elites (int): number of employed bees which provided for good location
            n_others (int): number of employed bees which provided for other location
            patch_size (float): patch_variables = patch_variables * patch_reduction
            patch_reduction (float): the reduction factor
            n_sites (int): 3 bees (employed bees, onlookers and scouts),
            n_elite_sites (int): 1 good partition
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "n_elites",
                "n_others",
                "patch_size",
                "patch_reduction",
                "n_sites",
                "n_elite_sites",
            ],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.n_elites = cy.validator(int, n_elites, [4, 20], "n_elites")
        self.n_others = cy.validator(int, n_others, [2, 5], "n_others")
        self.patch_size = cy.validator(float, patch_size, [2, 10], "patch_size")
        self.patch_reduction = cy.validator(float, patch_reduction, (0, 1.0), "patch_reduction")
        self.n_sites = cy.validator(int, n_sites, [2, 5], "n_sites")
        self.n_elite_sites = cy.validator(int, n_elite_sites, [1, 3], "n_elite_sites")

    def search_neighborhood__(self, parent=None, neigh_size=None):
        pop_neigh = []
        for idx in range(0, neigh_size):
            t1 = self.generator.integers(0, len(parent.solution) - 1)
            new_bee = parent.solution.copy()
            new_bee[t1] = (
                (parent.solution[t1] + self.generator.uniform() * self.patch_size)
                if self.generator.uniform() < 0.5
                else (parent.solution[t1] - self.generator.uniform() * self.patch_size)
            )
            pos_new = self.correct_solution(new_bee)
            agent = self.generate_empty_agent(pos_new)
            pop_neigh.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_neigh[-1].target = self.get_target(pos_new)
        pop_neigh = self.update_target_for_population(pop_neigh)
        return self.get_best_agent(pop_neigh, self.problem.minmax)

    def evolve_agents(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx < self.n_sites:
                if idx < self.n_elite_sites:
                    neigh_size = self.n_elites
                else:
                    neigh_size = self.n_others
                agent = self.search_neighborhood__(self.objs[idx], neigh_size)
            else:
                agent = self.generate_agent()
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )

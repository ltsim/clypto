#!/usr/bin/env python
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class CleverBookBeesA(_LegacyOptimizer):
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
    >>> from clypto.native.collection.legacy.swarm_based import BeesA    >>> import numpy as np
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
            **kwargs: object
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
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_elites = self.validator.check_int("n_elites", n_elites, [4, 20])
        self.n_others = self.validator.check_int("n_others", n_others, [2, 5])
        self.patch_size = self.validator.check_float("patch_size", patch_size, [2, 10])
        self.patch_reduction = self.validator.check_float(
            "patch_reduction", patch_reduction, (0, 1.0)
        )
        self.n_sites = self.validator.check_int("n_sites", n_sites, [2, 5])
        self.n_elite_sites = self.validator.check_int(
            "n_elite_sites", n_elite_sites, [1, 3]
        )
        self.set_parameters(
            [
                "epoch",
                "pop_size",
                "n_elites",
                "n_others",
                "patch_size",
                "patch_reduction",
                "n_sites",
                "n_elite_sites",
            ]
        )
        self.sort_flag = True

    def search_neighborhood__(self, parent=None, neigh_size=None):
        """
        Search 1 best position in neigh_size position
        """
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx < self.n_sites:
                if idx < self.n_elite_sites:
                    neigh_size = self.n_elites
                else:
                    neigh_size = self.n_others
                agent = self.search_neighborhood__(self.pop[idx], neigh_size)
            else:
                agent = self.generate_agent()
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )

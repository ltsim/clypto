#!/usr/bin/env python
# Created by "Thieu" at 14:56, 19/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.legacy.evolutionary_based.CRO.OriginalCRO cimport OriginalCRO


cdef class OCRO(OriginalCRO):
    """
    The original version of: Opposition-based Coral Reefs Optimization (OCRO)

    Links:
        1. https://dx.doi.org/10.2991/ijcis.d.190930.003

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + po (float): [0.2, 0.5], the rate between free/occupied at the beginning
        + Fb (float): [0.6, 0.9], BroadcastSpawner/ExistingCorals rate
        + Fa (float): [0.05, 0.3], fraction of corals duplicates its self and tries to settle in a different part of the reef
        + Fd (float): [0.05, 0.5], fraction of the worse health corals in reef will be applied depredation
        + Pd (float): [0.1, 0.7], the maximum of probability of depredation
        + GCR (float): [0.05, 0.2], probability for mutation process
        + gamma_min (float): [0.01, 0.1] factor for mutation process
        + gamma_max (float): [0.1, 0.5] factor for mutation process
        + n_trials (int): [2, 10], number of attempts for a larvar to set in the reef
        + restart_count (int): [10, 100], reset the whole population after global best solution is not improved after restart_count times

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.evolutionary_based import CRO    >>> import numpy as np
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
    >>> model = CRO.OCRO(epoch=1000, pop_size=50, po = 0.4, Fb = 0.9, Fa = 0.1, Fd = 0.1, Pd = 0.5, GCR = 0.1, gamma_min = 0.02, gamma_max = 0.2, n_trials = 5, restart_count = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Nguyen, T., Nguyen, B.M. and Nguyen, G., 2019. Efficient time-series forecasting using
    neural network and opposition-based coral reefs optimization. International Journal of Computational
    Intelligence Systems, 12(2), p.1144.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            po: float = 0.4,
            Fb: float = 0.9,
            Fa: float = 0.1,
            Fd: float = 0.1,
            Pd: float = 0.5,
            GCR: float = 0.1,
            gamma_min: float = 0.02,
            gamma_max: float = 0.2,
            n_trials: int = 3,
            restart_count: int = 20,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            po (float): the rate between free/occupied at the beginning
            Fb (float): BroadcastSpawner/ExistingCorals rate
            Fa (float): fraction of corals duplicates its self and tries to settle in a different part of the reef
            Fd (float): fraction of the worse health corals in reef will be applied depredation
            Pd (float): Probability of depredation
            GCR (float): probability for mutation process
            gamma_min (float): [0.01, 0.1] factor for mutation process
            gamma_max (float): [0.1, 0.5] factor for mutation process
            n_trials (int): number of attempts for a larva to set in the reef.
            restart_count (int): reset the whole population after global best solution is not improved after restart_count times
        """
        super().__init__(
            epoch,
            pop_size,
            po,
            Fb,
            Fa,
            Fd,
            Pd,
            GCR,
            gamma_min,
            gamma_max,
            n_trials,
            **kwargs
        )
        self.restart_count = self.validator.check_int(
            "restart_count", restart_count, [2, int(epoch / 2)]
        )
        self.set_parameters(
            [
                "epoch",
                "pop_size",
                "po",
                "Fb",
                "Fa",
                "Fd",
                "Pd",
                "GCR",
                "gamma_min",
                "gamma_max",
                "n_trials",
                "restart_count",
            ]
        )
        self.sort_flag = False

    def initialize_variables(self):
        self.reset_count = 0

    def local_search__(self, pop=None):
        pop_new = []
        for idx in range(0, len(pop)):
            random_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
            condition = self.generator.random(self.problem.n_dims) < 0.5
            pos_new = np.where(condition, self.g_best.solution, random_pos)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        return self.update_target_for_population(pop_new)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Broadcast Spawning Brooding
        larvae = self.broadcast_spawning_brooding__()
        self.larvae_setting__(larvae)
        ## Asexual Reproduction
        num_duplicate = int(len(self.occupied_idx_list) * self.Fa)
        pop_best = [self.pop[idx] for idx in self.occupied_idx_list]
        pop_best = self.get_sorted_and_trimmed_population(
            pop_best, num_duplicate, self.problem.minmax
        )
        pop_local_search = self.local_search__(pop_best)
        self.larvae_setting__(pop_local_search)
        ## Depredation
        if self.generator.random() < self.dyn_Pd:
            num__depredation__ = int(len(self.occupied_idx_list) * self.Fd)
            idx_list_sorted = self.sort_occupied_reef__()
            selected_depredator = idx_list_sorted[-num__depredation__:]
            for idx in selected_depredator:
                ### Using opposition-based leanring
                pos_oppo = self.generate_opposition_solution(self.pop[idx], self.g_best)
                agent = self.generate_agent(pos_oppo)
                if self.compare_target(
                        agent.target, self.pop[idx].target, self.problem.minmax
                ):
                    self.pop[idx] = agent
                else:
                    self.occupied_idx_list = self.occupied_idx_list[
                        ~np.isin(self.occupied_idx_list, [idx])
                    ]
                    self.occupied_list[idx] = 0
        if self.dyn_Pd <= self.Pd:
            self.dyn_Pd += self.alpha
        if self.G1 >= self.gamma_min:
            self.G1 -= self.gama
        self.reset_count += 1
        local_best = self.get_best_agent(self.pop, self.problem.minmax)
        if self.compare_target(
                local_best.target, self.g_best.target, self.problem.minmax
        ):
            self.reset_count = 0
        if self.reset_count == self.restart_count:
            self.pop = self.generate_population(self.pop_size)
            self.occupied_list = np.zeros(self.pop_size)
            self.occupied_idx_list = self.generator.choice(
                range(self.pop_size), self.num_occupied, replace=False
            )
            self.occupied_list[self.occupied_idx_list] = 1
            self.reset_count = 0

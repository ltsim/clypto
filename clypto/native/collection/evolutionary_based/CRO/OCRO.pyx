#!/usr/bin/env python
# Created by "Thieu" at 14:56, 19/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.evolutionary_based.CRO.OriginalCRO cimport OriginalCRO


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
    >>> from clypto.native.collection.evolutionary_based import CRO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = CRO.OCRO(epoch=1000, pop_size=50, po = 0.4, Fb = 0.9, Fa = 0.1, Fd = 0.1, Pd = 0.5, GCR = 0.1, gamma_min = 0.02, gamma_max = 0.2, n_trials = 5, restart_count = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Nguyen, T., Nguyen, B.M. and Nguyen, G., 2019. Efficient time-series forecasting using
    neural network and opposition-based coral reefs optimization. International Journal of Computational
    Intelligence Systems, 12(2), p.1144.
    """

    cdef public int restart_count

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
        self.restart_count = cy.validator(int, restart_count, [2, int(epoch / 2)], "restart_count")
        self.parameters = [ "epoch", "pop_size", "po", "Fb", "Fa", "Fd", "Pd", "GCR", "gamma_min", "gamma_max", "n_trials", "restart_count", ]
        self.sort_flag = False

    def initialize_variables(self):
        self.reset_count = 0

    def local_search__(self, pop=None):
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx in range(0, len(pop)):
            random_pos = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            condition = self.generator.random(self.problem.n_dims) < 0.5
            x = np.where(condition, self.g_best.solution, random_pos)
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            n_population.append(agent)
            if self.mode == "sequential":
                n_population[-1].evaluate(self.problem)
        return self.population.evaluate(n_population, self.mode)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Broadcast Spawning Brooding
        larvae = self.broadcast_spawning_brooding__()
        self.larvae_setting__(larvae)
        ## Asexual Reproduction
        num_duplicate = int(len(self.occupied_idx_list) * self.Fa)
        pop_best = [self.population[idx] for idx in self.occupied_idx_list]
        pop_best = cy.sort_agents(pop_best, self.problem.sense)[:num_duplicate]
        pop_local_search = self.local_search__(pop_best)
        self.larvae_setting__(pop_local_search)
        ## Depredation
        if self.generator.random() < self.dyn_Pd:
            num__depredation__ = int(len(self.occupied_idx_list) * self.Fd)
            idx_list_sorted = self.sort_occupied_reef__()
            selected_depredator = idx_list_sorted[-num__depredation__:]
            for idx in selected_depredator:
                ### Using opposition-based leanring
                pos_oppo = cy.opposite_solution(self.problem, self.generator, self.population[idx], self.g_best)
                agent = self.population.generate_agent(pos_oppo)
                if cy.is_better(agent, self.population[idx], self.problem.sense):
                    self.population[idx] = agent
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
        local_best = cy.duplicate_agent(self.population.sort()[0])
        if cy.is_better(local_best, self.g_best, self.problem.sense):
            self.reset_count = 0
        if self.reset_count == self.restart_count:
            self.population = self.population.generate(pop_size)
            self.occupied_list = np.zeros(pop_size)
            self.occupied_idx_list = self.generator.choice(
                range(pop_size), self.num_occupied, replace=False
            )
            self.occupied_list[self.occupied_idx_list] = 1
            self.reset_count = 0

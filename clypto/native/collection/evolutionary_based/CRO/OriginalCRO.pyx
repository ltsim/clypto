#!/usr/bin/env python
# Created by "Thieu" at 14:56, 19/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalCRO(cy.Optimizer):
    """
    The original version of: Coral Reefs Optimization (CRO)

    Links:
        1. https://downloads.hindawi.com/journals/tswj/2014/739768.pdf

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + po (float): [0.2, 0.5], the rate between free/occupied at the beginning
        + Fb (float): [0.6, 0.9], BroadcastSpawner/ExistingCorals rate
        + Fa (float): [0.05, 0.3], fraction of corals duplicates its self and tries to settle in a different part of the reef
        + Fd (float): [0.05, 0.5], fraction of the worse health corals in reef will be applied depredation
        + Pd (float): [0.1, 0.7], Probability of depredation
        + GCR (float): [0.05, 0.2], probability for mutation process
        + gamma_min (float): [0.01, 0.1] factor for mutation process
        + gamma_max (float): [0.1, 0.5] factor for mutation process
        + n_trials (int): [2, 10], number of attempts for a larvar to set in the reef.

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
    >>> model = CRO.OriginalCRO(epoch=1000, pop_size=50, po = 0.4, Fb = 0.9, Fa = 0.1, Fd = 0.1, Pd = 0.5, GCR = 0.1, gamma_min = 0.02, gamma_max = 0.2, n_trials = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Salcedo-Sanz, S., Del Ser, J., Landa-Torres, I., Gil-López, S. and Portilla-Figueras, J.A., 2014.
    The coral reefs optimization algorithm: a novel metaheuristic for efficiently solving optimization problems. The Scientific World Journal, 2014.
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
            Pd (float): the maximum of probability of depredation
            GCR (float): probability for mutation process
            gamma_min (float): factor for mutation process
            gamma_max (float): factor for mutation process
            n_trials (int): number of attempts for a larva to set in the reef.
        """
        super().__init__(parameters=[ "epoch", "pop_size", "po", "Fb", "Fa", "Fd", "Pd", "GCR", "gamma_min", "gamma_max", "n_trials", ], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)  # ~ number of space
        self.po = cy.validator(float, po, (0, 1.0), "po")
        self.Fb = cy.validator(float, Fb, (0, 1.0), "Fb")
        self.Fa = cy.validator(float, Fa, (0, 1.0), "Fa")
        self.Fd = cy.validator(float, Fd, (0, 1.0), "Fd")
        self.Pd = cy.validator(float, Pd, (0, 1.0), "Pd")
        self.GCR = cy.validator(float, GCR, (0, 1.0), "GCR")
        self.gamma_min = cy.validator(float, gamma_min, (0, 0.15), "gamma_min")
        self.gamma_max = cy.validator(float, gamma_max, (0.15, 1.0), "gamma_max")
        self.n_trials = cy.validator(int, n_trials, [2, int(self.population.size() / 2)], "n_trials")

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        self.reef = np.array([])
        self.occupied_position = (
            []
        )  # after a gen, you should update the occupied_position
        self.G1 = self.gamma_max
        self.alpha = 10 * self.Pd / self.epoch
        self.gama = 10 * (self.gamma_max - self.gamma_min) / self.epoch
        self.num_occupied = int(pop_size / (1 + self.po))
        self.dyn_Pd = 0
        self.occupied_list = np.zeros(pop_size)
        self.occupied_idx_list = self.generator.choice(
            list(range(pop_size)), self.num_occupied, replace=False
        )
        self.occupied_list[self.occupied_idx_list] = 1

    def gaussian_mutation__(self, position):
        random_pos = position + self.G1 * (
                self.problem.bounds.up - self.problem.bounds.low
        ) * self.generator.normal(0, 1, self.problem.n_dims)
        condition = self.generator.random(self.problem.n_dims) < self.GCR
        x = np.where(condition, random_pos, position)
        return cy.correct_solution(self.problem, x)

    ### Crossover
    def multi_point_cross__(self, pos1, pos2):
        p1, p2 = self.generator.choice(list(range(len(pos1))), 2, replace=False)
        start, end = min(p1, p2), max(p1, p2)
        x = np.concatenate((pos1[:start], pos2[start:end], pos1[end:]), axis=0)
        return cy.correct_solution(self.problem, x)

    def larvae_setting__(self, larvae):
        pop_size = self.population.size()
        # Trial to land on a square of reefs
        for larva in larvae:
            for idx in range(self.n_trials):
                pdx = self.generator.integers(0, pop_size - 1)
                if self.occupied_list[pdx] == 0:
                    self.population[pdx] = larva
                    self.occupied_idx_list = np.append(
                        self.occupied_idx_list, pdx
                    )  # Update occupied id
                    self.occupied_list[pdx] = 1  # Update occupied list
                    break
                else:
                    if cy.is_better(larva, self.population[pdx], self.problem.sense):
                        self.population[pdx] = larva
                        break

    def sort_occupied_reef__(self):
        def reef_fitness(idx):
            return self.population[idx].fitness

        return sorted(self.occupied_idx_list, key=reef_fitness)

    def broadcast_spawning_brooding__(self):
        # Step 1a
        larvae = []
        selected_corals = self.generator.choice(
            self.occupied_idx_list,
            int(len(self.occupied_idx_list) * self.Fb),
            replace=False,
        )
        for idx in self.occupied_idx_list:
            if idx not in selected_corals:
                x = self.gaussian_mutation__(self.population[idx].solution)
                agent = self.population.create_agent(x)
                larvae.append(agent)
                if self.mode == "sequential":
                    larvae[-1].evaluate(self.problem)
        # Step 1b
        while len(selected_corals) >= 2:
            id1, id2 = self.generator.choice(
                range(len(selected_corals)), 2, replace=False
            )
            x = self.multi_point_cross__(
                self.population[selected_corals[id1]].solution,
                self.population[selected_corals[id2]].solution,
            )
            agent = self.population.create_agent(x)
            larvae.append(agent)
            if self.mode == "sequential":
                larvae[-1].evaluate(self.problem)
            selected_corals = np.delete(selected_corals, [id1, id2])
        return self.population.evaluate(larvae, self.mode)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Broadcast Spawning Brooding
        larvae = self.broadcast_spawning_brooding__()
        self.larvae_setting__(larvae)
        ## Asexual Reproduction
        num_duplicate = int(len(self.occupied_idx_list) * self.Fa)
        pop_best = [self.population[idx] for idx in self.occupied_idx_list]
        pop_best = cy.sort_agents(pop_best, self.problem.sense)[:num_duplicate]
        self.larvae_setting__(pop_best)
        ## Depredation
        if self.generator.random() < self.dyn_Pd:
            num__depredation__ = int(len(self.occupied_idx_list) * self.Fd)
            idx_list_sorted = self.sort_occupied_reef__()
            selected_depredator = idx_list_sorted[-num__depredation__:]
            self.occupied_idx_list = np.setdiff1d(
                self.occupied_idx_list, selected_depredator
            )
            for idx in selected_depredator:
                self.occupied_list[idx] = 0
        if self.dyn_Pd <= self.Pd:
            self.dyn_Pd += self.alpha
        if self.G1 >= self.gamma_min:
            self.G1 -= self.gama

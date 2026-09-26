#!/usr/bin/env python
# Created by "Thieu" at 14:56, 19/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent cimport LegacyNativeAgent
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalCRO(LegacyNativeOptimizer):
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
    >>> from clypto.collection.evolutionary_based import CRO    >>> import numpy as np
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
    >>> model = CRO.OriginalCRO(epoch=1000, pop_size=50, po = 0.4, Fb = 0.9, Fa = 0.1, Fd = 0.1, Pd = 0.5, GCR = 0.1, gamma_min = 0.02, gamma_max = 0.2, n_trials = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

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
        *,
        name: str | None = None,
        mode: str | None = None,
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
        LegacyNativeOptimizer.__init__(
            self,
            parameters=[
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
            ],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.po = cy.validator(float, po, (0, 1.0), "po")
        self.Fb = cy.validator(float, Fb, (0, 1.0), "Fb")
        self.Fa = cy.validator(float, Fa, (0, 1.0), "Fa")
        self.Fd = cy.validator(float, Fd, (0, 1.0), "Fd")
        self.Pd = cy.validator(float, Pd, (0, 1.0), "Pd")
        self.GCR = cy.validator(float, GCR, (0, 1.0), "GCR")
        self.gamma_min = cy.validator(float, gamma_min, (0, 0.15), "gamma_min")
        self.gamma_max = cy.validator(float, gamma_max, (0.15, 1.0), "gamma_max")
        self.n_trials = cy.validator(int, n_trials, [2, int(self.pop_size / 2)], "n_trials")

    cdef void initialization(self):
        LegacyNativeOptimizer.initialization(self)
        self.reef = np.array([])
        self.occupied_position = (
            []
        )  # after a gen, you should update the occupied_position
        self.G1 = self.gamma_max
        self.alpha = 10 * self.Pd / self.epoch
        self.gama = 10 * (self.gamma_max - self.gamma_min) / self.epoch
        self.num_occupied = int(self.pop_size / (1 + self.po))
        self.dyn_Pd = 0
        self.occupied_list = np.zeros(self.pop_size)
        self.occupied_idx_list = self.generator.choice(
            list(range(self.pop_size)), self.num_occupied, replace=False
        )
        self.occupied_list[self.occupied_idx_list] = 1

    def gaussian_mutation__(self, position):
        random_pos = position + self.G1 * (
                self.problem.ub - self.problem.lb
        ) * self.generator.normal(0, 1, self.problem.n_dims)
        condition = self.generator.random(self.problem.n_dims) < self.GCR
        pos_new = np.where(condition, random_pos, position)
        return self.correct_solution(pos_new)

    def multi_point_cross__(self, pos1, pos2):
        p1, p2 = self.generator.choice(list(range(len(pos1))), 2, replace=False)
        start, end = min(p1, p2), max(p1, p2)
        pos_new = np.concatenate((pos1[:start], pos2[start:end], pos1[end:]), axis=0)
        return self.correct_solution(pos_new)

    def larvae_setting__(self, larvae):
        # Trial to land on a square of reefs
        for larva in larvae:
            for idx in range(self.n_trials):
                pdx = self.generator.integers(0, self.pop_size - 1)
                if self.occupied_list[pdx] == 0:
                    self.objs[pdx] = larva
                    self.occupied_idx_list = np.append(
                        self.occupied_idx_list, pdx
                    )  # Update occupied id
                    self.occupied_list[pdx] = 1  # Update occupied list
                    break
                else:
                    if self.compare_fitness(larva.target.fitness, self.objs[pdx].target.fitness, self.problem.minmax):
                        self.objs[pdx] = larva
                        break

    def sort_occupied_reef__(self):
        def reef_fitness(idx):
            return self.objs[idx].target.fitness

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
                pos_new = self.gaussian_mutation__(self.objs[idx].solution)
                agent = LegacyNativeAgent(pos_new, None)
                larvae.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    larvae[-1].target = self.get_target(pos_new)
        # Step 1b
        while len(selected_corals) >= 2:
            id1, id2 = self.generator.choice(
                range(len(selected_corals)), 2, replace=False
            )
            pos_new = self.multi_point_cross__(
                self.objs[selected_corals[id1]].solution,
                self.objs[selected_corals[id2]].solution,
            )
            agent = LegacyNativeAgent(pos_new, None)
            larvae.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                larvae[-1].target = self.get_target(pos_new)
            selected_corals = np.delete(selected_corals, [id1, id2])
        return ops.update_targets(self, larvae)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        self.objs = ops.agents_of(self.pop)
        ## Broadcast Spawning Brooding
        larvae = self.broadcast_spawning_brooding__()
        self.larvae_setting__(larvae)
        ## Asexual Reproduction
        num_duplicate = int(len(self.occupied_idx_list) * self.Fa)
        pop_best = [self.objs[idx] for idx in self.occupied_idx_list]
        pop_best = ops.sorted_agents(self, pop_best)[:num_duplicate]
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
        self.pop = ops.population_of(self.pop, self.objs)

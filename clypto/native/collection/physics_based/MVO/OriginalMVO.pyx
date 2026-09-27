#!/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.physics_based.MVO.DevMVO cimport DevMVO


cdef class OriginalMVO(DevMVO):
    """
    The original version of: Multi-Verse Optimizer (MVO)

    Links:
        1. https://dx.doi.org/10.1007/s00521-015-1870-7
        2. https://www.mathworks.com/matlabcentral/fileexchange/50112-multi-verse-optimizer-mvo

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wep_min (float): [0.05, 0.3], Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
        + wep_max (float: [0.75, 1.0], Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.physics_based import MVO    >>> import numpy as np
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
    >>> model = MVO.OriginalMVO(epoch=1000, pop_size=50, wep_min = 0.2, wep_max = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., Mirjalili, S.M. and Hatamlou, A., 2016. Multi-verse optimizer: a nature-inspired
    algorithm for global optimization. Neural Computing and Applications, 27(2), pp.495-513.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            wep_min: float = 0.2,
            wep_max: float = 1.0,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wep_min (float): Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
            wep_max (float: Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0
        """
        super().__init__(epoch, pop_size, wep_min, wep_max, **kwargs)

    # sorted_inflation_rates
    def roulette_wheel_selection__(self, weights=None):
        accumulation = np.cumsum(weights)
        p = self.generator.uniform() * accumulation[-1]
        chosen_idx = None
        for idx in range(len(accumulation)):
            if accumulation[idx] > p:
                chosen_idx = idx
                break
        return chosen_idx

    def normalize__(self, d, to_sum=True):
        pop_size = self.population.size()
        # d is a (n x dimension) np np.array
        d -= np.min(d, axis=0)
        if to_sum:
            total_vector = np.sum(d, axis=0)
            if 0 in total_vector:
                return self.generator.uniform(0.2, 0.8, pop_size)
            return d / np.sum(d, axis=0)
        else:
            ptp_vector = np.ptp(d, axis=0)
            if 0 in ptp_vector:
                return self.generator.uniform(0.2, 0.8, pop_size)
            return d / np.ptp(d, axis=0)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Eq. (3.3) in the paper
        wep = self.wep_min + epoch * ((self.wep_max - self.wep_min) / self.epoch)
        # Travelling Distance Rate (Formula): Eq. (3.4) in the paper
        tdr = 1 - epoch ** (1.0 / 6) / self.epoch ** (1.0 / 6)
        list_fitness_raw = np.array([item.fitness for item in self.population])
        maxx = max(list_fitness_raw)
        if maxx > (2 ** 64 - 1):
            list_fitness_normalized = self.generator.uniform(0, 0.1, pop_size)
        else:
            ### Normalize inflation rates (NI in Eq. (3.1) in the paper)
            list_fitness_normalized = np.reshape(
                self.normalize__(np.array([list_fitness_raw])), pop_size
            )  # Matrix
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            black_hole_pos = agent.solution.copy()
            for jdx in range(0, self.problem.n_dims):
                r1 = self.generator.uniform()
                if r1 < list_fitness_normalized[idx]:
                    white_hole_id = self.roulette_wheel_selection__(
                        (-1.0 * list_fitness_raw)
                    )
                    if white_hole_id == None or white_hole_id == -1:
                        white_hole_id = 0
                    # Eq. (3.1) in the paper
                    black_hole_pos[jdx] = self.population[white_hole_id].solution[jdx]
                # Eq. (3.2) in the paper if the boundaries are all the same
                r2 = self.generator.uniform()
                if r2 < wep:
                    r3 = self.generator.uniform()
                    if r3 < 0.5:
                        black_hole_pos[jdx] = self.g_best.solution[
                                                  jdx
                                              ] + tdr * self.generator.uniform(
                            self.problem.bounds.low[jdx], self.problem.bounds.up[jdx]
                        )
                    else:
                        black_hole_pos[jdx] = self.g_best.solution[
                                                  jdx
                                              ] - tdr * self.generator.uniform(
                            self.problem.bounds.low[jdx], self.problem.bounds.up[jdx]
                        )
            x = cy.correct_solution(self.problem, black_hole_pos)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

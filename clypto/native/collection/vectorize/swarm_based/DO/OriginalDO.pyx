#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 04:43, 02/03/2021 ----------%
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


cdef class OriginalDO(AgentListOptimizer):
    """
    The original version of: Dragonfly Optimization (DO)

    Links:
        1. https://link.springer.com/article/10.1007/s00521-015-1920-1

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import DO    >>> import numpy as np
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
    >>> model = DO.OriginalDO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2016. Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective,
    discrete, and multi-objective problems. Neural computing and applications, 27(4), pp.1053-1073.
    """

    cdef public object pop_delta
    cdef public object radius
    cdef public object delta_max

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

    cdef void initialization(self):
        AgentListOptimizer.initialization(self)
        self.pop_delta = self.generate_agents(self.pop_size)
        # Initial radius of dragonflies' neighborhoods
        self.radius = (self.problem.ub - self.problem.lb) / 10
        self.delta_max = (self.problem.ub - self.problem.lb) / 10
        self.pop = self.mirror__()

    def evolve_agents(self, epoch):
        _, (self.g_best,), (self.g_worst,) = self.get_special_agents(
            self.objs, n_best=1, n_worst=1, minmax=self.problem.minmax
        )

        r = (self.problem.ub - self.problem.lb) / 4 + (
                (self.problem.ub - self.problem.lb) * (2 * epoch / self.epoch)
        )
        w = 0.9 - epoch * ((0.9 - 0.4) / self.epoch)
        my_c = 0.1 - epoch * ((0.1 - 0) / (self.epoch / 2))
        my_c = 0 if my_c < 0 else my_c

        s = 2 * self.generator.random() * my_c  # Seperation weight
        a = 2 * self.generator.random() * my_c  # Alignment weight
        c = 2 * self.generator.random() * my_c  # Cohesion weight
        f = 2 * self.generator.random()  # Food attraction weight
        e = my_c  # Enemy distraction weight

        pop_new = []
        pop_delta_new = []
        for idx in range(0, self.pop_size):
            pos_neighbours = []
            pos_neighbours_delta = []
            neighbours_num = 0
            # Find the neighbouring solutions
            for j in range(0, self.pop_size):
                dist = np.abs(self.objs[idx].solution - self.objs[j].solution)
                if np.all(dist <= r) and np.all(dist != 0):
                    neighbours_num += 1
                    pos_neighbours.append(self.objs[j].solution)
                    pos_neighbours_delta.append(self.pop_delta[j].solution)
            pos_neighbours = np.array(pos_neighbours)
            pos_neighbours_delta = np.array(pos_neighbours_delta)

            # Separation: Eq 3.1, Alignment: Eq 3.2, Cohesion: Eq 3.3
            if neighbours_num > 1:
                S = (
                        np.sum(pos_neighbours, axis=0)
                        - neighbours_num * self.objs[idx].solution
                )
                A = np.sum(pos_neighbours_delta, axis=0) / neighbours_num
                C_temp = np.sum(pos_neighbours, axis=0) / neighbours_num
            else:
                S = np.zeros(self.problem.n_dims)
                A = self.pop_delta[idx].solution.copy()
                C_temp = self.objs[idx].solution.copy()
            C = C_temp - self.objs[idx].solution

            # Attraction to food: Eq 3.4
            dist_to_food = np.abs(self.objs[idx].solution - self.g_best.solution)
            if np.all(dist_to_food <= r):
                F = self.g_best.solution - self.objs[idx].solution
            else:
                F = np.zeros(self.problem.n_dims)

            # Distraction from enemy: Eq 3.5
            dist_to_enemy = np.abs(self.objs[idx].solution - self.g_worst.solution)
            if np.all(dist_to_enemy <= r):
                enemy = self.g_worst.solution + self.objs[idx].solution
            else:
                enemy = np.zeros(self.problem.n_dims)

            pos_new = self.objs[idx].solution.copy().astype(float)
            pos_delta_new = self.pop_delta[idx].solution.copy().astype(float)
            if np.any(dist_to_food > r):
                if neighbours_num > 1:
                    temp = (
                            w * self.pop_delta[idx].solution
                            + self.generator.uniform(0, 1, self.problem.n_dims) * A
                            + self.generator.uniform(0, 1, self.problem.n_dims) * C
                            + self.generator.uniform(0, 1, self.problem.n_dims) * S
                    )
                    temp = np.clip(temp, -1 * self.delta_max, self.delta_max)
                    pos_delta_new = temp.copy()
                    pos_new += temp
                else:  # Eq. 3.8
                    pos_new += (
                            self.get_levy_flight_step(beta=1.5, multiplier=0.01, case=-1)
                            * self.objs[idx].solution
                    )
                    pos_delta_new = np.zeros(self.problem.n_dims)
            else:
                # Eq. 3.6
                temp = (a * A + c * C + s * S + f * F + e * enemy) + w * self.pop_delta[
                    idx
                ].solution
                temp = np.clip(temp, -1 * self.delta_max, self.delta_max)
                pos_delta_new = temp
                pos_new += temp

            # Amend solution
            pos_new = self.correct_solution(pos_new)
            pos_delta_new = self.correct_solution(pos_delta_new)
            agent = self.generate_empty_agent(pos_new)
            agent_delta = self.generate_empty_agent(pos_delta_new)
            pop_new.append(agent)
            pop_delta_new.append(agent_delta)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                agent_delta.target = self.get_target(pos_delta_new)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
                self.pop_delta[idx] = self.get_better_agent(
                    agent_delta, self.pop_delta[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            pop_delta_new = self.update_target_for_population(pop_delta_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )
            self.pop_delta = self.greedy_selection_population(
                self.pop_delta, pop_delta_new, self.problem.minmax
            )

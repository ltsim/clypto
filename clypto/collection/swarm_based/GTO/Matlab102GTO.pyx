#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 21:58, 16/03/2023 ----------%
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


cdef class Matlab102GTO(AgentListOptimizer):
    """
    The conversion of Matlab code (version 1.0.2 - 27/04/2023) to Python code of: Giant Trevally Optimizer (GTO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/121358-giant-trevally-optimizer-gto
        2. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9955508

    Notes:
        1. The authors sent me an email asking to update the algorithm. In this version, they removed 2 for loops in the epoch (generations) based on my comments,
        so the computation time will reduce to 3*pop_size from 2*pop_size^2 + pop_size. However, this will also lead to a reduction in performance results.
        My question: Are the results in the paper valid?
        2. I have decided to implement the original version of the algorithm exactly as described in the paper (OriginalGTO).

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GTO    >>> import numpy as np
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
    >>> model = GTO.Matlab102GTO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Sadeeq, H. T., & Abdulazeez, A. M. (2022). Giant Trevally Optimizer (GTO): A Novel Metaheuristic
    Algorithm for Global Optimization and Challenging Engineering Problems. IEEE Access, 10, 121615-121640.
    """

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
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def evolve_agents(self, epoch):
        # Step 1: Extensive Search
        pop_new = []
        for idx in range(0, self.pop_size):
            # foraging movement patterns of giant trevallies are simulated using Eq.(4)
            pos_new = self.g_best.solution * self.generator.random() + (
                    (self.problem.ub - self.problem.lb) * self.generator.random()
                    + self.problem.lb
            ) * self.get_levy_flight_step(
                beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1
            )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )
        self.objs, self.g_best = self.update_global_best_agent(self.objs, save=False)

        # Step 2: Choosing Area
        A = 0.4
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_m = np.mean(np.array([agent.solution for agent in self.objs]), axis=0)
            # In the choosing area step, giant trevallies identify and select the best area in terms of
            # the amount of food (seabirds) within the selected search space where they can hunt for prey.
            r3 = self.generator.random()
            pos_new = (
                    self.g_best.solution * A * r3 + pos_m - self.objs[idx].solution * r3
            )  # Eq. 7

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )
        self.objs, self.g_best = self.update_global_best_agent(self.objs, save=False)

        # Step 3: Attacking
        H = self.generator.random() * 2.0 * (1.0 - epoch / self.epoch)  # Eq.(15)
        pop_new = []
        for idx in range(0, self.pop_size):
            # the distance between the prey and the attacker, and can be calculated using (12):
            dist = np.sum(np.abs(self.g_best.solution - self.objs[idx].solution))
            theta2 = (360 - 0) * self.generator.random() + 0
            theta1 = 1.3296 * np.sin(np.radians(theta2))  # calculate theta_1 using (10)
            # visual distortion indicates the apparent height of the bird, which is always seen
            # to be higher than its actual height due to the refraction of the light.
            VD = np.sin(np.radians(theta1)) * dist  # Eq. 11
            # the behavior of giant trevally when chasing and jumping out of the water is mathematically simulated using (13)
            pos_new = (
                    self.objs[idx].solution
                    * np.sin(np.radians(theta2))
                    * self.objs[idx].target.fitness
                    + VD
                    + H
            )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )

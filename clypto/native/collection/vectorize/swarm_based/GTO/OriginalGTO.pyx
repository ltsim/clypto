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


cdef class OriginalGTO(AgentListOptimizer):
    """
    The original version of: Giant Trevally Optimizer (GTO)

    Notes:
        1. This version is implemented exactly as described in the paper.
        2. https://www.mathworks.com/matlabcentral/fileexchange/121358-giant-trevally-optimizer-gto
        3. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9955508

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + A (float): a position-change-controlling parameter with a range from 0.3 to 0.4, default=0.4
        + H (float): initial value for specifies the jumping slope function, default=2.0

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
    >>> model = GTO.OriginalGTO(epoch=1000, pop_size=50, A=0.4, H=2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Sadeeq, H. T., & Abdulazeez, A. M. (2022). Giant Trevally Optimizer (GTO): A Novel Metaheuristic
    Algorithm for Global Optimization and Challenging Engineering Problems. IEEE Access, 10, 121615-121640.
    """

    cdef public object A
    cdef public object H

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        A: float = 0.4,
        H: float = 2.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            A (float): a position-change-controlling parameter with a range from 0.3 to 0.4, default=0.4
            H (float): initial value for specifies the jumping slope function, default=2.0
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "A", "H"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.A = cy.validator(float, A, [-10.0, 10.0], "A")
        self.H = cy.validator(float, H, [1.0, 10.0], "H")

    def evolve_agents(self, epoch):
        # Step 1: Extensive Search
        pop_new = []
        for idx in range(0, self.pop_size):
            # Eq.(4)
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
        pos_list = np.array([agent.solution for agent in self.objs])
        pos_m = np.mean(pos_list, axis=0)
        pop_new = []
        for idx in range(0, self.pop_size):
            r3 = self.generator.random()
            pos_new = (
                    self.g_best.solution * self.A * r3 + pos_m - self.objs[idx].solution * r3
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
        _, self.g_best = self.update_global_best_agent(self.objs, save=False)

        # Step 3: Attacking
        H = self.generator.random() * self.H * (1 - epoch / self.epoch)  # Eq.(15)
        pop_new = []
        for idx in range(0, self.pop_size):
            # the distance between the prey and the attacker, and can be calculated using (12):
            dist = np.sum(np.abs(self.g_best.solution - self.objs[idx].solution))
            theta2 = (360 - 0) * self.generator.random() + 0
            theta1 = (1.33 / 1.00029) * np.sin(
                np.radians(theta2)
            )  # calculate theta_1 using (10)
            VD = np.sin(np.radians(theta1)) * dist  # Eq. 11
            # Eq. (13)
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

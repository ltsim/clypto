#!/usr/bin/env python
# Created by "Thieu" at 15:05, 03/06/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.native.collection.legacy.swarm_based.SLO.ModifiedSLO cimport ModifiedSLO


cdef class ImprovedSLO(ModifiedSLO):
    """
    The original version: Improved Sea Lion Optimization (ImprovedSLO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): Local coefficient same as PSO, default = 1.2
        + c2 (float): Global coefficient same as PSO, default = 1.2

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import SLO    >>> import numpy as np
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
    >>> model = SLO.ImprovedSLO(epoch=1000, pop_size=50, c1=1.2, c2=1.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, Binh Minh, Trung Tran, Thieu Nguyen, and Giang Nguyen. "An improved sea lion optimization for workload elasticity
    prediction with neural networks." International Journal of Computational Intelligence Systems 15, no. 1 (2022): 90.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 1.2,
        c2: float = 1.2,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): Local coefficient same as PSO, default = 1.2
            c2 (float): Global coefficient same as PSO, default = 1.2
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.set_parameters(["epoch", "pop_size", "c1", "c2"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        c = 2.0 - 2.0 * epoch / self.epoch
        t0 = self.generator.random()
        v1 = np.sin(2 * np.pi * t0)
        v2 = np.sin(2 * np.pi * (1 - t0))
        SP_leader = np.abs(v1 * (1 + v2) / v2)
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            if SP_leader < 0.5:
                if (
                    c < 1
                ):  # Exploitation improved by historical movement + global best affect
                    # pos_new = g_best.solution - c * np.abs(2 * rand() * g_best.solution - pop[i].solution)
                    dif1 = np.abs(
                        2 * self.generator.random() * self.g_best.solution
                        - self.pop[idx].solution
                    )
                    dif2 = np.abs(
                        2 * self.generator.random() * self.pop[idx].local_solution
                        - self.pop[idx].solution
                    )
                    pos_new = self.c1 * self.generator.random() * (
                        self.pop[idx].solution - dif1
                    ) + self.c2 * self.generator.random() * (
                        self.pop[idx].solution - dif2
                    )
                else:  # Exploration improved by opposition-based learning
                    # Create a new solution by equation below
                    # Then create an opposition solution of above solution
                    # Compare both of them and keep the good one (Searching at both direction)
                    pos_new = self.g_best.solution + c * self.generator.normal(
                        0, 1, self.problem.n_dims
                    ) * (self.g_best.solution - self.pop[idx].solution)
                    pos_new = self.correct_solution(pos_new)
                    target_new = self.get_target(pos_new)
                    pos_new_oppo = (
                        self.problem.lb
                        + self.problem.ub
                        - self.g_best.solution
                        + self.generator.random() * (self.g_best.solution - pos_new)
                    )
                    pos_new_oppo = self.correct_solution(pos_new_oppo)
                    target_new_oppo = self.get_target(pos_new_oppo)
                    if self.compare_target(
                        target_new_oppo, target_new, self.problem.minmax
                    ):
                        pos_new = pos_new_oppo
            else:  # Exploitation
                pos_new = self.g_best.solution + np.cos(
                    2 * np.pi * self.generator.uniform(-1, 1)
                ) * np.abs(self.g_best.solution - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent.solution = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_target(
                pop_new[idx].target, self.pop[idx].target, self.problem.minmax
            ):
                self.pop[idx] = pop_new[idx].copy()
                if self.compare_target(
                    pop_new[idx].target, self.pop[idx].local_target, self.problem.minmax
                ):
                    self.pop[idx].local_solution = pop_new[idx].solution.copy()
                    self.pop[idx].local_target = pop_new[idx].target.copy()

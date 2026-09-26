#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.native.collection.legacy.physics_based.TWO.OriginalTWO cimport OriginalTWO


cdef class LevyTWO(OriginalTWO):
    """
    The Levy-flight version of: Tug of War Optimization (LevyTWO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.physics_based import TWO    >>> import numpy as np
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
    >>> model = TWO.LevyTWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = self.pop.copy()
        for idx in range(self.pop_size):
            pos_new = self.pop[idx].solution.copy().astype(float)
            for kdx in range(self.pop_size):
                if self.pop[idx].weight < self.pop[kdx].weight:
                    force = max(
                        self.pop[idx].weight * self.muy_s,
                        self.pop[kdx].weight * self.muy_s,
                    )
                    resultant_force = force - self.pop[idx].weight * self.muy_k
                    g = self.pop[kdx].solution - self.pop[idx].solution
                    acceleration = (
                        resultant_force * g / (self.pop[idx].weight * self.muy_k)
                    )
                    delta_x = 1 / 2 * acceleration + np.power(
                        self.alpha, epoch
                    ) * self.beta * (
                        self.problem.ub - self.problem.lb
                    ) * self.generator.normal(
                        0, 1, self.problem.n_dims
                    )
                    pos_new += delta_x
            pop_new[idx].solution = pos_new
        for idx in range(self.pop_size):
            pos_new = self.pop[idx].solution.copy().astype(float)
            for jdx in range(self.problem.n_dims):
                if (
                    pos_new[jdx] < self.problem.lb[jdx]
                    or pos_new[jdx] > self.problem.ub[jdx]
                ):
                    if self.generator.random() <= 0.5:
                        pos_new[jdx] = self.g_best.solution[
                            jdx
                        ] + self.generator.standard_normal() / epoch * (
                            self.g_best.solution[jdx] - pos_new[jdx]
                        )
                        if (
                            pos_new[jdx] < self.problem.lb[jdx]
                            or pos_new[jdx] > self.problem.ub[jdx]
                        ):
                            pos_new[jdx] = self.pop[idx].solution[jdx]
                    else:
                        if pos_new[jdx] < self.problem.lb[jdx]:
                            pos_new[jdx] = self.problem.lb[jdx]
                        if pos_new[jdx] > self.problem.ub[jdx]:
                            pos_new[jdx] = self.problem.ub[jdx]
            pop_new[idx].solution = self.correct_solution(pos_new)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx].target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    pop_new[idx], self.pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )
        ### Apply levy-flight here
        for idx in range(self.pop_size):
            ## Chance for each agent to update using levy is 50%
            if self.generator.random() < 0.5:
                levy_step = self.get_levy_flight_step(
                    beta=1.0, multiplier=0.01, size=self.problem.n_dims, case=-1
                )
                pos_new = pop_new[idx].solution + levy_step
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(
                    agent.target, pop_new[idx].target, self.problem.minmax
                ):
                    pop_new[idx] = agent
        self.pop = self.update_weight__(pop_new)

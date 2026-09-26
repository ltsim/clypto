#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from clypto.native.collection.legacy.physics_based.TWO.OppoTWO import OppoTWO
from clypto.native.collection.legacy.physics_based.TWO.LevyTWO import LevyTWO


class EnhancedTWO(OppoTWO, LevyTWO):
    """
    The original version of: Enhenced Tug of War Optimization (ETWO)

    Links:
        1. https://doi.org/10.1016/j.procs.2020.03.063

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.physics_based import TWO    >>> import numpy as np
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
    >>> model = TWO.EnhancedTWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Hoang, B., Nguyen, G. and Nguyen, B.M., 2020. A new workload prediction model using
    extreme learning machine and enhanced tug of war optimization. Procedia Computer Science, 170, pp.362-369.
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

    def _initialization(self):
        if self.pop is None:
            self.pop = self._generate_population(self.pop_size)
        pop_oppo = self.pop.copy()
        for idx in range(self.pop_size):
            pos_opposite = self.problem.bounds.up + self.problem.bounds.low - self.pop[idx].solution
            pos_new = self._correct_solution(pos_opposite)
            pop_oppo[idx].solution = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                pop_oppo[idx].target = self._get_target(pos_new)
        pop_oppo = self._update_target_for_population(pop_oppo)
        self.pop = self._get_sorted_and_trimmed_population(
            self.pop + pop_oppo, self.pop_size, self.problem.sense
        )
        self.pop = self.update_weight__(self.pop)

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

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
                        self.problem.bounds.up - self.problem.bounds.low
                    ) * self.generator.normal(
                        0, 1, self.problem.n_dims
                    )
                    pos_new += delta_x
            pop_new[idx].solution = pos_new
        for idx in range(self.pop_size):
            pos_new = self.pop[idx].solution.copy().astype(float)
            for jdx in range(self.problem.n_dims):
                if (
                    pos_new[jdx] < self.problem.bounds.low[jdx]
                    or pos_new[jdx] > self.problem.bounds.up[jdx]
                ):
                    if self.generator.random() <= 0.5:
                        pos_new[jdx] = self.g_best.solution[
                            jdx
                        ] + self.generator.standard_normal() / epoch * (
                            self.g_best.solution[jdx] - pos_new[jdx]
                        )
                        if (
                            pos_new[jdx] < self.problem.bounds.low[jdx]
                            or pos_new[jdx] > self.problem.bounds.up[jdx]
                        ):
                            pos_new[jdx] = self.pop[idx].solution[jdx]
                    else:
                        if pos_new[jdx] < self.problem.bounds.low[jdx]:
                            pos_new[jdx] = self.problem.bounds.low[jdx]
                        if pos_new[jdx] > self.problem.bounds.up[jdx]:
                            pos_new[jdx] = self.problem.bounds.up[jdx]
            pop_new[idx].solution = self._correct_solution(pos_new)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx].target = self._get_target(pos_new)
                self.pop[idx] = self._get_better_agent(
                    pop_new[idx], self.pop[idx], self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            self.pop = self._greedy_selection_population(
                self.pop, pop_new, self.problem.sense
            )

        for idx in range(self.pop_size):
            C_op = self._generate_opposition_solution(pop_new[idx], self.g_best)
            pos_new = self._correct_solution(C_op)
            agent = self._generate_agent(pos_new)
            if self._compare_target(
                agent.target, pop_new[idx].target, self.problem.sense
            ):
                pop_new[idx] = agent
            else:
                levy_step = self._get_levy_flight_step(
                    beta=1.0, multiplier=1.0, size=self.problem.n_dims, case=-1
                )
                pos_new = pop_new[idx].solution + 1.0 / np.sqrt(epoch) * levy_step
                pos_new = self._correct_solution(pos_new)
                agent = self._generate_agent(pos_new)
                if self._compare_target(
                    agent.target, pop_new[idx].target, self.problem.sense
                ):
                    pop_new[idx] = agent
        self.pop = self.update_weight__(pop_new)

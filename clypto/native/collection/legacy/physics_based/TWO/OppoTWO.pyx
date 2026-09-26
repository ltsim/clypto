#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.native.collection.legacy.physics_based.TWO.OriginalTWO cimport OriginalTWO


cdef class OppoTWO(OriginalTWO):
    """
    The opossition-based learning version: Tug of War Optimization (OTWO)

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
    >>> model = TWO.OppoTWO(epoch=1000, pop_size=50)
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

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        half_size = -(-self.pop_size // 2)  # ceil division, safe for odd pop_size
        list_idx = self.generator.choice(
            range(0, self.pop_size), half_size, replace=False
        )
        pop_temp = [self.pop[list_idx[idx]] for idx in range(0, half_size)]
        pop_oppo = []
        for idx in range(len(pop_temp)):
            pos_opposite = self.problem.ub + self.problem.lb - pop_temp[idx].solution
            pos_opposite = self.correct_solution(pos_opposite)
            agent = self.generate_empty_agent(pos_opposite)
            pop_oppo.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_oppo[-1].target = self.get_target(pos_opposite)
        pop_oppo = self.update_target_for_population(pop_oppo)
        self.pop = (pop_temp + pop_oppo)[: self.pop_size]
        self.pop = self.update_weight__(self.pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Apply force of others solution on each individual solution
        pop_new = self.pop.copy()
        for idx in range(self.pop_size):
            pos_new = pop_new[idx].solution.copy().astype(float)
            for jdx in range(self.pop_size):
                if self.pop[idx].weight < self.pop[jdx].weight:
                    force = max(
                        self.pop[idx].weight * self.muy_s,
                        self.pop[jdx].weight * self.muy_s,
                    )
                    resultant_force = force - self.pop[idx].weight * self.muy_k
                    g = self.pop[jdx].solution - self.pop[idx].solution
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
            self.pop[idx].solution = pos_new
        ## Amend solution and update fitness value
        for idx in range(self.pop_size):
            pos_new = self.g_best.solution + self.generator.normal(
                0, 1, self.problem.n_dims
            ) / (epoch) * (self.g_best.solution - pop_new[idx].solution)
            conditions = np.logical_or(
                pop_new[idx].solution < self.problem.lb,
                pop_new[idx].solution > self.problem.ub,
            )
            conditions = np.logical_and(
                conditions, self.generator.random(self.problem.n_dims) < 0.5
            )
            pos_new = np.where(conditions, pos_new, self.pop[idx].solution)
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
        ## Opposition-based here
        pop = []
        for idx in range(self.pop_size):
            C_op = self.generate_opposition_solution(self.pop[idx], self.g_best)
            pos_new = self.correct_solution(C_op)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(
                self.pop, pop, self.problem.minmax
            )
        self.pop = self.update_weight__(self.pop)

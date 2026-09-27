#!/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class HI_WOA(cy.Optimizer):
    """
    The original version of: Hybrid Improved Whale Optimization Algorithm (HI-WOA)

    Links:
        1. https://ieenp.explore.ieee.org/document/8900003

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + feedback_max (int): maximum iterations of each feedback, default = 10

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import WOA    >>> import numpy as np
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
    >>> model = WOA.HI_WOA(epoch=1000, pop_size=50, feedback_max = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Tang, C., Sun, W., Wu, W. and Xue, M., 2019, July. A hybrid improved whale optimization algorithm.
    In 2019 IEEE 15th International Conference on Control and Automation (ICCA) (pp. 362-367). IEEE.
    """

    cdef public int feedback_max

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            feedback_max: int = 10,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            feedback_max (int): maximum iterations of each feedback, default = 10
        """
        super().__init__(parameters=["epoch", "pop_size", "feedback_max"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.feedback_max = cy.validator(int, feedback_max, [2, 2 + int(self.epoch / 2)], "feedback_max")
        # The maximum of times g_best doesn't change -> need to change half of population

    def initialize_variables(self):
        pop_size = self.population.size()
        self.n_changes = int(pop_size / 2)
        self.dyn_feedback_count = 0

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        a = 2 + 2 * np.cos(np.pi / 2 * (1 + epoch / self.epoch))  # Eq. 8
        pop_new = []
        for idx in range(0, pop_size):
            r = self.generator.random()
            A = 2 * a * r - a
            C = 2 * r
            l = self.generator.uniform(-1, 1)
            p = 0.5
            b = 1
            if self.generator.uniform() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best.solution - self.population[idx].solution)
                    pos_new = self.g_best.solution - A * D
                else:
                    # x_rand = pop[self.generator.self.generator.randint(pop_size)]         # select random 1 position in pop
                    x_rand = self.problem.generate_solution()
                    D = np.abs(C * x_rand - self.population[idx].solution)
                    pos_new = x_rand - A * D
            else:
                D1 = np.abs(self.g_best.solution - self.population[idx].solution)
                pos_new = (
                        self.g_best.solution + np.exp(b * l) * np.cos(2 * np.pi * l) * D1
                )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

        ## Feedback Mechanism
        current_best = self.population.sort()[0].copy()
        if current_best.fitness == self.g_best.fitness:
            self.dyn_feedback_count += 1
        else:
            self.dyn_feedback_count = 0

        if self.dyn_feedback_count >= self.feedback_max:
            idx_list = self.generator.choice(
                range(0, pop_size), self.n_changes, replace=False
            )
            pop_child = self.population.generate(self.n_changes)
            for idx_counter, idx in enumerate(idx_list):
                self.population[idx] = pop_child[idx_counter]

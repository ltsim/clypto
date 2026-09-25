#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.legacy.physics_based.EO.OriginalEO cimport OriginalEO


cdef class ModifiedEO(OriginalEO):
    """
    The original version of: Modified Equilibrium Optimizer (MEO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2020.106542

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import EO    >>> import numpy as np
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
    >>> model = EO.ModifiedEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Gupta, S., Deep, K. and Mirjalili, S., 2020. An efficient equilibrium optimizer with mutation
    strategy for numerical optimization. Applied Soft Computing, 96, p.106542.
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
        self.sort_flag = False
        self.pop_len = int(self.pop_size / 3)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # ---------------- Memory saving-------------------  make equilibrium pool
        _, c_eq_list, _ = self.get_special_agents(
            self.pop, n_best=4, minmax=self.problem.minmax
        )
        c_pool = self.make_equilibrium_pool__(c_eq_list)
        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            lamda = self.generator.uniform(
                0, 1, self.problem.n_dims
            )  # lambda in Eq. 11
            r = self.generator.uniform(0, 1, self.problem.n_dims)  # r in Eq. 11
            c_eq = c_pool[
                self.generator.integers(0, len(c_pool))
            ].solution  # random selection 1 of candidate from the pool
            f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 11
            r1 = self.generator.uniform()
            r2 = self.generator.uniform()  # r1, r2 in Eq. 15
            gcp = 0.5 * r1 * np.ones(self.problem.n_dims) * (r2 >= self.GP)  # Eq. 15
            g0 = gcp * (c_eq - lamda * self.pop[idx].solution)  # Eq. 14
            g = g0 * f  # Eq. 13
            pos_new = (
                    c_eq
                    + (self.pop[idx].solution - c_eq) * f
                    + (g * self.V / lamda) * (1.0 - f)
            )  # Eq. 16
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )
        ## Sort the updated population based on fitness
        _, pop_s1, _ = self.get_special_agents(
            self.pop, n_best=self.pop_len, minmax=self.problem.minmax
        )
        ## Mutation scheme
        pop_s2 = pop_s1.copy()
        pop_s2_new = []
        for idx in range(0, self.pop_len):
            pos_new = pop_s2[idx].solution * (
                    1 + self.generator.normal(0, 1, self.problem.n_dims)
            )  # Eq. 12
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_s2_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop_s2[idx] = self.get_better_agent(
                    agent, pop_s2[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_s2_new = self.update_target_for_population(pop_s2_new)
            pop_s2 = self.greedy_selection_population(
                pop_s2_new, pop_s2, self.problem.minmax
            )

        ## Search Mechanism
        pos_s1_list = [agent.solution for agent in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        pop_s3 = []
        for idx in range(0, self.pop_len):
            pos_new = (c_pool[0].solution - pos_s1_mean) - self.generator.random() * (
                    self.problem.lb
                    + self.generator.random() * (self.problem.ub - self.problem.lb)
            )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_s3.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_s3[-1].target = self.get_target(pos_new)
        pop_s3 = self.update_target_for_population(pop_s3)
        ## Construct a new population
        self.pop = pop_s1 + pop_s2 + pop_s3
        n_left = self.pop_size - len(self.pop)
        idx_selected = self.generator.choice(
            range(0, len(c_pool)), n_left, replace=False
        )
        for idx in range(0, n_left):
            self.pop.append(c_pool[idx_selected[idx]])

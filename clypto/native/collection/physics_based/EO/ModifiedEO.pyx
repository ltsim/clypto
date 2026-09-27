#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.physics_based.EO.OriginalEO cimport OriginalEO


cdef class ModifiedEO(OriginalEO):
    """
    The original version of: Modified Equilibrium Optimizer (MEO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2020.106542

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.physics_based import EO    >>> import numpy as np
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
    >>> model = EO.ModifiedEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

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
        self.pop_len = int(self.population.size() / 3)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # ---------------- Memory saving-------------------  make equilibrium pool
        ranked = self.population.sort()
        c_eq_list = [cy.duplicate_agent(agent) for agent in ranked[:4]]
        c_pool = self.make_equilibrium_pool__(c_eq_list)
        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx in range(0, pop_size):
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
            g0 = gcp * (c_eq - lamda * self.population[idx].solution)  # Eq. 14
            g = g0 * f  # Eq. 13
            x = (
                    c_eq
                    + (self.population[idx].solution - c_eq) * f
                    + (g * self.V / lamda) * (1.0 - f)
            )  # Eq. 16
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            n_population.append(agent)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)
        ## Sort the updated population based on fitness
        ranked = self.population.sort()
        pop_s1 = [cy.duplicate_agent(agent) for agent in ranked[:self.pop_len]]
        ## Mutation scheme
        pop_s2 = pop_s1.copy()
        pop_s2_new = []
        for idx in range(0, self.pop_len):
            x = pop_s2[idx].solution * (
                    1 + self.generator.normal(0, 1, self.problem.n_dims)
            )  # Eq. 12
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop_s2_new.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                pop_s2[idx] = cy.get_better_agent(agent, pop_s2[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_s2_new = self.population.evaluate(pop_s2_new, self.mode)
            pop_s2 = cy.greedy_agents(pop_s2_new, pop_s2, self.problem.sense)

        ## Search Mechanism
        pos_s1_list = [agent.solution for agent in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        pop_s3 = []
        for idx in range(0, self.pop_len):
            x = (c_pool[0].solution - pos_s1_mean) - self.generator.random() * (
                    self.problem.bounds.low
                    + self.generator.random() * (self.problem.bounds.up - self.problem.bounds.low)
            )
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop_s3.append(agent)
        pop_s3 = self.population.evaluate(pop_s3, self.mode)
        ## Construct a new population
        self.population = self.population.spawn(pop_s1 + pop_s2 + pop_s3)
        n_left = pop_size - len(self.population)
        idx_selected = self.generator.choice(
            range(0, len(c_pool)), n_left, replace=False
        )
        for idx in range(0, n_left):
            self.population.append(c_pool[idx_selected[idx]])

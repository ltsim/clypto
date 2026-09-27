#!/usr/bin/env python
# Created by "Thieu" at 08:57, 14/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.human_based.FBIO.DevFBIO cimport DevFBIO


cdef class OriginalFBIO(DevFBIO):
    """
    The original version of: Forensic-Based Investigation Optimization (FBIO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2020.106339
        2. https://ww2.mathworks.cn/matlabcentral/fileexchange/76299-forensic-based-investigation-algorithm-fbi

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import FBIO    >>> import numpy as np
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
    >>> model = FBIO.OriginalFBIO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Chou, J.S. and Nguyen, N.M., 2020. FBI inspired meta-optimization. Applied Soft Computing, 93, p.106339.
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
        self.population = cy.population(pop_size, range=[5, 10000], cls=cy.Population)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Investigation team - team A
        # Step A1
        pop_new = []
        for idx in range(0, pop_size):
            n_change = self.generator.integers(0, self.problem.n_dims)
            nb1, nb2 = self.generator.choice(
                list(set(range(0, pop_size)) - {idx}), 2, replace=False
            )
            # Eq.(2) in FBI Inspired Meta - Optimization
            pos_a = self.population[idx].solution.copy()
            pos_a[n_change] = self.population[idx].solution[n_change] + (
                    self.generator.uniform() - 0.5
            ) * 2 * (
                                      self.population[idx].solution[n_change]
                                      - (self.population[nb1].solution[n_change] + self.population[nb2].solution[n_change])
                                      / 2
                              )
            ## Not good move here, change only 1 variable but check bound of all variable in solution
            pos_a = cy.reset_solution(self.problem, self.generator, pos_a)
            agent = self.population.create_agent(pos_a)
            pop_new.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

        # Step A2
        list_fitness = np.array([agent.fitness for agent in self.population])
        prob = self.probability__(list_fitness)
        pop_child = []
        for idx in range(0, pop_size):
            if self.generator.uniform() > prob[idx]:
                r1, r2, r3 = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx}), 3, replace=False
                )
                pos_a = self.population[idx].solution.copy()
                Rnd = np.floor(self.generator.uniform() * self.problem.n_dims) + 1
                for j in range(0, self.problem.n_dims):
                    if self.generator.uniform() < self.generator.uniform() or Rnd == j:
                        pos_a[j] = (
                                self.g_best.solution[j]
                                + self.population[r1].solution[j]
                                + self.generator.uniform()
                                * (self.population[r2].solution[j] - self.population[r3].solution[j])
                        )
                    ## In the original matlab code they do the else condition here, not good again because no need else here
                ## Same here, they do check the bound of all variable in solution
                ## pos_a = self.amend_position(pos_a, self.problem.bounds.low, self.problem.bounds.up)
            else:
                pos_a = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            pos_a = cy.reset_solution(self.problem, self.generator, pos_a)
            agent = self.population.create_agent(pos_a)
            pop_child.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_child = self.population.evaluate(pop_child, self.mode)
            self.population = self.population.spawn(cy.greedy_agents(pop_child, self.population, self.problem.sense))
        ## Persuing team - team B
        ## Step B1
        pop_new = []
        for idx in range(0, pop_size):
            pos_b = self.population[idx].solution.copy()
            for j in range(0, self.problem.n_dims):
                ### Eq.(6) in FBI Inspired Meta-Optimization
                pos_b[j] = self.generator.uniform() * self.population[idx].solution[
                    j
                ] + self.generator.uniform() * (
                                   self.g_best.solution[j] - self.population[idx].solution[j]
                           )
            pos_b = cy.reset_solution(self.problem, self.generator, pos_b)
            agent = self.population.create_agent(pos_b)
            pop_new.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        ## Step B2
        pop_child = []
        for idx, agent in enumerate(self.population.toarray()):
            rr = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
            if cy.is_better(agent, self.population[rr], self.problem.sense):
                ## Eq.(7) in FBI Inspired Meta-Optimization
                pos_b = (
                        agent.solution
                        + self.generator.uniform(0, 1, self.problem.n_dims)
                        * (self.population[rr].solution - agent.solution)
                        + self.generator.uniform()
                        * (self.g_best.solution - self.population[rr].solution)
                )
            else:
                ## Eq.(8) in FBI Inspired Meta-Optimization
                pos_b = (
                        agent.solution
                        + self.generator.uniform(0, 1, self.problem.n_dims)
                        * (agent.solution - self.population[rr].solution)
                        + self.generator.uniform()
                        * (self.g_best.solution - agent.solution)
                )
            pos_b = cy.reset_solution(self.problem, self.generator, pos_b)
            child = self.population.create_agent(pos_b)
            pop_child.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_child = self.population.evaluate(pop_child, self.mode)
            self.population = self.population.spawn(cy.greedy_agents(pop_child, self.population, self.problem.sense))

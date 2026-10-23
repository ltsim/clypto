#!/usr/bin/env python
# Created by "Thieu" at 16:58, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevGSKA(cy.Optimizer):
    """
    The developed version: Gaining Sharing Knowledge-based Algorithm (GSKA)

    Notes:
        + Third loop is removed, 2 parameters is removed
        + Solution represent junior or senior instead of dimension of solution
        + Equations is based vector, can handle large-scale problem
        + Apply the ideas of levy-flight and global best
        + Keep the better one after updating process

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.1, 0.5], percent of the best (p in the paper), default = 0.1
        + kr (float): [0.5, 0.9], knowledge ratio, default = 0.7

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import GSKA    >>> import numpy as np
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
    >>> model = GSKA.DevGSKA(epoch=1000, pop_size=50, pb = 0.1, kr = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    cdef public double kr
    cdef public double pb

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            pb: float = 0.1,
            kr: float = 0.7,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, n: pop_size, m: clusters
            pb (float): percent of the best 0.1%, 0.8%, 0.1% (p in the paper), default = 0.1
            kr (float): knowledge ratio, default = 0.7
        """
        super().__init__(parameters=["epoch", "pop_size", "pb", "kr"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.pb = cy.validator(float, pb, (0, 1.0), "pb")
        self.kr = cy.validator(float, kr, (0, 1.0), "kr")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        dd = int(np.ceil(pop_size * (1.0 - epoch / self.epoch)))
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            # If it is the best it chooses best+2, best+1
            if idx == 0:
                previ, nexti = idx + 2, idx + 1
            # If it is the worse it chooses worst-2, worst-1
            elif idx == pop_size - 1:
                previ, nexti = idx - 2, idx - 1
            # Other case it chooses i-1, i+1
            else:
                previ, nexti = idx - 1, idx + 1
            if idx < dd:  # senior gaining and sharing
                if self.generator.uniform() <= self.kr:
                    rand_idx = self.generator.choice(
                        list(set(range(0, pop_size)) - {previ, idx, nexti})
                    )
                    if cy.is_better(self.population[rand_idx], agent, self.problem.sense):
                        x = agent.solution + self.generator.uniform(
                            0, 1, self.problem.n_dims
                        ) * (
                                          self.population[previ].solution
                                          - self.population[nexti].solution
                                          + self.population[rand_idx].solution
                                          - agent.solution
                                  )
                    else:
                        x = self.g_best.solution + self.generator.uniform(
                            0, 1, self.problem.n_dims
                        ) * (self.population[rand_idx].solution - agent.solution)
                else:
                    x = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            else:  # junior gaining and sharing
                if self.generator.uniform() <= self.kr:
                    id1 = int(self.pb * pop_size)
                    id2 = int(id1 + pop_size * (1 - 2 * self.pb))
                    rand_best = self.generator.choice(list(set(range(0, id1)) - {idx}))
                    rand_worst = self.generator.choice(
                        list(set(range(id2, pop_size)) - {idx})
                    )
                    rand_mid = self.generator.choice(list(set(range(id1, id2)) - {idx}))
                    if cy.is_better(self.population[rand_mid], agent, self.problem.sense):
                        x = agent.solution + self.generator.uniform(
                            0, 1, self.problem.n_dims
                        ) * (
                                          self.population[rand_best].solution
                                          - self.population[rand_worst].solution
                                          + self.population[rand_mid].solution
                                          - agent.solution
                                  )
                    else:
                        x = self.g_best.solution + self.generator.uniform(
                            0, 1, self.problem.n_dims
                        ) * (self.population[rand_mid].solution - agent.solution)
                else:
                    x = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

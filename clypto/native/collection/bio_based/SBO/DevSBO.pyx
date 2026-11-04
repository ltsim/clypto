#!/usr/bin/env python
# Created by "Thieu" at 12:48, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevSBO(cy.Optimizer):
    """
    The developed version: Satin Bowerbird Optimizer (SBO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2017.01.006

    Notes:
        The original version can't handle negative fitness value.
        I remove all third loop for faster training, remove equation (1, 2) in the paper, calculate probability by roulette-wheel.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.5, 3.0] -> better [0.5, 2.0], the greatest step size
        + p_m (float): (0, 1.0) -> better [0.01, 0.2], mutation probability
        + psw (float): (0, 1.0) -> better [0.01, 0.1], proportion of space width (z in the paper)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.bio_based import SBO    >>> import numpy as np
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
    >>> model = SBO.DevSBO(epoch=1000, pop_size=50, alpha = 0.9, p_m =0.05, psw = 0.02)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            alpha: float = 0.94,
            p_m: float = 0.05,
            psw: float = 0.02,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): the greatest step size, default=0.94
            p_m (float): mutation probability, default=0.05
            psw (float): proportion of space width (z in the paper), default=0.02
        """
        super().__init__(parameters=["epoch", "pop_size", "alpha", "p_m", "psw"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.alpha = cy.validator(float, alpha, [0.5, 3.0], "alpha")
        self.p_m = cy.validator(float, p_m, (0, 1.0), "p_m")
        self.psw = cy.validator(float, psw, (0, 1.0), "psw")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # (percent of the difference between the upper and lower limit (Eq. 7))
        self.sigma = self.psw * (self.problem.bounds.up - self.problem.bounds.low)

        ## Calculate the probability of bowers using my equation
        fit_list = np.array([agent.fitness for agent in self.population])
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            ### Select a bower using roulette wheel
            rdx = cy.roulette_wheel(self.generator, self.problem.sense, fit_list)
            ### Calculating Step Size
            lamda = self.alpha * self.generator.uniform()
            x = agent.solution + lamda * (
                    (self.population[rdx].solution + self.g_best.solution) / 2
                    - agent.solution
            )
            ### Mutation
            temp = (
                    agent.solution
                    + self.generator.normal(0, 1, self.problem.n_dims) * self.sigma
            )
            x = np.where(
                self.generator.random(self.problem.n_dims) < self.p_m, temp, x
            )
            ### In-bound position
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

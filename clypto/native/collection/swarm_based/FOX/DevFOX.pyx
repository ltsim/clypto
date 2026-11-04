#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevFOX(cy.Optimizer):
    """
    The developed version of: Fox Optimizer (FOX)

    Notes (parameters):
        1. c1 (float): the coefficient of jumping (c1 in the paper), default = 0.18
        2. c2 (float): the coefficient of jumping (c2 in the paper), default = 0.82
        3. pp (float): the probability of choosing the exploration and exploitation phase, default=0.5

    Notes:
        1. Set parameter pp = 0.18 if you want to same as Original version
        2. The different between Dev and Original version is the equation: self.g_best.solution + self.generator.standard_normal(self.problem.n_dims) * (self.mint * aa)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import FOX    >>> import numpy as np
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
    >>> model = FOX.DevFOX(epoch=1000, pop_size=50, c1=0.18, c2=0.82, pp=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohammed, H., & Rashid, T. (2023). FOX: a FOX-inspired optimization algorithm. Applied Intelligence, 53(1), 1030-1050.
    """

    cdef public double c1
    cdef public double c2
    cdef public double pp

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            c1: float = 0.18,
            c2: float = 0.82,
            pp=0.5,
            **kwargs: object
    ) -> None:
        super().__init__(parameters=["epoch", "pop_size", "c1", "c2", "pp"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.c1 = cy.validator(float, c1, (-100.0, 100.0), "c1")
        self.c2 = cy.validator(float, c2, (-100.0, 100.0), "c2")
        self.pp = cy.validator(float, pp, (0.0, 1.0), "pp")

    def initialize_variables(self):
        self.mint = 10000000

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        aa = 2 * (1 - (1.0 / self.epoch))
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            if self.generator.random() >= 0.5:
                t1 = self.generator.random(self.problem.n_dims)
                sps = self.g_best.solution / t1
                dis = 0.5 * sps * t1
                tt = np.mean(t1)
                t = tt / 2
                jump = 0.5 * 9.81 * t ** 2
                if self.generator.random() > self.pp:
                    x = dis * jump * self.c1
                else:
                    x = dis * jump * self.c2
                if self.mint > tt:
                    self.mint = tt
            else:
                x = self.g_best.solution + self.generator.standard_normal(
                    self.problem.n_dims
                ) * (self.mint * aa)
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = child
        if self.mode != "sequential":
            self.population = self.population.spawn(self.population.evaluate(n_population, self.mode))

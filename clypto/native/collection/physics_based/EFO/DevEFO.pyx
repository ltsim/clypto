#!/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevEFO(cy.Optimizer):
    """
    The developed version: Electromagnetic Field Optimization (EFO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r_rate (float): [0.1, 0.6], default = 0.3, like mutation parameter in GA but for one variable
        + ps_rate (float): [0.5, 0.95], default = 0.85, like crossover parameter in GA
        + p_field (float): [0.05, 0.3], default = 0.1, portion of population, positive field
        + n_field (float): [0.3, 0.7], default = 0.45, portion of population, negative field

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.physics_based import EFO    >>> import numpy as np
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
    >>> model = EFO.DevEFO(epoch=1000, pop_size=50, r_rate = 0.3, ps_rate = 0.85, p_field = 0.1, n_field = 0.45)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            r_rate: float = 0.3,
            ps_rate: float = 0.85,
            p_field: float = 0.1,
            n_field: float = 0.45,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (float): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (float): default = 0.85    Like crossover parameter in GA
            p_field (float): default = 0.1     portion of population, positive field
            n_field (float): default = 0.45    portion of population, negative field
        """
        super().__init__(parameters=["epoch", "pop_size", "r_rate", "ps_rate", "p_field", "n_field"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.r_rate = cy.validator(float, r_rate, (0, 1.0), "r_rate")
        self.ps_rate = cy.validator(float, ps_rate, (0, 1.0), "ps_rate")
        self.p_field = cy.validator(float, p_field, (0, 1.0), "p_field")
        self.n_field = cy.validator(float, n_field, (0, 1.0), "n_field")
        self.phi = (1 + np.sqrt(5)) / 2  # golden ratio

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            r_idx1 = self.generator.integers(
                0, int(pop_size * self.p_field)
            )  # top
            r_idx2 = self.generator.integers(
                int(pop_size * (1 - self.n_field)), pop_size
            )  # bottom
            r_idx3 = self.generator.integers(
                int((pop_size * self.p_field) + 1),
                int(pop_size * (1 - self.n_field)),
            )  # middle
            if self.generator.random() < self.ps_rate:
                x = (
                        self.population[r_idx1].solution
                        + self.phi
                        * self.generator.random()
                        * (self.g_best.solution - self.population[r_idx3].solution)
                        + self.generator.random()
                        * (self.g_best.solution - self.population[r_idx2].solution)
                )
            else:
                x = self.problem.generate_solution()
            # replacement of one electromagnet of generated particle with a random number
            # (only for some generated particles) to bring diversity to the population
            if self.generator.random() < self.r_rate:
                RI = self.generator.integers(0, self.problem.n_dims)
                x[self.generator.integers(0, self.problem.n_dims)] = (
                    self.generator.uniform(self.problem.bounds.low[RI], self.problem.bounds.up[RI])
                )
            # checking whether the generated number is inside boundary or not
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

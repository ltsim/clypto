#!/usr/bin/env python
# Created by "Thieu" at 14:53, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalGOA(cy.Optimizer):
    """
    The original version of: Grasshopper Optimization Algorithm (GOA)

    Links:
        1. https://dx.doi.org/10.1016/j.advengsoft.2017.01.004
        2. https://www.mathworks.com/matlabcentral/fileexchange/61421-grasshopper-optimisation-algorithm-goa

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c_min (float): coefficient c min, default = 0.00004
        + c_max (float): coefficient c max, default = 2.0

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import GOA    >>> import numpy as np
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
    >>> model = GOA.OriginalGOA(epoch=1000, pop_size=50, c_min = 0.00004, c_max = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Saremi, S., Mirjalili, S. and Lewis, A., 2017. Grasshopper optimisation algorithm:
    theory and application. Advances in Engineering Software, 105, pp.30-47.
    """

    cdef public double c_max
    cdef public double c_min

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            c_min: float = 0.00004,
            c_max: float = 2.0,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c_min (float): coefficient c min, default=0.00004
            c_max (float): coefficient c max, default=2.0
        """
        super().__init__(parameters=["epoch", "pop_size", "c_min", "c_max"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.c_min = cy.validator(float, c_min, [0.00001, 0.2], "c_min")
        self.c_max = cy.validator(float, c_max, [0.2, 5.0], "c_max")

    def s_function__(self, r_vector=None):
        f = 0.5
        l = 1.5
        # Eq.(2.3) in the paper
        return f * np.exp(-r_vector / l) - np.exp(-r_vector)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Eq.(2.8) in the paper
        c = self.c_max - epoch * ((self.c_max - self.c_min) / self.epoch)
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            S_i_total = np.zeros(self.problem.n_dims)
            for j in range(0, pop_size):
                dist = np.sqrt(
                    np.sum((agent.solution - self.population[j].solution) ** 2)
                )
                r_ij_vector = (agent.solution - self.population[j].solution) / (
                        dist + self.EPSILON
                )  # xj - xi / dij in Eq.(2.7)
                xj_xi = 2 + np.remainder(dist, 2)  # |xjd - xid| in Eq. (2.7)
                ## The first part inside the big bracket in Eq. (2.7)   16 955 230 764    212 047 193 643
                ran = (c / 2) * (self.problem.bounds.up - self.problem.bounds.low)
                s_ij = ran * self.s_function__(xj_xi) * r_ij_vector
                S_i_total += s_ij
            x_new = (
                    c * self.generator.normal(0, 1, self.problem.n_dims) * S_i_total
                    + self.g_best.solution
            )  # Eq. (2.7) in the paper
            x = cy.correct_solution(self.problem, x_new)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

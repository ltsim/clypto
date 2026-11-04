#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class WMQIMRFO(cy.Optimizer):
    """
    The original version of: Wavelet Mutation and Quadratic Interpolation MRFO (WMQIMRFO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2021.108071

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2
        + pm (float): (0.0, 1.0), probability mutation, default = 0.5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import MRFO    >>> import numpy as np
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
    >>> model = MRFO.WMQIMRFO(epoch=1000, pop_size=50, somersault_range = 2.0, pm=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] G. Hu, M. Li, X. Wang et al., An enhanced manta ray foraging optimization algorithm for shape optimization of
    complex CCG-Ball curves, Knowledge-Based Systems (2022), doi: https://doi.org/10.1016/j.knosys.2021.108071.
    """

    cdef public double pm
    cdef public double somersault_range

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            somersault_range: float = 2.0,
            pm: float = 0.5,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
            pm (float): probability mutation, default = 0.5
        """
        super().__init__(parameters=["epoch", "pop_size", "somersault_range", "pm"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.somersault_range = cy.validator(float, somersault_range, [1.0, 5.0], "somersault_range")
        self.pm = cy.validator(float, pm, (0.0, 1.0), "pm")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_new = []
        for idx in range(0, pop_size):
            x_t = self.population[idx].solution
            x_t1 = self.population[idx - 1].solution

            ## Morlet wavelet mutation strategy
            ## Goal is to jump out of local optimum --> Performed in exploration stage
            s_constant = 2.0
            a = s_constant * (1.0 / s_constant) ** (1.0 - epoch / self.epoch)
            theta = self.generator.uniform(-2.5 * a, 2.5 * a)
            x = theta / a
            w = np.exp(-(x ** 2) / 2) * np.cos(5 * x)
            xichma = 1.0 / np.sqrt(a) * w

            if self.generator.random() < 0.5:  # Control parameter adjustment
                coef = np.log(1 + (np.e - 1.0) * epoch / self.epoch)  # Eq. 3.11

                r1 = self.generator.uniform()
                beta = (
                        2
                        * np.exp(r1 * (self.epoch - epoch) / self.epoch)
                        * np.sin(2 * np.pi * r1)
                )

                if coef < self.generator.random():  # Cyclone foraging
                    x_rand = self.problem.generate_solution()
                    if self.generator.random() < self.pm:  # Morlet wavelet mutation
                        if idx == 0:
                            pos_new = (
                                    x_rand
                                    + self.generator.random() * (x_rand - x_t)
                                    + beta * (x_rand - x_t)
                            )
                        else:
                            pos_new = (
                                    x_rand
                                    + self.generator.random() * (x_t1 - x_t)
                                    + beta * (x_rand - x_t)
                            )
                    else:
                        conditions = (
                                self.generator.uniform(0, 1, self.problem.n_dims) > 0.5
                        )
                        if idx == 0:
                            t1 = (
                                    x_rand
                                    + self.generator.random(self.problem.n_dims)
                                    * (x_rand - x_t)
                                    + beta * (x_rand - x_t)
                                    + xichma * (self.problem.bounds.up - x_t)
                            )
                            t2 = (
                                    x_rand
                                    + self.generator.random(self.problem.n_dims)
                                    * (x_rand - x_t)
                                    + beta * (x_rand - x_t)
                                    + xichma * (x_t - self.problem.bounds.low)
                            )
                        else:
                            t1 = (
                                    x_rand
                                    + self.generator.random(self.problem.n_dims)
                                    * (x_t1 - x_t)
                                    + beta * (x_rand - x_t)
                                    + xichma * (self.problem.bounds.up - x_t)
                            )
                            t2 = (
                                    x_rand
                                    + self.generator.random(self.problem.n_dims)
                                    * (x_t1 - x_t)
                                    + beta * (x_rand - x_t)
                                    + xichma * (x_t - self.problem.bounds.low)
                            )
                        pos_new = np.where(conditions, t1, t2)
                else:
                    if idx == 0:
                        pos_new = (
                                self.g_best.solution
                                + self.generator.random() * (self.g_best.solution - x_t)
                                + beta * (self.g_best.solution - x_t)
                        )
                    else:
                        pos_new = (
                                self.g_best.solution
                                + self.generator.random() * (x_t1 - x_t)
                                + beta * (self.g_best.solution - x_t)
                        )
            else:  # Chain foraging (Eq. 1,2)
                r = self.generator.random()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    pos_new = (
                            x_t
                            + r * (self.g_best.solution - x_t)
                            + alpha * (self.g_best.solution - x_t)
                    )
                else:
                    pos_new = (
                            x_t + r * (x_t1 - x_t) + alpha * (self.g_best.solution - x_t)
                    )
            pos_new = cy.correct_solution(self.problem, pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent, self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        ranked = self.population.sort()
        g_best = ranked[0]

        # Somersault foraging   (Eq. 8)
        pop_child = []
        for idx in range(0, pop_size):
            pos_new = self.population[idx].solution + self.somersault_range * (
                    self.generator.random() * g_best.solution
                    - self.generator.random() * self.population[idx].solution
            )
            pos_new = cy.correct_solution(self.problem, pos_new)
            agent = self.population.create_agent(pos_new)
            pop_child.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent, self.problem.sense)
        if self.mode != "sequential":
            pop_child = self.population.evaluate(pop_child, self.mode)
            self.population = self.population.greedy(pop_child)
        self.population = self.population.sort()
        g_best = self.population[0]

        # Quadratic Interpolation
        pop_new = []
        for idx, agent in enumerate(self.population.toarray()):
            idx2, idx3 = idx + 1, idx + 2
            if idx == pop_size - 2:
                idx2, idx3 = idx + 1, 0
            if idx == pop_size - 1:
                idx2, idx3 = 0, 1
            f1, f2, f3 = (
                agent.fitness,
                self.population[idx2].fitness,
                self.population[idx3].fitness,
            )
            x1, x2, x3 = (
                agent.solution,
                self.population[idx2].solution,
                self.population[idx3].solution,
            )
            a = (
                    f1 / ((x1 - x2) * (x1 - x3) + self.EPSILON)
                    + f2 / ((x2 - x1) * (x2 - x3) + self.EPSILON)
                    + f3 / ((x3 - x1) * (x3 - x2) + self.EPSILON)
            )
            gx = (
                         (x3 ** 2 - x2 ** 2) * f1 + (x1 ** 2 - x3 ** 2) * f2 + (x2 ** 2 - x1 ** 2) * f3
                 ) / (2 * ((x3 - x2) * f1 + (x1 - x3) * f2 + (x2 - x1) * f3) + self.EPSILON)
            pos_new = np.where(a > 0, gx, x1)
            pos_new = cy.correct_solution(self.problem, pos_new)
            child = self.population.create_agent(pos_new)
            pop_new.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], child, self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalMRFO(cy.Optimizer):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.103300

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2

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
    >>> model = MRFO.OriginalMRFO(epoch=1000, pop_size=50, somersault_range = 2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Zhang, Z. and Wang, L., 2020. Manta ray foraging optimization: An effective bio-inspired
    optimizer for engineering applications. Engineering Applications of Artificial Intelligence, 87, p.103300.
    """

    cdef public double somersault_range

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            somersault_range: float = 2.0,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
        """
        super().__init__(parameters=["epoch", "pop_size", "somersault_range"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.somersault_range = cy.validator(float, somersault_range, [1.0, 5.0], "somersault_range")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx in range(0, pop_size):
            # Cyclone foraging (Eq. 5, 6, 7)
            if self.generator.random() < 0.5:
                r1 = self.generator.uniform()
                beta = (
                        2
                        * np.exp(r1 * (self.epoch - epoch) / self.epoch)
                        * np.sin(2 * np.pi * r1)
                )

                if (epoch + 1) / self.epoch < self.generator.random():
                    x_rand = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
                    if idx == 0:
                        x_t1 = (
                                x_rand
                                + self.generator.uniform()
                                * (x_rand - self.population[idx].solution)
                                + beta * (x_rand - self.population[idx].solution)
                        )
                    else:
                        x_t1 = (
                                x_rand
                                + self.generator.uniform()
                                * (self.population[idx - 1].solution - self.population[idx].solution)
                                + beta * (x_rand - self.population[idx].solution)
                        )
                else:
                    if idx == 0:
                        x_t1 = (
                                self.g_best.solution
                                + self.generator.uniform()
                                * (self.g_best.solution - self.population[idx].solution)
                                + beta * (self.g_best.solution - self.population[idx].solution)
                        )
                    else:
                        x_t1 = (
                                self.g_best.solution
                                + self.generator.uniform()
                                * (self.population[idx - 1].solution - self.population[idx].solution)
                                + beta * (self.g_best.solution - self.population[idx].solution)
                        )
            # Chain foraging (Eq. 1,2)
            else:
                r = self.generator.uniform()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    x_t1 = (
                            self.population[idx].solution
                            + r * (self.g_best.solution - self.population[idx].solution)
                            + alpha * (self.g_best.solution - self.population[idx].solution)
                    )
                else:
                    x_t1 = (
                            self.population[idx].solution
                            + r * (self.population[idx - 1].solution - self.population[idx].solution)
                            + alpha * (self.g_best.solution - self.population[idx].solution)
                    )
            x = cy.correct_solution(self.problem, x_t1)
            agent = self.population.create_agent(x)
            n_population.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent, self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)
        ranked = self.population.sort()
        g_best = ranked[0]
        pop_child = []
        for idx, agent in enumerate(self.population.toarray()):
            # Somersault foraging   (Eq. 8)
            x_t1 = agent.solution + self.somersault_range * (
                    self.generator.uniform() * g_best.solution
                    - self.generator.uniform() * agent.solution
            )
            x = cy.correct_solution(self.problem, x_t1)
            child = self.population.create_agent(x)
            pop_child.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], child, self.problem.sense)
        if self.mode != "sequential":
            pop_child = self.population.evaluate(pop_child, self.mode)
            self.population = self.population.greedy(pop_child)

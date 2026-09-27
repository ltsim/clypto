#!/usr/bin/env python
# Created by "Thieu" at 12:17, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalIWO(cy.Optimizer):
    """
    The original version of: Invasive Weed Optimization (IWO)

    Links:
        1. https://pdfs.semanticscholar.org/734c/66e3757620d3d4016410057ee92f72a9853d.pdf

    Notes:
        Better to use normal distribution instead of uniform distribution,
        updating population by sorting both parent population and child population

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + seed_min (int): [1, 3], Number of Seeds (min)
        + seed_max (int): [4, pop_size/2], Number of Seeds (max)
        + exponent (int): [2, 4], Variance Reduction Exponent
        + sigma_start (float): [0.5, 5.0], The initial value of Standard Deviation
        + sigma_end (float): (0, 0.5), The final value of Standard Deviation

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.bio_based import IWO    >>> import numpy as np
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
    >>> model = IWO.OriginalIWO(epoch=1000, pop_size=50, seed_min = 3, seed_max = 9, exponent = 3, sigma_start = 0.6, sigma_end = 0.01)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mehrabian, A.R. and Lucas, C., 2006. A novel numerical optimization algorithm inspired from weed colonization.
    Ecological informatics, 1(4), pp.355-366.
    """

    cdef public int exponent
    cdef public int seed_max
    cdef public int seed_min
    cdef public double sigma_end
    cdef public double sigma_start

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            seed_min: int = 2,
            seed_max: int = 10,
            exponent: int = 2,
            sigma_start: float = 1.0,
            sigma_end: float = 0.01,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            seed_min (int): Number of Seeds (min)
            seed_max (int): Number of seeds (max)
            exponent (int): Variance Reduction Exponent
            sigma_start (float): The initial value of standard deviation
            sigma_end (float): The final value of standard deviation
        """
        super().__init__(parameters=[ "epoch", "pop_size", "seed_min", "seed_max", "exponent", "sigma_start", "sigma_end", ], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.seed_min = cy.validator(int, seed_min, [1, 3], "seed_min")
        self.seed_max = cy.validator(int, seed_max, [4, int(self.population.size() / 2)], "seed_max")
        self.exponent = cy.validator(int, exponent, [2, 4], "exponent")
        self.sigma_start = cy.validator(float, sigma_start, [0.5, 5.0], "sigma_start")
        self.sigma_end = cy.validator(float, sigma_end, (0, 0.5), "sigma_end")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Update Standard Deviation
        sigma = (1.0 - epoch / self.epoch) ** self.exponent * (
                self.sigma_start - self.sigma_end
        ) + self.sigma_end
        pop = self.population.sort()
        list_best = [cy.duplicate_agent(agent) for agent in pop[:1]]
        list_worst = [cy.duplicate_agent(agent) for agent in pop[::-1][:1]]
        best, worst = list_best[0], list_worst[0]
        pop_new = []
        for idx in range(0, pop_size):
            temp = best.fitness - worst.fitness
            if temp == 0:
                ratio = self.generator.random()
            else:
                ratio = (pop[idx].fitness - worst.fitness) / temp
            s = int(np.ceil(self.seed_min + (self.seed_max - self.seed_min) * ratio))
            if s > int(np.sqrt(pop_size)):
                s = int(np.sqrt(pop_size))
            pop_local = []
            for jdx in range(s):
                # Initialize Offspring and Generate Random Location
                x = pop[idx].solution + sigma * self.generator.normal(
                    0, 1, self.problem.n_dims
                )
                x = cy.correct_solution(self.problem, x)
                agent = self.population.create_agent(x)
                pop_local.append(agent)
            pop_local = self.population.evaluate(pop_local, self.mode)
            pop_new += pop_local
        self.population = self.population.spawn(cy.sort_agents(pop_new, self.problem.sense)[:pop_size])

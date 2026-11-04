#!/usr/bin/env python
# Created by "Thieu" at 15:05, 03/06/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class OriginalSLO(cy.Optimizer):
    """
    The original version of: Sea Lion Optimization Algorithm (SLO)

    Notes:
        + There are some unclear equations and parameters in the original paper
        + https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
        + https://doi.org/10.14569/IJACSA.2019.0100548

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import SLO    >>> import numpy as np
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
    >>> model = SLO.OriginalSLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Masadeh, R., Mahafzah, B.A. and Sharieh, A., 2019. Sea lion optimization algorithm. Sea, 10(5), p.388.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=cy.Population)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        c = 2 - 2 * epoch / self.epoch
        t0 = self.generator.random()
        v1 = np.sin(2 * np.pi * t0)
        v2 = np.sin(2 * np.pi * (1 - t0))
        SP_leader = np.abs(
            v1 * (1 + v2) / v2
        )  # In the paper this is not clear how to calculate
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            if SP_leader < 0.25:
                if c < 1:
                    x = self.g_best.solution - c * np.abs(
                        2 * self.generator.random() * self.g_best.solution
                        - agent.solution
                    )
                else:
                    ri = self.generator.choice(
                        list(set(range(0, pop_size)) - {idx})
                    )  # random index
                    x = self.population[ri].solution - c * np.abs(
                        2 * self.generator.random() * self.population[ri].solution
                        - agent.solution
                    )
            else:
                x = (
                    np.abs(self.g_best.solution - agent.solution)
                    * np.cos(2 * np.pi * self.generator.uniform(-1, 1))
                    + self.g_best.solution
                )
            # In the paper doesn't check also doesn't update old solution at this point
            x = cy.reset_solution(self.problem, self.generator, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], child, self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

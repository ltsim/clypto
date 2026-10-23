#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from math import gamma
import numpy as np
cimport clypto.core as cy



cdef class OriginalMSA(cy.Optimizer):
    """
    The original version: Moth Search Algorithm (MSA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/59010-moth-search-ms-algorithm
        2. https://doi.org/10.1007/s12293-016-0212-3

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_best (int): [3, 10], how many of the best moths to keep from one generation to the next, default=5
        + partition (float): [0.3, 0.8], The proportional of first partition, default=0.5
        + max_step_size (float): [0.5, 2.0], Max step size used in Levy-flight technique, default=1.0

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import MSA    >>> import numpy as np
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
    >>> model = MSA.OriginalMSA(epoch=1000, pop_size=50, n_best = 5, partition = 0.5, max_step_size = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., 2018. Moth search algorithm: a bio-inspired metaheuristic algorithm for
    global optimization problems. Memetic Computing, 10(2), pp.151-164.
    """

    cdef public double max_step_size
    cdef public int n_best
    cdef public double partition

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            n_best: int = 5,
            partition: float = 0.5,
            max_step_size: float = 1.0,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_best (int): how many of the best moths to keep from one generation to the next, default=5
            partition (float): The proportional of first partition, default=0.5
            max_step_size (float): Max step size used in Levy-flight technique, default=1.0
        """
        super().__init__(parameters=["epoch", "pop_size", "n_best", "partition", "max_step_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[10, 10000])
        self.n_best = cy.validator(int, n_best, [2, int(self.population.size() / 2)], "n_best")
        self.partition = cy.validator(float, partition, (0, 1.0), "partition")
        self.max_step_size = cy.validator(float, max_step_size, (0, 5.0), "max_step_size")
        # np1 in paper
        self.n_moth1 = int(np.ceil(self.partition * self.population.size()))
        # np2 in paper, we actually don't need this variable
        self.n_moth2 = self.population.size() - self.n_moth1
        # you can change this ratio so as to get much better performance
        self.golden_ratio = (np.sqrt(5) - 1) / 2.0

    def _levy_walk(self, iteration):
        beta = 1.5  # Eq. 2.23
        sigma = (
                        gamma(1 + beta)
                        * np.sin(np.pi * (beta - 1) / 2)
                        / (gamma(beta / 2) * (beta - 1) * 2 ** ((beta - 2) / 2))
                ) ** (1 / (beta - 1))
        u = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up) * sigma
        v = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
        step = u / np.abs(v) ** (1.0 / (beta - 1))  # Eq. 2.21
        scale = self.max_step_size / iteration
        delta_x = scale * step
        return delta_x

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_best = [cy.duplicate_agent(agent) for agent in self.population[: self.n_best]]
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            # Migration operator
            if idx < self.n_moth1:
                # scale = self.max_step_size / (epoch+1)       # Smaller step for local walk
                x = agent.solution + self.generator.random(
                    self.problem.n_dims
                ) * self._levy_walk(epoch)
            else:
                # Flying in a straight line
                temp_case1 = agent.solution + self.generator.random(
                    self.problem.n_dims
                ) * self.golden_ratio * (self.g_best.solution - agent.solution)
                temp_case2 = agent.solution + self.generator.random(
                    self.problem.n_dims
                ) * (1.0 / self.golden_ratio) * (
                                     self.g_best.solution - agent.solution
                             )
                x = np.where(
                    self.generator.random(self.problem.n_dims) < 0.5,
                    temp_case2,
                    temp_case1,
                )
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], child, self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)
        self.population = self.population.sort()
        # Replace the worst with the previous generation's elites.
        for idx in range(0, self.n_best):
            self.population[-1 - idx] = cy.duplicate_agent(pop_best[idx])

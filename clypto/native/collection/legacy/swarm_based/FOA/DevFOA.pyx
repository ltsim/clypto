#!/usr/bin/env python
# Created by "Thieu" at 14:01, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
from clypto.native.collection.legacy.swarm_based.FOA.OriginalFOA cimport OriginalFOA
cimport clypto.core as cy


cdef class DevFOA(OriginalFOA):
    """
    The developed version: Fruit-fly Optimization Algorithm (FOA)

    Notes:
        + The fitness function (small function) is changed by taking the distance each 2 adjacent dimensions
        + Update the position if only new generated solution is better
        + The updated position is created by norm distance * gaussian random number

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import FOA    >>> import numpy as np
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
    >>> model = FOA.DevFOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        c = 1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, pop_size):
            pos_new = self.population[idx].solution + self.generator.normal(
                self.problem.bounds.low, self.problem.bounds.up
            )
            pos_new = (
                c * self.generator.random() * self.norm_consecutive_adjacent__(pos_new)
            )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                # the classic code evaluates pos_new, not the agent's smell vector
                agent.update_solution(self.population.evaluate_solution(pos_new), agent.solution)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = cy.greedy_agents(pop_new, self.population, self.problem.sense)

#!/usr/bin/env python
# Created by "Thieu" at 14:01, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

from clypto.native.collection.legacy.swarm_based.FOA.OriginalFOA cimport OriginalFOA


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
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
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

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        c = 1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution + self.generator.normal(
                self.problem.bounds.low, self.problem.bounds.up
            )
            pos_new = (
                c * self.generator.random() * self.norm_consecutive_adjacent__(pos_new)
            )
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.pop[idx] = self._get_better_agent(
                    agent, self.pop[idx], self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            self.pop = self._greedy_selection_population(
                pop_new, self.pop, self.problem.sense
            )

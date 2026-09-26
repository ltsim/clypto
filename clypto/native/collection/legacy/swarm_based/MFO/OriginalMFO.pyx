#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class OriginalMFO(LegacyOptimizer):
    """
    The developed version: Moth-Flame Optimization (MFO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import MFO    >>> import numpy as np
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
    >>> model = MFO.OriginalMFO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2015. Moth-flame optimization algorithm: A novel nature-inspired
    heuristic paradigm. Knowledge-based systems, 89, pp.228-249.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self._set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # Number of flames Eq.(3.14) in the paper (linearly decreased)
        num_flame = round(self.pop_size - epoch * ((self.pop_size - 1) / self.epoch))
        # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1.0 + epoch * (-1.0 / self.epoch)
        pop_flames = self._get_sorted_population(self.pop, self.problem.sense)
        g_best = pop_flames[0].copy()
        pop_new = []
        for idx in range(0, self.pop_size):
            #   D in Eq.(3.13)
            distance_to_flame = np.abs(
                pop_flames[idx].solution - self.pop[idx].solution
            )
            t = (a - 1) * self.generator.uniform() + 1
            b = 1
            # Update the position of the moth with respect to its corresponding flame, Eq.(3.12).
            temp_1 = (
                    distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi)
                    + pop_flames[idx].solution
            )
            # Update the position of the moth with respect to one flame Eq.(3.12).
            temp_2 = (
                    distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi)
                    + g_best.solution
            )
            list_idx = idx * np.ones(self.problem.n_dims)
            pos_new = np.where(list_idx < num_flame, temp_1, temp_2)
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.pop[idx] = self._get_better_agent(
                    self.pop[idx], agent, self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            self.pop = self._greedy_selection_population(
                self.pop, pop_new, self.problem.sense
            )

#!/usr/bin/env python
# Created by "Thieu" at 10:08, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class SwarmHC(LegacyOptimizer):
    """
    The developed version: Swarm-based Hill Climbing (S-HC)

    Notes
    ~~~~~
    + Based on swarm-of people are trying to climb on the mountain idea
    + The number of neighbour solutions are equal to population size
    + The step size to calculate neighbour is randomized and based on rank of solution.
        + The guys near on top of mountain will move slower than the guys on bottom of mountain.
        + Imagination: exploration when far from global best, and exploitation when near global best
    + Who on top of mountain first will be the winner. (global optimal)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + neighbour_size (int): [2, pop_size/2], fixed parameter, sensitive exploitation parameter, Default: 10

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.math_based import HC    >>> import numpy as np
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
    >>> model = HC.SwarmHC(epoch=1000, pop_size=50, neighbour_size = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, neighbour_size=10, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 10
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.neighbour_size = self.validator.check_int(
            "neighbour_size", neighbour_size, [2, int(self.pop_size / 2)]
        )
        self._set_parameters(["epoch", "pop_size", "neighbour_size"])
        self.sort_flag = False

    def _evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ranks = np.array(list(range(1, self.pop_size + 1)))
        ranks = ranks / np.sum(ranks)
        step_size = np.mean(self.problem.bounds.up - self.problem.bounds.low) * np.exp(
            -2 * epoch / self.epoch
        )
        ss = step_size * ranks
        pop = []
        for idx in range(0, self.pop_size):
            pop_neighbours = []
            for jdx in range(0, self.neighbour_size):
                pos_new = (
                        self.pop[idx].solution
                        + self.generator.normal(0, 1, self.problem.n_dims) * ss[idx]
                )
                pos_new = self._correct_solution(pos_new)
                agent = self._generate_empty_agent(pos_new)
                pop_neighbours.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_neighbours[-1].target = self._get_target(pos_new)
            pop_neighbours = self._update_target_for_population(pop_neighbours)
            best_local = self._get_best_agent(pop_neighbours, self.problem.sense)
            pop.append(best_local)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx] = self._get_better_agent(
                    best_local, self.pop[idx], self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self._greedy_selection_population(
                self.pop, pop, self.problem.sense
            )

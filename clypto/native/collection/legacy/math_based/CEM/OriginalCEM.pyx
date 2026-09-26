#!/usr/bin/env python
# Created by "Thieu" at 18:08, 19/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class OriginalCEM(LegacyOptimizer):
    """
    The original version of: Cross-Entropy Method (CEM)

    Links:
        1. https://github.com/clever-algorithms/CleverAlgorithms
        2. https://doi.org/10.1007/s10479-005-5724-z

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_best (int): N selected solutions as a samples for next evolution
        + alpha (float): weight factor for means and stdevs (normal distribution)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.math_based import CEM    >>> import numpy as np
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
    >>> model = CEM.OriginalCEM(epoch=1000, pop_size=50, n_best = 20, alpha = 0.7)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] De Boer, P.T., Kroese, D.P., Mannor, S. and Rubinstein, R.Y., 2005. A tutorial on the
    cross-entropy method. Annals of operations research, 134(1), pp.19-67.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            n_best: int = 10,
            alpha: float = 0.7,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_best (int): N selected solutions as a samples for next evolution
            alpha (float): weight factor for means and stdevs (normal distribution)
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_best = self.validator.check_int(
            "n_best", n_best, [2, int(self.pop_size / 2)]
        )
        self.alpha = self.validator.check_float("alpha", alpha, (0, 1.0))
        self._set_parameters(["epoch", "pop_size", "n_best", "alpha"])
        self.sort_flag = True

    def _initialize_variables(self):
        self.means = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
        self.stdevs = np.abs(self.problem.bounds.up - self.problem.bounds.low)

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Selected the best samples and update means and stdevs
        pop_best = self.pop[: self.n_best]
        pos_list = np.array([agent.solution for agent in pop_best])
        means_new = np.mean(pos_list, axis=0)
        means_new_repeat = np.repeat(means_new.reshape((1, -1)), self.n_best, axis=0)
        stdevs_new = np.mean((pos_list - means_new_repeat) ** 2, axis=0)
        self.means = self.alpha * self.means + (1.0 - self.alpha) * means_new
        self.stdevs = np.abs(self.alpha * self.stdevs + (1.0 - self.alpha) * stdevs_new)
        ## Create new population for next generation
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.generator.normal(self.means, self.stdevs)
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
                self.pop, pop_new, self.problem.sense
            )

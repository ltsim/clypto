#!/usr/bin/env python
# Created by "Thieu" at 12:51, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class OriginalBBOA(LegacyOptimizer):
    """
    The original version of: Brown-Bear Optimization Algorithm (BBOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/125490-brown-bear-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.bio_based import BBOA    >>> import numpy as np
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
    >>> model = BBOA.OriginalBBOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Prakash, T., Singh, P. P., Singh, V. P., & Singh, S. N. (2023). A Novel Brown-bear Optimization
    Algorithm for Solving Economic Dispatch Problem. In Advanced Control & Optimization Paradigms for
    Energy System Operation and Management (pp. 137-164). River Publishers.
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
        pp = epoch / self.epoch

        ## Pedal marking behaviour
        pop_new = []
        for idx in range(0, self.pop_size):
            if pp <= 1 / 3:  # Gait while walking
                pos_new = self.pop[idx].solution + (
                        -pp
                        * self.generator.random(self.problem.n_dims)
                        * self.pop[idx].solution
                )
            elif 1 / 3 < pp <= 2 / 3:  # Careful Stepping
                qq = pp * self.generator.random(self.problem.n_dims)
                pos_new = self.pop[idx].solution + (
                        qq
                        * (
                                self.g_best.solution
                                - self.generator.integers(1, 3) * self.g_worst.solution
                        )
                )
            else:
                ww = 2 * pp * np.pi * self.generator.random(self.problem.n_dims)
                pos_new = (
                        self.pop[idx].solution
                        + (ww * self.g_best.solution - np.abs(self.pop[idx].solution))
                        - (ww * self.g_worst.solution - np.abs(self.pop[idx].solution))
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
                self.pop, pop_new, self.problem.sense
            )

        ## Sniffing of pedal marks
        pop_new = []
        for idx in range(0, self.pop_size):
            kk = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            if self._compare_target(
                    self.pop[idx].target, self.pop[kk].target, self.problem.sense
            ):
                pos_new = self.pop[idx].solution + self.generator.random() * (
                        self.pop[idx].solution - self.pop[kk].solution
                )
            else:
                pos_new = self.pop[idx].solution + self.generator.random() * (
                        self.pop[kk].solution - self.pop[idx].solution
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
                self.pop, pop_new, self.problem.sense
            )

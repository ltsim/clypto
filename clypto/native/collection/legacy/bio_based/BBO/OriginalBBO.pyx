#!/usr/bin/env python
# Created by "Thieu" at 12:24, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class OriginalBBO(LegacyOptimizer):
    """
    The original version of: Biogeography-Based Optimization (BBO)

    Links:
        1. https://ieeexplore.ieee.org/abstract/document/4475427

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_m (float): (0, 1) -> better [0.01, 0.2], Mutation probability
        + n_elites (int): (2, pop_size/2) -> better [2, 5], Number of elites will be keep for next generation

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.bio_based import BBO    >>> import numpy as np
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
    >>> model = BBO.OriginalBBO(epoch=1000, pop_size=50, p_m=0.01, n_elites=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Simon, D., 2008. Biogeography-based optimization. IEEE transactions on evolutionary computation, 12(6), pp.702-713.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            p_m: float = 0.01,
            n_elites: int = 2,
            **kwargs: object
    ) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 100
            p_m: Mutation probability, default=0.01
            n_elites: Number of elites will be keep for next generation, default=2
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_m = self.validator.check_float("p_m", p_m, (0.0, 1.0))
        self.n_elites = self.validator.check_int(
            "n_elites", n_elites, [2, int(self.pop_size / 2)]
        )
        self._set_parameters(["epoch", "pop_size", "p_m", "n_elites"])
        self.sort_flag = False
        self.mu = (self.pop_size + 1 - np.array(range(1, self.pop_size + 1))) / (
                self.pop_size + 1
        )
        self.mr = 1 - self.mu

    def _evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch: The current iteration
        """
        _, pop_elites, _ = self._get_special_agents(
            self.pop, n_best=self.n_elites, sense=self.problem.sense
        )
        pop = []
        for idx in range(0, self.pop_size):
            # Probabilistic migration to the i-th position
            pos_new = self.pop[idx].solution.copy()
            for j in range(self.problem.n_dims):
                if self.generator.random() < self.mr[idx]:  # Should we immigrate?
                    # Pick a position from which to emigrate (roulette wheel selection)
                    random_number = self.generator.random() * np.sum(self.mu)
                    select = self.mu[0]
                    select_index = 0
                    while (random_number > select) and (
                            select_index < self.pop_size - 1
                    ):
                        select_index += 1
                        select += self.mu[select_index]
                    # this is the migration step
                    pos_new[j] = self.pop[select_index].solution[j]
            noise = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            condition = self.generator.random(self.problem.n_dims) < self.p_m
            pos_new = np.where(condition, noise, pos_new)
            pos_new = self._correct_solution(pos_new)
            agent_new = self._generate_empty_agent(pos_new)
            pop.append(agent_new)
            if self.mode not in self.AVAILABLE_MODES:
                agent_new.target = self._get_target(pos_new)
                self.pop[idx] = self._get_better_agent(
                    self.pop[idx], agent_new, sense=self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop = self._update_target_for_population(pop)
            self.pop = self._greedy_selection_population(
                self.pop, pop, self.problem.sense
            )
        # replace the solutions with their new migrated and mutated versions then Merge Populations
        self.pop = self._get_sorted_and_trimmed_population(
            self.pop + pop_elites, self.pop_size, self.problem.sense
        )

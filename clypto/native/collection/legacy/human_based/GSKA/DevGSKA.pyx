#!/usr/bin/env python
# Created by "Thieu" at 16:58, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class DevGSKA(LegacyOptimizer):
    """
    The developed version: Gaining Sharing Knowledge-based Algorithm (GSKA)

    Notes:
        + Third loop is removed, 2 parameters is removed
        + Solution represent junior or senior instead of dimension of solution
        + Equations is based vector, can handle large-scale problem
        + Apply the ideas of levy-flight and global best
        + Keep the better one after updating process

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.1, 0.5], percent of the best (p in the paper), default = 0.1
        + kr (float): [0.5, 0.9], knowledge ratio, default = 0.7

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import GSKA    >>> import numpy as np
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
    >>> model = GSKA.DevGSKA(epoch=1000, pop_size=50, pb = 0.1, kr = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            pb: float = 0.1,
            kr: float = 0.7,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, n: pop_size, m: clusters
            pb (float): percent of the best 0.1%, 0.8%, 0.1% (p in the paper), default = 0.1
            kr (float): knowledge ratio, default = 0.7
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pb = self.validator.check_float("pb", pb, (0, 1.0))
        self.kr = self.validator.check_float("kr", kr, (0, 1.0))
        self._set_parameters(["epoch", "pop_size", "pb", "kr"])
        self.sort_flag = True

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        dd = int(np.ceil(self.pop_size * (1.0 - epoch / self.epoch)))
        pop_new = []
        for idx in range(0, self.pop_size):
            # If it is the best it chooses best+2, best+1
            if idx == 0:
                previ, nexti = idx + 2, idx + 1
            # If it is the worse it chooses worst-2, worst-1
            elif idx == self.pop_size - 1:
                previ, nexti = idx - 2, idx - 1
            # Other case it chooses i-1, i+1
            else:
                previ, nexti = idx - 1, idx + 1
            if idx < dd:  # senior gaining and sharing
                if self.generator.uniform() <= self.kr:
                    rand_idx = self.generator.choice(
                        list(set(range(0, self.pop_size)) - {previ, idx, nexti})
                    )
                    if self._compare_target(
                            self.pop[rand_idx].target,
                            self.pop[idx].target,
                            self.problem.sense,
                    ):
                        pos_new = self.pop[idx].solution + self.generator.uniform(
                            0, 1, self.problem.n_dims
                        ) * (
                                          self.pop[previ].solution
                                          - self.pop[nexti].solution
                                          + self.pop[rand_idx].solution
                                          - self.pop[idx].solution
                                  )
                    else:
                        pos_new = self.g_best.solution + self.generator.uniform(
                            0, 1, self.problem.n_dims
                        ) * (self.pop[rand_idx].solution - self.pop[idx].solution)
                else:
                    pos_new = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            else:  # junior gaining and sharing
                if self.generator.uniform() <= self.kr:
                    id1 = int(self.pb * self.pop_size)
                    id2 = int(id1 + self.pop_size * (1 - 2 * self.pb))
                    rand_best = self.generator.choice(list(set(range(0, id1)) - {idx}))
                    rand_worst = self.generator.choice(
                        list(set(range(id2, self.pop_size)) - {idx})
                    )
                    rand_mid = self.generator.choice(list(set(range(id1, id2)) - {idx}))
                    if self._compare_target(
                            self.pop[rand_mid].target,
                            self.pop[idx].target,
                            self.problem.sense,
                    ):
                        pos_new = self.pop[idx].solution + self.generator.uniform(
                            0, 1, self.problem.n_dims
                        ) * (
                                          self.pop[rand_best].solution
                                          - self.pop[rand_worst].solution
                                          + self.pop[rand_mid].solution
                                          - self.pop[idx].solution
                                  )
                    else:
                        pos_new = self.g_best.solution + self.generator.uniform(
                            0, 1, self.problem.n_dims
                        ) * (self.pop[rand_mid].solution - self.pop[idx].solution)
                else:
                    pos_new = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
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

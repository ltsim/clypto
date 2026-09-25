#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer._native.legacy cimport _LegacyOptimizer


cdef class AugmentedAEO(_LegacyOptimizer):
    """
    The original version of: Augmented Artificial Ecosystem Optimization (AAEO)

    Notes:
        + Used linear weight factor reduce from 2 to 0 through time
        + Applied Levy-flight technique and the global best solution

    Examples
    ~~~~~~~~
    >>> from clypto.collection.system_based import AEO    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = AEO.AugmentedAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Van Thieu, N., Barma, S. D., Van Lam, T., Kisi, O., & Mahesha, A. (2022). Groundwater level modeling
    using Augmented Artificial Ecosystem Optimization. Journal of Hydrology, 129034.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Production - Update the worst agent
        # Eq. 2, 3, 1
        wf = 2 * (1 - epoch / self.epoch)  # Weight factor
        a = (1.0 - epoch / self.epoch) * self.generator.random()
        x1 = (1 - a) * self.pop[-1].solution + a * self.generator.uniform(
            self.problem.lb, self.problem.ub
        )
        pos_new = self.correct_solution(x1)
        agent = self.generate_agent(pos_new)
        self.pop[-1] = agent
        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            if self.generator.random() < 0.5:
                rand = self.generator.random()
                # Eq. 4, 5, 6
                c = (
                        0.5
                        * self.generator.normal(0, 1)
                        / np.abs(self.generator.normal(0, 1))
                )  # Consumption factor
                j = 1 if idx == 0 else self.generator.integers(0, idx)
                ### Herbivore
                if rand < 1.0 / 3:
                    pos_new = self.pop[idx].solution + wf * c * (
                            self.pop[idx].solution - self.pop[0].solution
                    )  # Eq. 6
                ### Omnivore
                elif 1.0 / 3 <= rand <= 2.0 / 3:
                    pos_new = self.pop[idx].solution + wf * c * (
                            self.pop[idx].solution - self.pop[j].solution
                    )  # Eq. 7
                ### Carnivore
                else:
                    r2 = self.generator.uniform()
                    pos_new = self.pop[idx].solution + wf * c * (
                            r2 * (self.pop[idx].solution - self.pop[0].solution)
                            + (1 - r2) * (self.pop[idx].solution - self.pop[j].solution)
                    )
            else:
                pos_new = self.pop[idx].solution + self.get_levy_flight_step(
                    1.0, 0.001, case=-1
                ) * (1.0 / np.sqrt(epoch)) * np.sign(self.generator.random() - 0.5) * (
                                  self.pop[idx].solution - self.g_best.solution
                          )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(
                self.pop[:-1], pop_new, self.problem.minmax
            )
        ## find current best used in decomposition
        best = self.get_best_agent(self.pop, self.problem.minmax)
        ## Decomposition
        ### Eq. 10, 11, 12, 9   idx, pop, g_best, local_best
        pop_child = []
        for idx in range(0, self.pop_size):
            if self.generator.random() < 0.5:
                pos_new = best.solution + self.generator.normal(
                    0, 1, self.problem.n_dims
                ) * (best.solution - self.pop[idx].solution)
            else:
                beta = self.generator.uniform(0.01, 1.0)
                pos_new = best.solution + self.get_levy_flight_step(
                    beta=beta, multiplier=0.01, size=self.problem.n_dims, case=0
                ) * (best.solution - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(
                self.pop, pop_child, self.problem.minmax
            )

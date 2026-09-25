#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native.fuzzy import FuzzySystem as FS

from clypto.optimizer._native.legacy cimport _LegacyOptimizer


cdef class FuzzyGWO(_LegacyOptimizer):
    """
    The original version of: Fuzzy Hierarchical Operator - Grey Wolf Optimizer (FHO-GWO or FuzzyGWO or F-GWO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2017.03.048

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.FuzzyGWO(epoch=1000, pop_size=50, fuzzy_name="increase")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rodríguez, Luis, Oscar Castillo, José Soria, Patricia Melin, Fevrier Valdez, Claudia I. Gonzalez, Gabriela E. Martinez, and Jesus Soto. "A fuzzy hierarchical operator in the grey wolf optimizer algorithm." Applied Soft Computing 57 (2017): 315-328.
    """

    FUZZY_OPERATORS = ["increase", "decrease"]

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        fuzzy_name: str = "increase",
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            fuzzy_name (str): type of fuzzy operator to use, default = "increase"
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.fuzzy_name = self.validator.check_str(
            "fuzzy_name", fuzzy_name, FuzzyGWO.FUZZY_OPERATORS
        )
        self.set_parameters(["epoch", "pop_size", "fuzzy_name"])
        self.sort_flag = False

    def initialize_variables(self) -> None:
        self.fuzzy_system = FS(self.fuzzy_name)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 - 2.0 * epoch / self.epoch
        _, list_best, _ = self.get_special_agents(
            self.pop, n_best=3, minmax=self.problem.minmax
        )
        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(
                C1 * list_best[0].solution - self.pop[idx].solution
            )
            X2 = list_best[1].solution - A2 * np.abs(
                C2 * list_best[1].solution - self.pop[idx].solution
            )
            X3 = list_best[2].solution - A3 * np.abs(
                C3 * list_best[2].solution - self.pop[idx].solution
            )

            # Get fuzzy weights
            FW_alpha, FW_beta, FW_delta = self.fuzzy_system.get_fuzzy_weights(
                epoch, self.epoch
            )
            total_weight = FW_alpha + FW_beta + FW_delta
            pos_new = (X1 * FW_alpha + X2 * FW_beta + X3 * FW_delta) / total_weight
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
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )

#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class ER_GWO(_LegacyOptimizer):
    """
    The original version of: Efficient and Robust Grey Wolf Optimizer (ER-GWO)

    Notes:
        + Slow convergence speed due to the (miu_factor)^(iteration) ==> Big number
        + Three more parameters than original GWO, increase the complexity of the algorithm.

    Links:
        1. https://doi.org/10.1007/s00500-019-03939-y

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
    >>> model = GWO.ER_GWO(epoch=1000, pop_size=50, a_initial=2.0, a_final=0.0, miu_factor=1.0001)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Long, W., Cai, S., Jiao, J. et al. An efficient and robust grey wolf optimizer algorithm for large-scale numerical optimization. Soft Comput 24, 997–1026 (2020).
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        a_initial: float = 2.0,
        a_final: float = 0.0,
        miu_factor: float = 1.0001,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a_initial (float): initial value of coefficient a, default = 2.0
            a_final (float): final value of coefficient a, default = 0.0
            miu_factor (float): nonlinear coefficient for equation (8), default = 1.0001
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.a_initial = self.validator.check_float("a_initial", a_initial, [0.0, 10.0])
        self.a_final = self.validator.check_float(
            "a_final", a_final, [0.0, self.a_initial]
        )
        self.miu_factor = self.validator.check_float(
            "miu_factor", miu_factor, [1.0001, 1.01]
        )  # Required in paper
        self.set_parameters(["epoch", "pop_size", "a_initial", "a_final", "miu_factor"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = self.a_initial - (self.a_initial - self.a_final) * self.miu_factor**epoch
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
            dist1 = np.linalg.norm(X1)
            dist2 = np.linalg.norm(X2)
            dist3 = np.linalg.norm(X3)
            total = dist1 + dist2 + dist3
            if total == 0:
                # Avoid division by zero
                pos_new = (X1 + X2 + X3) / 3.0
            else:
                # Normalize distances to avoid division by zero
                pos_new = (X1 * dist1 + X2 * dist2 + X3 * dist3) / total
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

#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.legacy.swarm_based.GWO.OriginalGWO cimport OriginalGWO


cdef class IGWO(OriginalGWO):
    """
    The original version of: Improved Grey Wolf Optimization (IGWO)

    Notes:
        1. Link: https://doi.org/10.1007/s00366-017-0567-1
        2. Inspired by: Mohammadtaher Abbasi (https://github.com/mtabbasi)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + a_min (float): Lower bound of a, default = 0.02
        + a_max (float): Upper bound of a, default = 2.2

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.IGWO(epoch=1000, pop_size=50, a_min = 0.02, a_max = 2.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kaveh, A. & Zakian, P.. (2018). Improved GWO algorithm for optimal design of truss structures.
    Engineering with Computers. 34. 10.1007/s00366-017-0567-1.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        a_min: float = 0.02,
        a_max: float = 2.2,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a_min (float): Lower bound of a, default = 0.02
            a_max (float): Upper bound of a, default = 2.2
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.a_min = self.validator.check_float("a_min", a_min, (0.0, 1.6))
        self.a_max = self.validator.check_float("a_max", a_max, [1.0, 4.0])
        self.set_parameters(["epoch", "pop_size", "a_min", "a_max"])
        self.growth_alpha = 2
        self.growth_delta = 3

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm.

        Args:
            epoch (int): The current iteration
        """
        _, list_best, _ = self.get_special_agents(
            self.pop, n_best=3, minmax=self.problem.minmax
        )
        pop_new = []
        for idx in range(0, self.pop_size):
            # IGWO functions
            a_alpha = self.a_max * np.exp(
                (epoch / self.epoch) ** self.growth_alpha
                * np.log(self.a_min / self.a_max)
            )
            a_delta = self.a_max * np.exp(
                (epoch / self.epoch) ** self.growth_delta
                * np.log(self.a_min / self.a_max)
            )
            a_beta = (a_alpha + a_delta) * 0.5
            A1 = a_alpha * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a_beta * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a_delta * (2 * self.generator.random(self.problem.n_dims) - 1)
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
            pos_new = (X1 + X2 + X3) / 3.0
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

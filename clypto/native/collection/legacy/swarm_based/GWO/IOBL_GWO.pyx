#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class IOBL_GWO(_LegacyOptimizer):
    """
    The original version of: Improved Opposite-based Learning Grey Wolf Optimizer (IOBL-GWO)

    Notes:
        + In the paper, they called it "Improved Grey Wolf Optimizer (IGWO)", but there are many improved versions of GWO.
        + So based on their proposed equations, we called it as "Improved Opposite-based Learning Grey Wolf Optimizer (IOBL-GWO)".
        + This algorithm is heavily (4x - 6X slower than original) because of multiple times of calculating the fitness of agent in each population.

    Links:
        1. https://doi.org/10.1007/s12652-020-02153-1

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
    >>> model = GWO.IOBL_GWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Bansal, J. C., & Singh, S. (2021). A better exploration strategy in Grey Wolf Optimizer. Journal of Ambient Intelligence and Humanized Computing, 12(1), 1099-1118.
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
        self.is_parallelizable = False
        self.sort_flag = False

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
        for idx in range(0, self.pop_size):
            # Try explorative equation first
            r1, r2, r3, r4, r5 = self.generator.random(5)
            if r5 >= 0.5:  # Exploration around random wolf
                # Select random wolf from population
                jdx = self.generator.choice(list(set(range(self.pop_size)) - {idx}))
                x_rand = self.pop[jdx].solution
                pos_new = x_rand - r1 * np.abs(x_rand - 2 * r2 * self.pop[idx].solution)
            else:  # Exploration around alpha wolf
                # Calculate average position of all wolves
                x_avg = np.mean([agent.solution for agent in self.pop], axis=0)
                pos_new = (list_best[0].solution - x_avg) - r3 * (
                    self.problem.lb + r4 * (self.problem.ub - self.problem.lb)
                )
            # Apply boundary constraints
            pos_new = self.correct_solution(pos_new)
            tar_new = self.get_target(pos_new)
            if self.compare_target(tar_new, self.pop[idx].target, self.problem.minmax):
                # If new position is better, update the agent
                agent = self.generate_empty_agent(pos_new)
                agent.target = tar_new
                self.pop[idx] = agent
            else:
                # If not better, use original GWO update
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
                pos_new = (X1 + X2 + X3) / 3.0
                pos_new = self.correct_solution(pos_new)
                tar_new = self.get_target(pos_new)
                # Create new agent with updated position
                if self.compare_target(
                    tar_new, self.pop[idx].target, self.problem.minmax
                ):
                    agent = self.generate_empty_agent(pos_new)
                    agent.target = tar_new
                    self.pop[idx] = agent

        # Apply Opposition-Based Learning (OBL) for leading wolves
        _, list_best, _ = self.get_special_agents(
            self.pop, n_best=3, minmax=self.problem.minmax
        )
        pop_sorted, indices = self.get_sorted_indices_population(
            self.pop, minmax=self.problem.minmax
        )
        obl_alpha = self.generate_agent(
            solution=self.problem.lb + self.problem.ub - pop_sorted[0].solution
        )
        obl_beta = self.generate_agent(
            solution=self.problem.lb + self.problem.ub - pop_sorted[1].solution
        )
        obl_delta = self.generate_agent(
            solution=self.problem.lb + self.problem.ub - pop_sorted[2].solution
        )
        obl_pop = [obl_alpha, obl_beta, obl_delta]

        # Replace worst 3 wolves with opposite solutions if they are better
        for idx in range(0, 3):
            if self.compare_target(
                obl_pop[idx].target,
                self.pop[indices[-3 + idx]].target,
                self.problem.minmax,
            ):
                self.pop[idx] = obl_pop[idx]

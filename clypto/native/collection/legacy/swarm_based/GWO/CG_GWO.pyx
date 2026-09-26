#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class CG_GWO(LegacyOptimizer):
    """
    The original version of: Cauchy‑Gaussian mutation and improved search strategy GWO (CG‑GWO)

    Notes:
        + This algorithm can't be parallelized because of the 'single' update mode.
        + Meaning that the updating of the pack is based on order and sequence of the wolves.

    Links:
        1. https://doi.org/10.1038/s41598-022-23713-9

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.CG_GWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Li, K., Li, S., Huang, Z. et al. Grey Wolf Optimization algorithm based on Cauchy-Gaussian mutation and improved search strategy. Sci Rep 12, 18961 (2022).
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

    def cauchy_gaussian_mutation(self, best, leader, epoch):
        """
        Apply Cauchy-Gaussian mutation to leader wolves
        """
        # Calculate dynamic parameters (equations 11 and 12)
        eps2 = (epoch / self.epoch) ** 2
        eps1 = 1 - eps2

        # Calculate sigma (equation 9)
        if abs(best.target.fitness) > 1e-10:
            sigma = np.exp(
                (leader.target.fitness - best.target.fitness) / abs(best.target.fitness)
            )
        else:
            sigma = 1.0
        # Generate Cauchy and Gaussian random variables
        c_rand = self.generator.standard_cauchy(size=self.problem.n_dims) * sigma**2 + 0
        g_rand = self.generator.normal(loc=0, scale=sigma**2, size=self.problem.n_dims)

        # Apply mutation (equation 8)
        mutated_pos = leader.solution * (1 + eps1 * c_rand + eps2 * g_rand)
        return mutated_pos

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 - 2.0 * epoch / self.epoch
        _, list_best, _ = self._get_special_agents(
            self.pop, n_best=3, sense=self.problem.sense
        )

        # Apply Cauchy-Gaussian mutation to leaders
        alpha_pos = self.cauchy_gaussian_mutation(list_best[0], list_best[0], epoch)
        alpha_pos = self._correct_solution(alpha_pos)
        alpha = self._generate_agent(solution=alpha_pos)

        beta_pos = self.cauchy_gaussian_mutation(list_best[0], list_best[1], epoch)
        beta_pos = self._correct_solution(beta_pos)
        beta = self._generate_agent(solution=beta_pos)

        delta_pos = self.cauchy_gaussian_mutation(list_best[0], list_best[2], epoch)
        delta_pos = self._correct_solution(delta_pos)
        delta = self._generate_agent(solution=delta_pos)

        leaders = [alpha, beta, delta]
        # Greedy selection mechanism
        list_best = self._greedy_selection_population(
            list_best, leaders, self.problem.sense
        )

        pop_new = []
        for idx in range(0, self.pop_size):
            ## Apply improved search strategy

            # Apply improved search strategy (equation 13)
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
                    self.problem.bounds.low + r4 * (self.problem.bounds.up - self.problem.bounds.low)
                )
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_agent(pos_new)

            if self._compare_target(
                self.pop[idx].target, agent.target, self.problem.sense
            ):
                # If new position is not better, use original GWO update
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
                pos_new = self._correct_solution(pos_new)
                agent = self._generate_agent(pos_new)

            if self._compare_target(
                agent.target, self.pop[idx].target, self.problem.sense
            ):
                # If new position is better, update the agent
                self.pop[idx] = agent

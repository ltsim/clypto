#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class CG_GWO(cy.Optimizer):
    """
    The original version of: Cauchy‑Gaussian mutation and improved search strategy GWO (CG‑GWO)

    Notes:
        + This algorithm can't be parallelized because of the 'single' update mode.
        + Meaning that the updating of the pack is based on order and sequence of the wolves.

    Links:
        1. https://doi.org/10.1038/s41598-022-23713-9

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

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
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    def cauchy_gaussian_mutation(self, best, leader, epoch):
        """
        Apply Cauchy-Gaussian mutation to leader wolves
        """
        # Calculate dynamic parameters (equations 11 and 12)
        eps2 = (epoch / self.epoch) ** 2
        eps1 = 1 - eps2

        # Calculate sigma (equation 9)
        if abs(best.fitness) > 1e-10:
            sigma = np.exp(
                (leader.fitness - best.fitness) / abs(best.fitness)
            )
        else:
            sigma = 1.0
        # Generate Cauchy and Gaussian random variables
        c_rand = self.generator.standard_cauchy(size=self.problem.n_dims) * sigma**2 + 0
        g_rand = self.generator.normal(loc=0, scale=sigma**2, size=self.problem.n_dims)

        # Apply mutation (equation 8)
        mutated_pos = leader.solution * (1 + eps1 * c_rand + eps2 * g_rand)
        return mutated_pos

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # linearly decreased from 2 to 0
        a = 2 - 2.0 * epoch / self.epoch
        ranked = self.population.sort()
        list_best = [cy.duplicate_agent(agent) for agent in ranked[:3]]

        # Apply Cauchy-Gaussian mutation to leaders
        alpha_pos = self.cauchy_gaussian_mutation(list_best[0], list_best[0], epoch)
        alpha_pos = cy.correct_solution(self.problem, alpha_pos)
        alpha = self.population.generate_agent(solution=alpha_pos)

        beta_pos = self.cauchy_gaussian_mutation(list_best[0], list_best[1], epoch)
        beta_pos = cy.correct_solution(self.problem, beta_pos)
        beta = self.population.generate_agent(solution=beta_pos)

        delta_pos = self.cauchy_gaussian_mutation(list_best[0], list_best[2], epoch)
        delta_pos = cy.correct_solution(self.problem, delta_pos)
        delta = self.population.generate_agent(solution=delta_pos)

        leaders = [alpha, beta, delta]
        # Greedy selection mechanism
        list_best = cy.greedy_agents(list_best, leaders, self.problem.sense)

        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            ## Apply improved search strategy

            # Apply improved search strategy (equation 13)
            r1, r2, r3, r4, r5 = self.generator.random(5)
            if r5 >= 0.5:  # Exploration around random wolf
                # Select random wolf from population
                jdx = self.generator.choice(list(set(range(pop_size)) - {idx}))
                x_rand = self.population[jdx].solution
                x = x_rand - r1 * np.abs(x_rand - 2 * r2 * agent.solution)
            else:  # Exploration around alpha wolf
                # Calculate average position of all wolves
                x_avg = np.mean([child.solution for child in self.population], axis=0)
                x = (list_best[0].solution - x_avg) - r3 * (
                    self.problem.bounds.low + r4 * (self.problem.bounds.up - self.problem.bounds.low)
                )
            x = cy.correct_solution(self.problem, x)
            child = self.population.generate_agent(x)

            if cy.is_better(agent, child, self.problem.sense):
                # If new position is not better, use original GWO update
                A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C1 = 2 * self.generator.random(self.problem.n_dims)
                C2 = 2 * self.generator.random(self.problem.n_dims)
                C3 = 2 * self.generator.random(self.problem.n_dims)
                X1 = list_best[0].solution - A1 * np.abs(
                    C1 * list_best[0].solution - agent.solution
                )
                X2 = list_best[1].solution - A2 * np.abs(
                    C2 * list_best[1].solution - agent.solution
                )
                X3 = list_best[2].solution - A3 * np.abs(
                    C3 * list_best[2].solution - agent.solution
                )
                x = (X1 + X2 + X3) / 3.0
                x = cy.correct_solution(self.problem, x)
                child = self.population.generate_agent(x)

            if cy.is_better(child, agent, self.problem.sense):
                # If new position is better, update the child
                self.population[idx] = child

#!/usr/bin/env python
# Created by "Thieu" at 22:08, 28/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalSOO(cy.Optimizer):
    """
    The original version of: Stellar Oscillation Optimizer (SOO)

    Notes:
        + The MATLAB code in the link below by the author is completely different from the pseudocode in
        the original paper. I don’t understand how the author could write such an incorrect implementation
        and still obtain good results. There are only two possibilities: either the author fabricated the
        results in the paper, or the paper itself is fundamentally flawed.

        + For example, you can see equation number 8 — it involves taking the average of two new positions.
        However, in the code, it is incorrectly implemented as position 1 plus half of position 2.
        Even more concerning is that the pseudocode in the paper is completely different from the actual code.
        The MATLAB coding quality is really poor. In the pseudocode, it states that the fitness should be
        calculated and the global best as well as the top 3 best should be updated — yet this is entirely missing in the code.

        + Therefore, I do not recommend users to use this algorithm, as it lacks integrity between the
        results in the paper and the actual experimental implementation.

    Links:
        1. https://mathworks.com/matlabcentral/fileexchange/161921-stellar-oscillation-optimizer-meta-heuristic-optimimization

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.physics_based import SOO    >>> import numpy as np
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
    >>> model = SOO.OriginalSOO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Rodan, A., Al-Tamimi, A. K., Al-Alnemer, L., & Mirjalili, S. (2025).
    Stellar oscillation optimizer: a nature-inspired metaheuristic optimization algorithm. Cluster Computing, 28(6), 362.
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

    def initialize_variables(self):
        self.initial_period = 3

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Update period and angular frequency
        caf = 2 * np.pi / (self.initial_period + 0.001 * epoch)

        # Update scaling factor
        scaler = 2 * (1.0 - epoch / self.epoch)

        # Update positions of star oscillators
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx in range(pop_size):
            r1 = self.generator.random(size=self.problem.n_dims)
            r2 = self.generator.random(size=self.problem.n_dims)
            r3 = self.generator.random(size=self.problem.n_dims)

            # Calculate oscillation positions
            osc1 = (
                    scaler
                    * (caf * r1 - 1)
                    * (
                            self.population[idx].solution
                            - np.abs(r1 * np.sin(r2) * np.abs(r3 * self.g_best.solution))
                    )
            )
            osc1_pos = self.g_best.solution - r1 * r3 * osc1
            osc2 = (
                    scaler
                    * (caf * r1 - 1)
                    * (
                            self.population[idx].solution
                            - np.abs(r1 * np.cos(r2) * np.abs(r3 * self.g_best.solution))
                    )
            )
            osc2_pos = self.g_best.solution - r2 * r3 * osc2
            x = r3 * (osc1_pos + osc2_pos) / 2
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            n_population.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

        # Get top 3 stars
        ranked = self.population.sort()
        best3 = [cy.duplicate_agent(agent) for agent in ranked[:3]]

        # Perform oscillatory movement update (a new batch: MEALPY reused the first one, so the batch
        # modes failed with twice pop_size candidates)
        n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            # Average of top star positions
            avg3 = np.mean([child.solution for child in best3], axis=0)

            # Select 3 random indices different from current
            r1, r2, r3 = self.generator.choice(
                list(set(range(pop_size)) - {idx}), size=3, replace=False
            )

            # Generate new position based on oscillatory movement
            rf = self.generator.random()
            x = avg3 + 0.5 * (
                    np.sin(rf * np.pi) * (self.population[r1].solution - self.population[r2].solution)
                    + np.cos((1 - rf) * np.pi)
                    * (self.population[r1].solution - self.population[r3].solution)
            )
            ## Probabilistic update
            x = np.where(
                self.generator.random(size=self.problem.n_dims) <= 0.5,
                x,
                agent.solution,
            )
            # Apply boundary constraints
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

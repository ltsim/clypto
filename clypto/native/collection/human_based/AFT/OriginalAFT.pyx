#!/usr/bin/env python
# Created by "Thieu" at 22:47, 15/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalAFT(cy.Optimizer):
    """
    The original version of: Ali baba and the Forty Thieves (AFT) optimizer

    Notes:
        + https://doi.org/10.1007/s00521-021-06392-x

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import AFT    >>> import numpy as np
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
    >>> model = AFT.OriginalAFT(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Braik, M., Ryalat, M. H., & Al-Zoubi, H. (2022). A novel meta-heuristic algorithm for solving
    numerical optimization problems: Ali Baba and the forty thieves. Neural Computing and Applications, 34(1), 409-455.
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

    def before_main_loop(self):
        # Initialize best positions (Marjaneh's astute plans)
        self.pop_best = self.population.copy()  # It is like local best positions like in PSO
        # self.population is population of alibaba ==> It will always update with new version no matter what

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Calculate AFT parameters
        # Perception potential - decreases over iterations
        Pp = 0.1 * np.log(2.75 * (epoch / self.epoch) ** 0.1)

        # Tracking distance - decreases over iterations
        Td = 2 * np.exp(-2 * (epoch / self.epoch) ** 2)

        # Generate random candidate followers indices
        random_followers = self.generator.integers(0, pop_size, size=pop_size)

        # Update positions for each thief
        for idx, agent in enumerate(self.population.toarray()):
            if self.generator.random() >= 0.5:
                # Thieves know where to search (TRUE case)
                if self.generator.random() > Pp:
                    # Case 1: Follow global best with tracking distance
                    direction = np.sign(self.generator.random() - 0.5)
                    movement = (
                            Td
                            * (self.pop_best[idx].solution - agent.solution)
                            * self.generator.random()
                            + Td
                            * (
                                    agent.solution
                                    - self.pop_best[random_followers[idx]].solution
                            )
                            * self.generator.random()
                    )
                    x = self.g_best.solution + movement * direction
                else:
                    # Case 3: Random exploration within tracking distance
                    x = self.problem.bounds.low + Td * (
                            self.problem.bounds.up - self.problem.bounds.low
                    ) * self.generator.random(self.problem.n_dims)
            else:
                # Thieves don't know where to search - opposite direction (Marjaneh's tricks)
                direction = np.sign(self.generator.random() - 0.5)
                movement = (
                        Td
                        * (self.pop_best[idx].solution - agent.solution)
                        * self.generator.random()
                        + Td
                        * (
                                agent.solution
                                - self.pop_best[random_followers[idx]].solution
                        )
                        * self.generator.random()
                )
                x = self.g_best.solution - movement * direction
            # Clip to bounds
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            self.population[idx] = child
            # self.pop_baba[idx] = child
            if self.mode == "sequential":
                # self.pop_baba[idx].evaluate(self.problem)
                self.population[idx].evaluate(self.problem)
        if self.mode != "sequential":
            self.population = self.population.spawn(self.population.evaluate(self.population, self.mode))
            self.pop_best = cy.greedy_agents(self.pop_best, self.population, self.problem.sense)

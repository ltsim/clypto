#!/usr/bin/env python
# Created by "Thieu" at 21:58, 16/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class Matlab101GTO(cy.Optimizer):
    """
    The conversion of Matlab code (version 1.0.1 - 29/11/2022) to Python code of: Giant Trevally Optimizer (GTO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/121358-giant-trevally-optimizer-gto
        2. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9955508

    Notes:
        1. This algorithm costs a huge amount of computational resources in each epoch.
        Therefore, be careful when using the maximum number of generations as a stopping condition.
        2. Other algorithms update around K*pop_size times in each epoch, this algorithm updates around 2*pop_size^2 + pop_size times
        3. This version is used by the authors to compared with other algorithms in their paper.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import GTO    >>> import numpy as np
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
    >>> model = GTO.Matlab101GTO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Sadeeq, H. T., & Abdulazeez, A. M. (2022). Giant Trevally Optimizer (GTO): A Novel Metaheuristic
    Algorithm for Global Optimization and Challenging Engineering Problems. IEEE Access, 10, 121615-121640.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Step 1: Extensive Search
        for idx in range(0, pop_size):
            pop_new = []
            for jdx in range(0, pop_size):
                if idx == jdx:
                    continue
                # foraging movement patterns of giant trevallies are simulated using Eq.(4)
                pos_new = self.g_best.solution * self.generator.random() + (
                        (self.problem.bounds.up - self.problem.bounds.low) * self.generator.random()
                        + self.problem.bounds.low
                ) * cy.levy_flight(self.generator, beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
                pos_new = self.population.correct_solution(pos_new)
                agent = self.population.create_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].evaluate(self.problem)
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population[idx] = cy.sort_agents(pop_new + [self.population[idx]], self.problem.sense)[0].copy()
        ranked = self.population.sort()
        self.g_best = ranked[0]

        # Step 2: Choosing Area
        pos_list = np.array([agent.solution for agent in self.population])
        pos_m = np.mean(pos_list, axis=0)
        A = 0.4
        pop_new = []
        for idx in range(0, pop_size):
            # In the choosing area step, giant trevallies identify and select the best area in terms of
            # the amount of food (seabirds) within the selected search space where they can hunt for prey.
            r3 = self.generator.random()
            pos_new = (
                    self.g_best.solution * A * r3 + pos_m - self.population[idx].solution * r3
            )  # Eq. 7
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        ranked = self.population.sort()
        self.g_best = ranked[0]

        # Step 3: Attacking
        H = self.generator.random() * 2.0 * (1.0 - epoch / self.epoch)  # Eq.(15)
        for idx in range(0, pop_size):
            pop_new = []
            for jdx in range(0, pop_size):
                if idx == jdx:
                    continue
                # the distance between the prey and the attacker, and can be calculated using (12):
                dist = np.sum(np.abs(self.g_best.solution - self.population[idx].solution))
                theta2 = (360 - 0) * self.generator.random() + 0
                theta1 = 1.3296 * np.sin(
                    np.radians(theta2)
                )  # calculate theta_1 using (10)
                # visual distortion indicates the apparent height of the bird, which is always seen
                # to be higher than its actual height due to the refraction of the light.
                VD = np.sin(np.radians(theta1)) * dist  # Eq. 11
                # the behavior of giant trevally when chasing and jumping out of the water is mathematically simulated using (13)
                pos_new = (
                        self.population[idx].solution
                        * np.sin(np.radians(theta2))
                        * self.population[idx].fitness
                        + VD
                        + H
                )
                pos_new = self.population.correct_solution(pos_new)
                agent = self.population.create_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].evaluate(self.problem)
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population[idx] = cy.sort_agents(pop_new + [self.population[idx]], self.problem.sense)[0].copy()

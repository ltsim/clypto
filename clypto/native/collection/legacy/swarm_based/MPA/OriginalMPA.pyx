#!/usr/bin/env python
# Created by "Thieu" at 17:28, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalMPA(cy.Optimizer):
    """
    The developed version: Marine Predators Algorithm (MPA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0957417420302025
        2. https://www.mathworks.com/matlabcentral/fileexchange/74578-marine-predators-algorithm-mpa

    Notes:
        1. To use the original paper, set the training mode = "swarm"
        2. They update the whole population at the same time before update the fitness
        3. Two variables that they consider it as constants which are FADS = 0.2 and P = 0.5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import MPA    >>> import numpy as np
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
    >>> model = MPA.OriginalMPA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
    Marine Predators Algorithm: A nature-inspired metaheuristic. Expert systems with applications, 152, 113377.
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
        self.FADS = 0.2
        self.P = 0.5

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        CF = (1 - epoch / self.epoch) ** (2 * epoch / self.epoch)
        RL = cy.levy_flight(self.generator, beta=1.5, multiplier=0.05, size=(pop_size, self.problem.n_dims), case=-1)
        RB = self.generator.standard_normal((pop_size, self.problem.n_dims))
        per1 = self.generator.permutation(pop_size)
        per2 = self.generator.permutation(pop_size)
        pop_new = []
        for idx in range(0, pop_size):
            R = self.generator.random(self.problem.n_dims)
            if epoch < self.epoch / 3:  # Phase 1 (Eq.12)
                step_size = RB[idx] * (
                        self.g_best.solution - RB[idx] * self.population[idx].solution
                )
                pos_new = self.population[idx].solution + self.P * R * step_size
            elif self.epoch / 3 < epoch < 2 * self.epoch / 3:  # Phase 2 (Eqs. 13 & 14)
                if idx > pop_size / 2:
                    step_size = RB[idx] * (
                            RB[idx] * self.g_best.solution - self.population[idx].solution
                    )
                    pos_new = self.g_best.solution + self.P * CF * step_size
                else:
                    step_size = RL[idx] * (
                            self.g_best.solution - RL[idx] * self.population[idx].solution
                    )
                    pos_new = self.population[idx].solution + self.P * R * step_size
            else:  # Phase 3 (Eq. 15)
                step_size = RL[idx] * (
                        RL[idx] * self.g_best.solution - self.population[idx].solution
                )
                pos_new = self.g_best.solution + self.P * CF * step_size
            pos_new = self.population.correct_solution(pos_new)
            if self.generator.random() < self.FADS:
                u = np.where(
                    self.generator.random(self.problem.n_dims) < self.FADS, 1, 0
                )
                pos_new = (
                        pos_new
                        + CF
                        * (
                                self.problem.bounds.low
                                + self.generator.random(self.problem.n_dims)
                                * (self.problem.bounds.up - self.problem.bounds.low)
                        )
                        * u
                )
            else:
                r = self.generator.random()
                step_size = (self.FADS * (1 - r) + r) * (
                        self.population[per1[idx]].solution - self.population[per2[idx]].solution
                )
                pos_new = pos_new + step_size
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent, self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

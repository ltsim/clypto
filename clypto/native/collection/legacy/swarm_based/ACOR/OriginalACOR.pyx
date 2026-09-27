#!/usr/bin/env python
# Created by "Thieu" at 14:14, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalACOR(cy.Optimizer):
    """
    The original version of: Ant Colony Optimization Continuous (ACOR)

    Notes:
        + Use Gaussian Distribution (np.random.normal() function) instead of random number (np.random.rand())
        + Amend solution when they went out of space

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + sample_count (int): [2, 10000], Number of Newly Generated Samples, default = 25
        + intent_factor (float): [0.2, 1.0], Intensification Factor (Selection Pressure), (q in the paper), default = 0.5
        + zeta (float): [1, 2, 3], Deviation-Distance Ratio, default = 1.0

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import ACOR    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "sense": "min",
    >>> }
    >>>
    >>> model = ACOR.OriginalACOR(epoch=1000, pop_size=50, sample_count = 25, intent_factor = 0.5, zeta = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Socha, K. and Dorigo, M., 2008. Ant colony optimization for continuous domains.
    European journal of operational research, 185(3), pp.1155-1173.
    """

    cdef public double intent_factor
    cdef public int sample_count
    cdef public double zeta

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            sample_count: int = 25,
            intent_factor: float = 0.5,
            zeta: float = 1.0,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            sample_count: Number of Newly Generated Samples, default = 25
            intent_factor: Intensification Factor (Selection Pressure) (q in the paper), default = 0.5
            zeta: Deviation-Distance Ratio, default = 1.0
        """
        super().__init__(parameters=["epoch", "pop_size", "sample_count", "intent_factor", "zeta"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.sample_count = cy.validator(int, sample_count, [2, 10000], "sample_count")
        self.intent_factor = cy.validator(float, intent_factor, (0, 1.0), "intent_factor")
        self.zeta = cy.validator(float, zeta, (0, 5), "zeta")

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Calculate Selection Probabilities
        pop_rank = np.array([idx for idx in range(1, pop_size + 1)])
        qn = self.intent_factor * pop_size
        matrix_w = (
                1 / (np.sqrt(2 * np.pi) * qn) * np.exp(-0.5 * ((pop_rank - 1) / qn) ** 2)
        )
        matrix_p = matrix_w / np.sum(matrix_w)  # Normalize to find the probability.
        # Means and Standard Deviations
        matrix_pos = np.array([agent.solution for agent in self.population])
        matrix_sigma = []
        for idx in range(0, pop_size):
            matrix_i = np.repeat(
                self.population[idx].solution.reshape((1, -1)), pop_size, axis=0
            )
            D = np.sum(np.abs(matrix_pos - matrix_i), axis=0)
            temp = self.zeta * D / (pop_size - 1)
            matrix_sigma.append(temp)
        matrix_sigma = np.array(matrix_sigma)

        # Generate Samples
        pop_new = []
        for idx in range(0, self.sample_count):
            child = np.zeros(self.problem.n_dims)
            for jdx in range(0, self.problem.n_dims):
                rdx = cy.roulette_wheel(self.generator, self.problem.sense, matrix_p)
                child[jdx] = (
                        self.population[rdx].solution[jdx]
                        + self.generator.normal() * matrix_sigma[rdx, jdx]
                )  # (1)
            pos_new = self.population.correct_solution(child)  # (2)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].evaluate(self.problem)
        pop_new = self.population.evaluate(pop_new, self.mode)
        self.population = cy.sort_agents(self.population + pop_new, self.problem.sense)[:pop_size]

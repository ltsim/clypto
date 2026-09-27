#!/usr/bin/env python
# Created by "Thieu" at 18:09, 13/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalRIME(cy.Optimizer):
    """
    The original version of: physical phenomenon of RIME-ice  (RIME)

    Links:
        1. https://doi.org/10.1016/j.neucom.2023.02.010
        2. https://www.mathworks.com/matlabcentral/fileexchange/124610-rime-a-physics-based-optimization

    Notes (parameters):
        1. sr (float): Soft-rime parameters, default=5.0
        2. The algorithm is straightforward and does not require any specialized knowledge or techniques.
        3. The algorithm may exhibit slow convergence and may not perform optimally.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.physics_based import RIME    >>> import numpy as np
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
    >>> model = RIME.OriginalRIME(epoch=1000, pop_size=50, sr = 5.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Su, H., Zhao, D., Heidari, A. A., Liu, L., Zhang, X., Mafarja, M., & Chen, H. (2023). RIME: A physics-based optimization. Neurocomputing.
    """

    cdef public double sr

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, sr: float = 5.0, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            sr (float): Soft-rime parameters, default=5.0
        """
        super().__init__(parameters=["epoch", "pop_size", "sr"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.sr = cy.validator(float, sr, (0.0, 100.0), "sr")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        rime_factor = (
                (self.generator.random() - 0.5)
                * 2
                * np.cos(np.pi * epoch / (self.epoch / 10))
                * (1 - np.round(epoch * self.sr / self.epoch) / self.sr)
        )
        ee = np.sqrt((epoch + 1) / self.epoch)
        fits = np.array([agent.fitness for agent in self.population]).reshape((1, -1))
        fits_norm = fits / np.linalg.norm(fits, axis=1, keepdims=True)
        LB = self.problem.bounds.low
        UB = self.problem.bounds.up
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            x = agent.solution.copy()
            for jdx in range(0, self.problem.n_dims):
                # Soft-rime search strategy
                if self.generator.random() < ee:
                    x[jdx] = self.g_best.solution[jdx] + rime_factor * (
                            LB[jdx] + self.generator.random() * (UB[jdx] - LB[jdx])
                    )
                # Hard-rime puncture mechanism
                if self.generator.random() < fits_norm[0, idx]:
                    x[jdx] = self.g_best.solution[jdx]
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)

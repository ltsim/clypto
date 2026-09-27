#!/usr/bin/env python
# Created by "Thieu" at 17:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.math_based.SCA.DevSCA cimport DevSCA


cdef class OriginalSCA(DevSCA):
    """
    The original version of: Sine Cosine Algorithm (SCA)

    Links:
        1. https://doi.org/10.1016/j.knosys.2015.12.022
        2. https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.math_based import SCA    >>> import numpy as np
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
    >>> model = SCA.OriginalSCA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2016. SCA: a sine cosine algorithm for solving optimization problems. Knowledge-based systems, 96, pp.120-133.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.population = cy.population(pop_size, range=[5, 10000], cls=cy.Population)
        self.sort_flag = False

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            # Eq 3.4, r1 decreases linearly from a to 0
            a = 2.0
            r1 = a * (1.0 - epoch / self.epoch)
            x = agent.solution.copy()
            for jdx in range(self.problem.n_dims):  # j-th dimension
                # Update r2, r3, and r4 for Eq. (3.3)
                r2 = 2 * np.pi * self.generator.uniform()
                r3 = 2 * self.generator.uniform()
                r4 = self.generator.uniform()
                # Eq. 3.3, 3.1 and 3.2
                if r4 < 0.5:
                    x[jdx] = x[jdx] + r1 * np.sin(r2) * np.abs(
                        r3 * self.g_best.solution[jdx] - x[jdx]
                    )
                else:
                    x[jdx] = x[jdx] + r1 * np.cos(r2) * np.abs(
                        r3 * self.g_best.solution[jdx] - x[jdx]
                    )
            # Check the bound
            x = cy.reset_solution(self.problem, self.generator, x)
            child = self.population.create_agent(x)
            n_population.append(child)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)

#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.physics_based.EO.OriginalEO cimport OriginalEO


cdef class AdaptiveEO(OriginalEO):
    """
    The original version of: Adaptive Equilibrium Optimization (AEO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2020.103836

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.physics_based import EO    >>> import numpy as np
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
    >>> model = EO.AdaptiveEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Wunnava, A., Naik, M.K., Panda, R., Jena, B. and Abraham, A., 2020. A novel interdependence based
    multilevel thresholding technique using adaptive equilibrium optimizer. Engineering Applications of
    Artificial Intelligence, 94, p.103836.
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
        self.sort_flag = False
        self.pop_len = int(self.population.size() / 3)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # ---------------- Memory saving-------------------  make equilibrium pool
        ranked = self.population.sort()
        c_eq_list = [cy.duplicate_agent(agent) for agent in ranked[:4]]
        c_pool = self.make_equilibrium_pool__(c_eq_list)
        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)
        ## Memory saving, Eq 20, 21
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            lamda = self.generator.uniform(0, 1, self.problem.n_dims)
            r = self.generator.uniform(0, 1, self.problem.n_dims)
            c_eq = c_pool[
                self.generator.integers(0, len(c_pool))
            ].solution  # random selection 1 of candidate from the pool
            f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 14
            r1 = self.generator.uniform()
            r2 = self.generator.uniform()
            gcp = 0.5 * r1 * np.ones(self.problem.n_dims) * (r2 >= self.GP)
            g0 = gcp * (c_eq - lamda * agent.solution)
            g = g0 * f
            fit_average = np.mean([item.fitness for item in self.population])  # Eq. 19
            x = (
                    c_eq
                    + (agent.solution - c_eq) * f
                    + (g * self.V / lamda) * (1.0 - f)
            )  # Eq. 9
            if cy.better_fitness(agent.fitness, fit_average, self.problem.sense):
                x = np.multiply(
                    x, (0.5 + self.generator.uniform(0, 1, self.problem.n_dims))
                )
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

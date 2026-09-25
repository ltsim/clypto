#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.legacy.physics_based.EO.OriginalEO cimport OriginalEO


cdef class AdaptiveEO(OriginalEO):
    """
    The original version of: Adaptive Equilibrium Optimization (AEO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2020.103836

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import EO    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = EO.AdaptiveEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

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
        self.pop_len = int(self.pop_size / 3)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # ---------------- Memory saving-------------------  make equilibrium pool
        _, c_eq_list, _ = self.get_special_agents(
            self.pop, n_best=4, minmax=self.problem.minmax
        )
        c_pool = self.make_equilibrium_pool__(c_eq_list)
        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)
        ## Memory saving, Eq 20, 21
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            lamda = self.generator.uniform(0, 1, self.problem.n_dims)
            r = self.generator.uniform(0, 1, self.problem.n_dims)
            c_eq = c_pool[
                self.generator.integers(0, len(c_pool))
            ].solution  # random selection 1 of candidate from the pool
            f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 14
            r1 = self.generator.uniform()
            r2 = self.generator.uniform()
            gcp = 0.5 * r1 * np.ones(self.problem.n_dims) * (r2 >= self.GP)
            g0 = gcp * (c_eq - lamda * self.pop[idx].solution)
            g = g0 * f
            fit_average = np.mean([item.target.fitness for item in self.pop])  # Eq. 19
            pos_new = (
                    c_eq
                    + (self.pop[idx].solution - c_eq) * f
                    + (g * self.V / lamda) * (1.0 - f)
            )  # Eq. 9
            if self.compare_fitness(
                    self.pop[idx].target.fitness, fit_average, self.problem.minmax
            ):
                pos_new = np.multiply(
                    pos_new, (0.5 + self.generator.uniform(0, 1, self.problem.n_dims))
                )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )

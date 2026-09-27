#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalEO(cy.Optimizer):
    """
    The original version of: Equilibrium Optimizer (EO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2019.105190
        2. https://www.mathworks.com/matlabcentral/fileexchange/73352-equilibrium-optimizer-eo

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.physics_based import EO    >>> import numpy as np
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
    >>> model = EO.OriginalEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Faramarzi, A., Heidarinejad, M., Stephens, B. and Mirjalili, S., 2020. Equilibrium optimizer: A novel
    optimization algorithm. Knowledge-Based Systems, 191, p.105190.
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
        ## Fixed parameter proposed by authors
        self.V = 1
        self.a1 = 2
        self.a2 = 1
        self.GP = 0.5

    def make_equilibrium_pool__(self, list_equilibrium=None):
        pos_list = [agent.solution for agent in list_equilibrium]
        pos_mean = np.mean(pos_list, axis=0)
        pos_mean = self.population.correct_solution(pos_mean)
        agent = self.population.generate_agent(pos_mean)
        list_equilibrium.append(agent)
        return list_equilibrium

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # ---------------- Memory saving-------------------  make equilibrium pool
        ranked = self.population.sort()
        c_eq_list = [agent.copy() for agent in ranked[:4]]
        c_pool = self.make_equilibrium_pool__(c_eq_list)
        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)
        pop_new = []
        for idx in range(0, pop_size):
            lamda = self.generator.uniform(
                0, 1, self.problem.n_dims
            )  # lambda in Eq. 11
            r = self.generator.uniform(0, 1, self.problem.n_dims)  # r in Eq. 11
            c_eq = c_pool[
                self.generator.integers(0, len(c_pool))
            ].solution  # random selection 1 of candidate from the pool
            f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 11
            r1 = self.generator.uniform()
            r2 = self.generator.uniform()  # r1, r2 in Eq. 15
            gcp = 0.5 * r1 * np.ones(self.problem.n_dims) * (r2 >= self.GP)  # Eq. 15
            g0 = gcp * (c_eq - lamda * self.population[idx].solution)  # Eq. 14
            g = g0 * f  # Eq. 13
            pos_new = (
                    c_eq
                    + (self.population[idx].solution - c_eq) * f
                    + (g * self.V / lamda) * (1.0 - f)
            )  # Eq. 16
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

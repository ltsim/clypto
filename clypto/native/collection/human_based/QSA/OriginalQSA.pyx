#!/usr/bin/env python
# Created by "Thieu" at 10:21, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.native.collection.human_based.QSA.DevQSA cimport DevQSA
cimport clypto.core as cy


cdef class OriginalQSA(DevQSA):
    """
    The original version of: Queuing Search Algorithm (QSA)

    Links:
       1. https://www.sciencedirect.com/science/article/abs/pii/S0307904X18302890

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import QSA    >>> import numpy as np
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
    >>> model = QSA.OriginalQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhang, J., Xiao, M., Gao, L. and Pan, Q., 2018. Queuing search algorithm: A novel metaheuristic algorithm
    for solving engineering optimization problems. Applied Mathematical Modelling, 63, pp.464-490.
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
        self.sort_flag = True

    def update_business_3__(self, pop, g_best):
        pop_size = self.population.size()
        pr = [idx / pop_size for idx in range(1, pop_size + 1)]
        pop_new = []
        for idx in range(pop_size):
            x = pop[idx].solution.copy()
            for jdx in range(self.problem.n_dims):
                if self.generator.random() > pr[idx]:
                    i1, i2 = self.generator.choice(pop_size, 2, replace=False)
                    e = self.generator.exponential(0.5)
                    X1 = pop[i1].solution
                    X2 = pop[i2].solution
                    x[jdx] = X1[jdx] + e * (X2[jdx] - pop[idx].solution[jdx])
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop_new.append(agent)
        pop_new = cy.greedy_agents(pop, self.population.evaluate(pop_new, self.mode), self.problem.sense, self.mode)
        return pop_new

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.population, epoch)
        pop = self.update_business_2__(pop)
        self.population = self.population.spawn(self.update_business_3__(pop, self.g_best))

#!/usr/bin/env python
# Created by "Thieu" at 11:10, 15/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalBMO(cy.Optimizer):
    """
    The original version: Barnacles Mating Optimizer (BMO)

    Links:
        1. https://ieeexplore.ieee.org/document/8441097

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pl (int): [1, pop_size - 1], barnacle’s threshold

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.bio_based import BMO    >>> import numpy as np
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
    >>> model = BMO.OriginalBMO(epoch=1000, pop_size=50, pl = 4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2018. Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm
    for global optimisation problems. International journal of bio-inspired computation, 12(1), pp.1-22.
    """

    cdef public int pl

    def __init__(self, epoch=10000, pop_size=100, pl=5, **kwargs):
        super().__init__(parameters=["epoch", "pop_size", "pl"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.pl = cy.validator(int, pl, [1, self.population.size() - 1], "pl")

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        k1 = self.generator.permutation(pop_size)
        k2 = self.generator.permutation(pop_size)
        temp = np.abs(k1 - k2)
        pop_new = []
        for idx in range(0, pop_size):
            if temp[idx] <= self.pl:
                p = self.generator.uniform(0, 1)
                pos_new = (
                        p * self.population[k1[idx]].solution
                        + (1 - p) * self.population[k2[idx]].solution
                )
            else:
                pos_new = self.generator.uniform(0, 1) * self.population[k2[idx]].solution
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].evaluate(self.problem)
        self.population = self.population.evaluate(pop_new, self.mode)

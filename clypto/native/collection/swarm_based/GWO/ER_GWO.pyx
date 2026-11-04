#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class ER_GWO(cy.Optimizer):
    """
    The original version of: Efficient and Robust Grey Wolf Optimizer (ER-GWO)

    Notes:
        + Slow convergence speed due to the (miu_factor)^(iteration) ==> Big number
        + Three more parameters than original GWO, increase the complexity of the algorithm.

    Links:
        1. https://doi.org/10.1007/s00500-019-03939-y

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.ER_GWO(epoch=1000, pop_size=50, a_initial=2.0, a_final=0.0, miu_factor=1.0001)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Long, W., Cai, S., Jiao, J. et al. An efficient and robust grey wolf optimizer algorithm for large-scale numerical optimization. Soft Comput 24, 997–1026 (2020).
    """

    cdef public double a_final
    cdef public double a_initial
    cdef public double miu_factor

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        a_initial: float = 2.0,
        a_final: float = 0.0,
        miu_factor: float = 1.0001,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a_initial (float): initial value of coefficient a, default = 2.0
            a_final (float): final value of coefficient a, default = 0.0
            miu_factor (float): nonlinear coefficient for equation (8), default = 1.0001
        """
        super().__init__(parameters=["epoch", "pop_size", "a_initial", "a_final", "miu_factor"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.a_initial = cy.validator(float, a_initial, [0.0, 10.0], "a_initial")
        self.a_final = cy.validator(float, a_final, [0.0, self.a_initial], "a_final")
        self.miu_factor = cy.validator(float, miu_factor, [1.0001, 1.01], "miu_factor")  # Required in paper

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # linearly decreased from 2 to 0
        a = self.a_initial - (self.a_initial - self.a_final) * self.miu_factor**epoch
        ranked = self.population.sort()
        list_best = [cy.duplicate_agent(agent) for agent in ranked[:3]]
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(
                C1 * list_best[0].solution - agent.solution
            )
            X2 = list_best[1].solution - A2 * np.abs(
                C2 * list_best[1].solution - agent.solution
            )
            X3 = list_best[2].solution - A3 * np.abs(
                C3 * list_best[2].solution - agent.solution
            )
            dist1 = np.linalg.norm(X1)
            dist2 = np.linalg.norm(X2)
            dist3 = np.linalg.norm(X3)
            total = dist1 + dist2 + dist3
            if total == 0:
                # Avoid division by zero
                x = (X1 + X2 + X3) / 3.0
            else:
                # Normalize distances to avoid division by zero
                x = (X1 * dist1 + X2 * dist2 + X3 * dist3) / total
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)

#!/usr/bin/env python
# Created by "Thieu" at 11:38, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalSSO(cy.Optimizer):
    """
    The original version of: Salp Swarm Optimization (SSO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2017.07.002

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import SSO    >>> import numpy as np
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
    >>> model = SSO.OriginalSSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., Gandomi, A.H., Mirjalili, S.Z., Saremi, S., Faris, H. and Mirjalili, S.M., 2017.
    Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems. Advances in Engineering Software, 114, pp.163-191.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Eq. (3.2) in the paper
        c1 = 2 * np.exp(-((4 * epoch / self.epoch) ** 2))
        pop_new = []
        for idx in range(0, pop_size):
            if idx < pop_size / 2:
                c2_list = self.generator.random(self.problem.n_dims)
                c3_list = self.generator.random(self.problem.n_dims)
                pos_new_1 = self.g_best.solution + c1 * (
                        (self.problem.bounds.up - self.problem.bounds.low) * c2_list + self.problem.bounds.low
                )
                pos_new_2 = self.g_best.solution - c1 * (
                        (self.problem.bounds.up - self.problem.bounds.low) * c2_list + self.problem.bounds.low
                )
                pos_new = np.where(c3_list < 0.5, pos_new_1, pos_new_2)
            else:
                # Eq. (3.4) in the paper
                pos_new = (self.population[idx].solution + self.population[idx - 1].solution) / 2
            # Check if salps go out of the search space and bring it back then re-calculate its fitness value
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

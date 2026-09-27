#!/usr/bin/env python
# Created by "Thieu" at 15:53, 07/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalAO(cy.Optimizer):
    """
    The original version of: Aquila Optimization (AO)

    Links:
        1. https://doi.org/10.1016/j.cie.2021.107250

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import AO    >>> import numpy as np
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
    >>> model = AO.OriginalAO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A.A., Al-Qaness, M.A. and Gandomi, A.H., 2021.
    Aquila optimizer: a novel meta-heuristic optimization algorithm. Computers & Industrial Engineering, 157, p.107250.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        alpha = delta = 0.1
        g1 = 2 * self.generator.random() - 1  # Eq. 16
        g2 = 2 * (1 - epoch / self.epoch)  # Eq. 17
        dim_list = np.array(list(range(1, self.problem.n_dims + 1)))
        miu = 0.00565
        r0 = 10
        r = r0 + miu * dim_list
        w = 0.005
        phi0 = 3 * np.pi / 2
        phi = -w * dim_list + phi0
        x = r * np.sin(phi)  # Eq.(9)
        y = r * np.cos(phi)  # Eq.(10)
        QF = epoch ** (
                (2 * self.generator.random() - 1) / (1 - self.epoch) ** 2
        )  # Eq.(15)        Quality function
        pop_new = []
        for idx in range(0, pop_size):
            x_mean = np.mean(np.array([agent.solution for agent in self.population]), axis=0)
            levy_step = cy.levy_flight(self.generator, beta=1.5, multiplier=1.0, size=None, case=-1)
            if epoch <= (2 / 3) * self.epoch:  # Eq. 3, 4
                if self.generator.random() < 0.5:
                    pos_new = self.g_best.solution * (
                            1 - epoch / self.epoch
                    ) + self.generator.random() * (x_mean - self.g_best.solution)
                else:
                    idx = self.generator.choice(
                        list(set(range(0, pop_size)) - {idx})
                    )
                    pos_new = (
                            self.g_best.solution * levy_step
                            + self.population[idx].solution
                            + self.generator.random() * (y - x)
                    )  # Eq. 5
            else:
                if self.generator.random() < 0.5:
                    pos_new = (
                            alpha * (self.g_best.solution - x_mean)
                            - self.generator.random()
                            * (
                                    self.generator.random()
                                    * (self.problem.bounds.up - self.problem.bounds.low)
                                    + self.problem.bounds.low
                            )
                            * delta
                    )  # Eq. 13
                else:
                    pos_new = (
                            QF * self.g_best.solution
                            - (g2 * self.population[idx].solution * self.generator.random())
                            - g2 * levy_step
                            + self.generator.random() * g1
                    )  # Eq. 14
            pos_new = cy.correct_solution(self.problem, pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

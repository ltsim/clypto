#!/usr/bin/env python
# Created by "Thieu" at 15:53, 07/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class AAO(cy.Optimizer):
    """
    The original version of: Adaptive Aquila Optimizer (AAO)

    Links:
        1. https://doi.org/10.1016/j.rineng.2024.103261

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
    >>> model = AO.AAO(epoch=1000, pop_size=50, sharpness=10.0, sigmoid_midpoint=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Al-Selwi, S. M., Hassan, M. F., Abdulkadir, S. J., Ragab, M. G., Alqushaibi, A., & Sumiea, E. H. (2024).
    Smart grid stability prediction using adaptive aquila optimizer and ensemble stacked bilstm. Results in Engineering, 24, 103261.
    """

    cdef public double sharpness
    cdef public double sigmoid_midpoint

    def __init__(
            self, epoch=10000, pop_size=100, sharpness=10.0, sigmoid_midpoint=0.5, **kwargs
    ):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            sharpness (float): is a positive variable that controls the sharpness of the transition between exploration and exploitation, default is 10.0, Valid range: [0.1, 10000.0].
            sigmoid_midpoint (float): a variable that controls the midpoint of the sigmoid function as it determines when the transition should be applied, default is 0.5, Valid range: [0.0, 1.0].
        """
        super().__init__(parameters=["epoch", "pop_size", "sharpness", "sigmoid_midpoint"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.sharpness = cy.validator(float, sharpness, [0.1, 10000.0], "sharpness")
        self.sigmoid_midpoint = cy.validator(float, sigmoid_midpoint, [0.0, 1.0], "sigmoid_midpoint")

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
        cdef cy.Population n_population = cy.empty_snapshot(self.population)

        for idx, agent in enumerate(self.population.toarray()):
            x_mean = np.mean(np.array([child.solution for child in self.population]), axis=0)
            levy_step = cy.levy_flight(self.generator, beta=1.5, multiplier=1.0, size=None, case=-1)

            # Dynamically balance the exploration and exploitation phases
            sigmoid_factor = 1 / (
                    1
                    + np.exp(-self.sharpness * (epoch / self.epoch - self.sigmoid_midpoint))
            )

            if np.random.rand() <= (1 - sigmoid_factor):
                if self.generator.random() < 0.5:
                    pos_new = self.g_best.solution * (
                            1 - epoch / self.epoch
                    ) + self.generator.random() * (
                                      x_mean - self.g_best.solution
                              )  # Eq. (3) and Eq. (4)
                else:
                    idx = self.generator.choice(
                        list(set(range(0, pop_size)) - {idx})
                    )
                    pos_new = (
                            self.g_best.solution * levy_step
                            + agent.solution
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
                            - (g2 * agent.solution * self.generator.random())
                            - g2 * levy_step
                            + self.generator.random() * g1
                    )  # Eq. 14
            pos_new = cy.correct_solution(self.problem, pos_new)
            child = self.population.create_agent(pos_new)
            n_population.append(child)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)

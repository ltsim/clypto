#!/usr/bin/env python
# Created by "Thieu" at 15:53, 07/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class AAO(LegacyOptimizer):
    """
    The original version of: Adaptive Aquila Optimizer (AAO)

    Links:
        1. https://doi.org/10.1016/j.rineng.2024.103261

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import AO    >>> import numpy as np
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
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Al-Selwi, S. M., Hassan, M. F., Abdulkadir, S. J., Ragab, M. G., Alqushaibi, A., & Sumiea, E. H. (2024).
    Smart grid stability prediction using adaptive aquila optimizer and ensemble stacked bilstm. Results in Engineering, 24, 103261.
    """

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
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.sharpness = self.validator.check_float(
            "sharpness", sharpness, [0.1, 10000.0]
        )
        self.sigmoid_midpoint = self.validator.check_float(
            "sigmoid_midpoint", sigmoid_midpoint, [0.0, 1.0]
        )
        self._set_parameters(["epoch", "pop_size", "sharpness", "sigmoid_midpoint"])
        self.sort_flag = False

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
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

        for idx in range(0, self.pop_size):
            x_mean = np.mean(np.array([agent.solution for agent in self.pop]), axis=0)
            levy_step = self._get_levy_flight_step(beta=1.5, multiplier=1.0, case=-1)

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
                        list(set(range(0, self.pop_size)) - {idx})
                    )
                    pos_new = (
                            self.g_best.solution * levy_step
                            + self.pop[idx].solution
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
                            - (g2 * self.pop[idx].solution * self.generator.random())
                            - g2 * levy_step
                            + self.generator.random() * g1
                    )  # Eq. 14
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.pop[idx] = self._get_better_agent(
                    agent, self.pop[idx], self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            self.pop = self._greedy_selection_population(
                self.pop, pop_new, self.problem.sense
            )

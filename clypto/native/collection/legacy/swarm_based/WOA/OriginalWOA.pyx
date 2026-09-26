#!/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class OriginalWOA(_LegacyOptimizer):
    """
    The original version of: Whale Optimization Algorithm (WOA)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2016.01.008
        2. https://mathworks.com/matlabcentral/fileexchange/55667-the-whale-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import WOA    >>> import numpy as np
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
    >>> model = WOA.OriginalWOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm. Advances in engineering software, 95, pp.51-67.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        a = 2 - 2 * epoch / self.epoch  # linearly decreased from 2 to 0
        a2 = -1 + epoch * ((-1) / self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            r1, r2 = self.generator.random(size=2)
            A = a * (2 * r1 - a)
            C = 2 * r2
            b = 1
            l = (a2 - 1) * self.generator.random() + 1
            p = self.generator.random()

            pos_new = self.pop[idx].solution.copy()
            for jdx in range(0, self.problem.n_dims):
                if p < 0.5:
                    if np.abs(A) >= 1:
                        id_r = self.generator.choice(
                            list(set(range(0, self.pop_size)) - {idx})
                        )
                        D_X_rand = abs(
                            C * self.pop[id_r].solution[jdx]
                            - self.pop[idx].solution[jdx]
                        )
                        pos_new[jdx] = self.pop[id_r].solution[jdx] - A * D_X_rand
                    else:
                        D_Leader = abs(
                            C * self.g_best.solution[jdx] - self.pop[idx].solution[jdx]
                        )
                        pos_new[jdx] = self.g_best.solution[jdx] - A * D_Leader
                else:
                    D1 = abs(self.g_best.solution[jdx] - self.pop[idx].solution[jdx])
                    pos_new[jdx] = (
                            D1 * np.exp(b * l) * np.cos(l * 2 * np.pi)
                            + self.g_best.solution[jdx]
                    )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)

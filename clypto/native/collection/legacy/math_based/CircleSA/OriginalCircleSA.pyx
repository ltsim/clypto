#!/usr/bin/env python
# Created by "Thieu" at 17:38, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class OriginalCircleSA(LegacyOptimizer):
    """
    The original version of: Circle Search Algorithm (CircleSA)

    Links:
        1. https://doi.org/10.3390/math10101626
        2. https://www.mdpi.com/2227-7390/10/10/1626

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.math_based import CircleSA    >>> import numpy as np
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
    >>> model = CircleSA.OriginalCircleSA(epoch=1000, pop_size=50, c_factor=0.8)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Qais, M. H., Hasanien, H. M., Turky, R. A., Alghuwainem, S., Tostado-Véliz, M., & Jurado, F. (2022).
    Circle Search Algorithm: A Geometry-Based Metaheuristic Optimization Algorithm. Mathematics, 10(10), 1626.
    """

    def __init__(self, epoch=10000, pop_size=100, c_factor=0.8, **kwargs):
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c_factor = self.validator.check_float("c_factor", c_factor, (0, 1.0))
        self._set_parameters(["epoch", "pop_size", "c_factor"])
        self.sort_flag = False

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        a = np.pi - np.pi * (epoch / self.epoch) ** 2  # Eq. 8
        p = 1 - 0.9 * (epoch / self.epoch) ** 0.5
        threshold = self.c_factor * self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            w = a * self.generator.random() - a
            if epoch > threshold:
                x_new = self.g_best.solution + (
                        self.g_best.solution - self.pop[idx].solution
                ) * np.tan(w * self.generator.random())
            else:
                x_new = self.g_best.solution - (
                        self.g_best.solution - self.pop[idx].solution
                ) * np.tan(w * p)
            pos_new = self._correct_solution(x_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self._get_target(pos_new)
        self.pop = self._update_target_for_population(pop_new)

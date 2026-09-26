#!/usr/bin/env python
# Created by "Thieu" at 10:08, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class OriginalHC(_LegacyOptimizer):
    """
    The original version of: Hill Climbing (HC)

    Notes:
        + The number of neighbour solutions are equal to user defined
        + The step size to calculate neighbour group is randomized
        + HC is single-based solution, so the pop_size parameter is not matter in this algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + neighbour_size (int): [2, 1000], fixed parameter, sensitive exploitation parameter, Default: 50

    Examples
    ~~~~~~~~
    >>> from clypto.collection.math_based import HC    >>> import numpy as np
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
    >>> model = HC.OriginalHC(epoch=1000, pop_size=50, neighbour_size = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mitchell, M., Holland, J. and Forrest, S., 1993. When will a genetic algorithm
    outperform hill climbing. Advances in neural information processing systems, 6.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 2,
            neighbour_size: int = 50,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 2
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 50
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
        self.neighbour_size = self.validator.check_int(
            "neighbour_size", neighbour_size, [2, 1000]
        )
        self.set_parameters(["epoch", "pop_size", "neighbour_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        step_size = np.exp(-2 * epoch / self.epoch)
        pop_neighbours = []
        for idx in range(0, self.neighbour_size):
            pos_new = (
                    self.g_best.solution
                    + self.generator.uniform(self.problem.lb, self.problem.ub) * step_size
            )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_neighbours.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_neighbours[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_neighbours)

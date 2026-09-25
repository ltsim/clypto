#!/usr/bin/env python
# Created by "Thieu" at 10:08, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalHC(LegacyNativeOptimizer):
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

    cdef public int neighbour_size

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 2,
        neighbour_size: int = 50,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 2
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 50
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "neighbour_size"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [2, 10000], "pop_size")
        self.neighbour_size = cy.validator(int, neighbour_size, [2, 1000], "neighbour_size")

    cdef void evolve(self, int epoch):
        cdef Py_ssize_t k = self.neighbour_size, d = self.pop.d
        step_size = np.exp(-2 * epoch / self.epoch)
        pos = np.array(self.g_best_x()) + self.generator.uniform(self.problem.lb, self.problem.ub, (k, d)) * step_size
        self.pop = self.new_population(self.correct_solution(pos))

#!/usr/bin/env python
# Created by "Thieu" at 22:47, 15/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalAFT(LegacyNativeOptimizer):
    """
    The original version of: Ali baba and the Forty Thieves (AFT) optimizer

    Notes:
        + https://doi.org/10.1007/s00521-021-06392-x

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import AFT    >>> import numpy as np
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
    >>> model = AFT.OriginalAFT(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Braik, M., Ryalat, M. H., & Al-Zoubi, H. (2022). A novel meta-heuristic algorithm for solving
    numerical optimization problems: Ali Baba and the forty thieves. Neural Computing and Applications, 34(1), 409-455.
    """

    cdef public object pop_best

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void before_main_loop(self):
        self.pop_best = self.pop.take(np.arange(self.pop.n))  # It is like local best positions like in PSO

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        Pb = self.pop_best.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        Pp = 0.1 * np.log(2.75 * (epoch_c / self.epoch) ** 0.1)
        Td = 2 * np.exp(-2 * (epoch_c / self.epoch) ** 2)
        follower = rng.integers(0, n, size=n)
        direction = np.sign(rng.random((n, 1)) - 0.5)
        movement = Td * (Pb - X) * rng.random((n, 1)) + Td * (X - Pb[follower]) * rng.random((n, 1))
        around_best = rng.random(n) >= 0.5
        keep = rng.random(n) > Pp
        pos = np.where(around_best[:, None],
                       np.where(keep[:, None], g + movement * direction, lb + Td * (ub - lb) * rng.random((n, d))),
                       g - movement * direction)
        ops.replace(self, pos)
        ops.greedy(self, self.pop, dst=self.pop_best)  # local best positions

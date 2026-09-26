#!/usr/bin/env python
# Created by "Thieu" at 17:55, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalHBA(LegacyNativeOptimizer):
    """
    The original version of: Honey Badger Algorithm (HBA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0378475421002901
        2. https://www.mathworks.com/matlabcentral/fileexchange/98204-honey-badger-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import HBA    >>> import numpy as np
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
    >>> model = HBA.OriginalHBA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F. A., Houssein, E. H., Hussain, K., Mabrouk, M. S., & Al-Atabany, W. (2022). Honey Badger Algorithm: New metaheuristic
    algorithm for solving optimization problems. Mathematics and Computers in Simulation, 192, 84-110.
    """

    cdef public object beta
    cdef public object C

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

    cdef void initialize_variables(self):
        self.beta = 6  # the ability of HB to get the food  Eq.(4)
        self.C = 2  # constant in Eq. (3)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        tt = self.epoch
        alpha = self.C * np.exp(-tt / self.epoch)  # density factor in Eq. (3)
        # intensity in Eq. (2): distance to the best and to the next agent
        di = (np.linalg.norm(X - g, axis=1) + self.EPSILON) ** 2
        si = (np.linalg.norm(X - np.roll(X, -1, axis=0), axis=1) + self.EPSILON) ** 2
        I = (rng.random(n) * si / (4 * np.pi * di))[:, None]
        F = rng.choice([1, -1], size=(n, 1))
        R = rng.random((5, n, d))
        dif = g - X
        temp1 = g + F * self.beta * I * g + F * R[0] * alpha * dif * np.abs(np.cos(2 * np.pi * R[1]) * (1 - np.cos(2 * np.pi * R[2])))
        temp2 = g + F * R[4] * alpha * dif
        ops.step(self, np.where(R[3] < 0.5, temp1, temp2))

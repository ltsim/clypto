#!/usr/bin/env python
# Created by "Thieu" at 17:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevSCA(VectorizeOptimizer):
    """
    The developed version: Sine Cosine Algorithm (SCA)

    Notes:
        + The flow and few equations are changed
        + Third loops are removed faster computational time

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.math_based import SCA    >>> import numpy as np
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
    >>> model = SCA.DevSCA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

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
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def _evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t n = pop.n, d = pop.d
        # Eq 3.4, r1 decreases linearly from a to 0
        a = 2.0
        r1 = a * (1.0 - epoch / self.epoch)
        g = np.array(self.g_best_x())
        R = self.generator.random((n, 3, d))  # per agent: r2 and r3 draws, then the sin/cos mask
        # Update r2, r3, and r4 for Eq. (3.3), remove third loop here
        r2 = 2 * np.pi * R[:, 0]
        r3 = 2 * R[:, 1]
        X = pop.X
        # Eq. 3.3, 3.1 and 3.2
        pos_new1 = X + r1 * np.sin(r2) * np.abs(r3 * g - X)
        pos_new2 = X + r1 * np.cos(r2) * np.abs(r3 * g - X)
        pos_new = np.where(R[:, 2] < 0.5, pos_new1, pos_new2)
        # Check the bound
        cand.X[:] = self._correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

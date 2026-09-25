#!/usr/bin/env python
# Created by "Thieu" at 17:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.collection.math_based.SCA.DevSCA cimport DevSCA
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalSCA(DevSCA):
    """
    The original version of: Sine Cosine Algorithm (SCA)

    Links:
        1. https://doi.org/10.1016/j.knosys.2015.12.022
        2. https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.collection.math_based import SCA    >>> import numpy as np
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
    >>> model = SCA.OriginalSCA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2016. SCA: a sine cosine algorithm for solving optimization problems. Knowledge-based systems, 96, pp.120-133.
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
        super().__init__(epoch, pop_size, name=name, mode=mode)
        self.sort_flag = False

    cdef object amend_solution(self, object solution):
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(
            np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub),
            solution,
            rand_pos,
        )

    cdef void evolve(self, int epoch):
        # amend_solution draws random numbers after each agent, so the per-agent draw
        # order is kept; evaluation and selection are batched.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        # Eq 3.4, r1 decreases linearly from a to 0
        a = 2.0
        r1 = a * (1.0 - epoch / self.epoch)
        g = np.array(self.g_best_x())
        X, Xc = pop.X, cand.X
        for idx in range(n):
            pos_new = X[idx].copy()
            for jdx in range(d):  # j-th dimension
                # Update r2, r3, and r4 for Eq. (3.3)
                r2 = 2 * np.pi * self.generator.uniform()
                r3 = 2 * self.generator.uniform()
                r4 = self.generator.uniform()
                # Eq. 3.3, 3.1 and 3.2
                if r4 < 0.5:
                    pos_new[jdx] = pos_new[jdx] + r1 * np.sin(r2) * np.abs(r3 * g[jdx] - pos_new[jdx])
                else:
                    pos_new[jdx] = pos_new[jdx] + r1 * np.cos(r2) * np.abs(r3 * g[jdx] - pos_new[jdx])
            # Check the bound
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

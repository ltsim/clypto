#!/usr/bin/env python
# Created by "Thieu" at 17:48, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.collection.music_based.HS.DevHS cimport DevHS
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalHS(DevHS):
    """
    The original version of: Harmony Search (HS)

    Links:
        1. https://doi.org/10.1177/003754970107600201

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c_r (float): [0.1, 0.5], Harmony Memory Consideration Rate), default = 0.15
        + pa_r (float): [0.3, 0.8], Pitch Adjustment Rate, default=0.5

    Examples
    ~~~~~~~~
    >>> from clypto.collection.music_based import HS    >>> import numpy as np
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
    >>> model = HS.OriginalHS(epoch=1000, pop_size=50, c_r = 0.95, pa_r = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Geem, Z.W., Kim, J.H. and Loganathan, G.V., 2001. A new heuristic
    optimization algorithm: harmony search. simulation, 76(2), pp.60-68.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c_r: float = 0.95,
        pa_r: float = 0.05,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c_r (float): Harmony Memory Consideration Rate), default = 0.15
            pa_r (float): Pitch Adjustment Rate, default=0.5
        """
        super().__init__(epoch, pop_size, c_r, pa_r, name=name, mode=mode)

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, jdx, n = self.pop_size, d = pop.d
        lb, ub = self.problem.lb, self.problem.ub
        mean = (lb + ub) / 2
        std_dev = abs(ub - lb) / 6  # This assumes a range of +/- 3 standard deviations
        X, Xc = pop.X, cand.X
        for idx in range(n):
            pos_new = self.generator.uniform(lb, ub)
            for jdx in range(d):
                # Use Harmony Memory
                if self.generator.uniform() <= self.c_r:
                    random_index = self.generator.integers(0, self.pop_size)
                    pos_new[jdx] = X[random_index, jdx]
                # Pitch Adjustment
                if self.generator.uniform() <= self.pa_r:
                    delta = self.dyn_fw * self.generator.normal(mean, std_dev)  # Gaussian(Normal)
                    pos_new[jdx] = pos_new[jdx] + delta[jdx]
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        # Update Damp Fret Width
        self.dyn_fw = self.dyn_fw * self.fw_damp
        # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
        merged = pop.concat(cand)
        self.pop = merged.take(self.sorted_order(merged)[:self.pop_size])

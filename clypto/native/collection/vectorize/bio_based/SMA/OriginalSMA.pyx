#!/usr/bin/env python
# Created by "Thieu" at 20:22, 12/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.bio_based.SMA.DevSMA cimport DevSMA
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalSMA(DevSMA):
    """
    The original version of: Slime Mould Algorithm (SMA)

    Links:
        1. https://doi.org/10.1016/j.future.2020.03.055
        2. https://www.researchgate.net/publication/340431861_Slime_mould_algorithm_A_new_method_for_stochastic_optimization

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_t (float): (0, 1.0) -> better [0.01, 0.1], probability threshold (z in the paper)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.bio_based import SMA    >>> import numpy as np
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
    >>> model = SMA.OriginalSMA(epoch=1000, pop_size=50, p_t = 0.03)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Li, S., Chen, H., Wang, M., Heidari, A.A. and Mirjalili, S., 2020. Slime mould algorithm: A new method for
    stochastic optimization. Future Generation Computer Systems, 111, pp.300-323.
    """

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        p_t = 0.03,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): number of population size, default = 100
            p_t (float): probability threshold (z in the paper), default = 0.03
        """
        super().__init__(epoch, pop_size, p_t, name=name, mode=mode)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        gb_fit = self.current_g_best().target.fitness
        ss = gb_fit - pop.F[-1] + self.EPSILON
        lb, ub = self.problem.lb, self.problem.ub
        sign = np.where(np.arange(n) <= int(self.pop_size / 2), 1.0, -1.0)[:, None]
        self.weights = 1 + sign * rng.uniform(0, 1, (n, d)) * np.log10((gb_fit - np.asarray(pop.F)) / ss + 1)[:, None]
        aa = np.arctanh(-(epoch / self.epoch) + 1)  # Eq.(2.4)
        bb = 1 - epoch / self.epoch
        p = np.tanh(np.abs(np.asarray(pop.F) - gb_fit))[:, None]  # Eq.(2.2)
        vb = rng.uniform(-aa, aa, (n, d))  # Eq.(2.3)
        vc = rng.uniform(-bb, bb, (n, d))
        ia, ib = ops.two_others(self, n, d)  # per dimension: two random other agents
        cols = np.arange(d)[None, :]
        pos = np.where(rng.random((n, d)) < p, g + vb * (self.weights * X[ia, cols] - X[ib, cols]), vc * X)  # Eq.(2.1)
        pos = np.where((rng.random(n) < self.p_t)[:, None], lb + rng.random((n, d)) * (ub - lb), pos)  # Eq.(2.7)
        ops.replace(self, pos)

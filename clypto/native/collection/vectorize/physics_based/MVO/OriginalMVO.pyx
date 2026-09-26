#!/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.physics_based.MVO.DevMVO cimport DevMVO
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalMVO(DevMVO):
    """
    The original version of: Multi-Verse Optimizer (MVO)

    Links:
        1. https://dx.doi.org/10.1007/s00521-015-1870-7
        2. https://www.mathworks.com/matlabcentral/fileexchange/50112-multi-verse-optimizer-mvo

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wep_min (float): [0.05, 0.3], Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
        + wep_max (float: [0.75, 1.0], Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import MVO    >>> import numpy as np
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
    >>> model = MVO.OriginalMVO(epoch=1000, pop_size=50, wep_min = 0.2, wep_max = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., Mirjalili, S.M. and Hatamlou, A., 2016. Multi-verse optimizer: a nature-inspired
    algorithm for global optimization. Neural Computing and Applications, 27(2), pp.495-513.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        wep_min: float = 0.2,
        wep_max: float = 1.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wep_min (float): Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
            wep_max (float: Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0
        """
        super().__init__(epoch, pop_size, wep_min, wep_max, name=name, mode=mode)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        wep = self.wep_min + epoch * ((self.wep_max - self.wep_min) / self.epoch)
        tdr = 1 - epoch ** (1.0 / 6) / (<object>self.epoch) ** (<object>(1.0 / 6))
        raw = np.array(pop.F)
        if raw.max() > (2 ** 64 - 1):
            norm = rng.uniform(0, 0.1, n)
        else:
            f = raw - raw.min()
            norm = f / f.sum() if f.sum() != 0 else rng.uniform(0.2, 0.8, n)
        # per (agent, dimension): take the value of a white hole chosen by roulette wheel on -fitness
        cum = np.cumsum(-1.0 * raw)
        p = rng.random((n, d)) * cum[-1]
        greater = cum[None, None, :] > p[..., None]
        white = np.where(greater.any(axis=-1), greater.argmax(axis=-1), 0)
        pos = np.where(rng.random((n, d)) < norm[:, None], X[white, np.arange(d)[None, :]], X)
        # wormhole: move around the best
        u = rng.uniform(lb, ub, (n, d))
        pos = np.where(rng.random((n, d)) < wep, np.where(rng.random((n, d)) < 0.5, g + tdr * u, g - tdr * u), pos)
        ops.step(self, pos)

#!/usr/bin/env python
# Created by "Thieu" at 18:09, 13/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalRIME(VectorizeOptimizer):
    """
    The original version of: physical phenomenon of RIME-ice  (RIME)

    Links:
        1. https://doi.org/10.1016/j.neucom.2023.02.010
        2. https://www.mathworks.com/matlabcentral/fileexchange/124610-rime-a-physics-based-optimization

    Notes (parameters):
        1. sr (float): Soft-rime parameters, default=5.0
        2. The algorithm is straightforward and does not require any specialized knowledge or techniques.
        3. The algorithm may exhibit slow convergence and may not perform optimally.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.physics_based import RIME    >>> import numpy as np
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
    >>> model = RIME.OriginalRIME(epoch=1000, pop_size=50, sr = 5.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Su, H., Zhao, D., Heidari, A. A., Liu, L., Zhang, X., Mafarja, M., & Chen, H. (2023). RIME: A physics-based optimization. Neurocomputing.
    """

    cdef public object sr

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        sr: float = 5.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            sr (float): Soft-rime parameters, default=5.0
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "sr"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.sr = cy.validator(float, sr, (0.0, 100.0), "sr")

    def _evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.bounds.low, self.problem.bounds.up
        rime_factor = (
                (rng.random() - 0.5)
                * 2
                * np.cos(np.pi * epoch / (self.epoch / 10))
                * (1 - np.round(epoch * self.sr / self.epoch) / self.sr)
        )
        ee = np.sqrt((epoch + 1) / self.epoch)
        fits = np.array(pop.F)
        fits_norm = (fits / np.linalg.norm(fits))[:, None]
        pos = np.where(rng.random((n, d)) < ee, g + rime_factor * (lb + rng.random((n, d)) * (ub - lb)), X)
        pos = np.where(rng.random((n, d)) < fits_norm, g, pos)
        ops.step(self, pos)

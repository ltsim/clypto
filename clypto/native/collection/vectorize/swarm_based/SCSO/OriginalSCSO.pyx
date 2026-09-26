#!/usr/bin/env python
# Created by "Thieu" at 17:36, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalSCSO(VectorizeOptimizer):
    """
    The original version of: Sand Cat Swarm Optimization (SCSO)

    Links:
        1. https://link.springer.com/article/10.1007/s00366-022-01604-x
        2. https://www.mathworks.com/matlabcentral/fileexchange/110185-sand-cat-swarm-optimization

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import SCSO    >>> import numpy as np
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
    >>> model = SCSO.OriginalSCSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Seyyedabbasi, A., & Kiani, F. (2022). Sand Cat swarm optimization: a nature-inspired algorithm to
    solve global optimization problems. Engineering with Computers, 1-25.
    """

    cdef public object ss
    cdef public object pp

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
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def _initialize_variables(self):
        self.ss = 2  # maximum Sensitivity range
        self.pp = np.arange(1, 361)

    def _evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        guides_r = self.ss - (self.ss * epoch / self.epoch)
        # roulette wheel over the 360 angles (the cumulative probabilities never change) and their cosines
        cum = np.cumsum(self.pp / np.sum(self.pp))
        cos_table = np.cos(np.arange(len(self.pp)))
        A = rng.random((n, 2))
        r = A[:, 0] * guides_r
        R = (2 * guides_r) * A[:, 1] - guides_r  # controls to transition phases
        U = rng.random((n, d, 3))
        teta = np.searchsorted(cum, U[..., 0], side="right")
        pos_attack = g - r[:, None] * np.abs(U[..., 1] * g - X) * cos_table[teta]
        cp = (U[..., 1] * n).astype(int)
        pos_search = r[:, None] * (X[cp, np.arange(d)[None, :]] - U[..., 2] * X)
        ops.replace(self, np.where((np.abs(R) <= 1)[:, None], pos_attack, pos_search))

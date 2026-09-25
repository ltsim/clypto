#!/usr/bin/env python
# Created by "Thieu" at 17:36, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalSCSO(LegacyNativeOptimizer):
    """
    The original version of: Sand Cat Swarm Optimization (SCSO)

    Links:
        1. https://link.springer.com/article/10.1007/s00366-022-01604-x
        2. https://www.mathworks.com/matlabcentral/fileexchange/110185-sand-cat-swarm-optimization

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import SCSO    >>> import numpy as np
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
        self.ss = 2  # maximum Sensitivity range
        self.pp = np.arange(1, 361)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        cols = np.arange(d)
        guides_r = self.ss - (self.ss * epoch / self.epoch)
        # roulette wheel over the 360 angles (the cumulative probabilities never change) and their cosines
        cum = np.cumsum(self.pp / np.sum(self.pp))
        cos_table = np.array([np.cos(k) for k in range(len(self.pp))])
        for idx in range(0, self.pop_size):
            r = self.generator.random() * guides_r
            R = (2 * guides_r) * self.generator.random() - guides_r  # controls to transition phases
            if -1 <= R <= 1:
                # per dimension: the roulette draw, then one draw for the random position
                U = self.generator.random((d, 2))
                teta = np.searchsorted(cum, U[:, 0], side="right")
                rand_pos = np.abs(U[:, 1] * g_best - Xp[idx])
                pos_new = g_best - r * rand_pos * cos_table[teta]
            else:
                # per dimension: the roulette draw (unused), the random agent and the random factor
                U = self.generator.random((d, 3))
                cp = (U[:, 1] * self.pop_size).astype(int)
                pos_new = r * (Xp[cp, cols] - U[:, 2] * Xp[idx])
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        self.pop = cand

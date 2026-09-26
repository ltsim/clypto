#!/usr/bin/env python
# Created by "Thieu" at 16:58, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalGSKA(VectorizeOptimizer):
    """
    The original version of: Gaining Sharing Knowledge-based Algorithm (GSKA)

    Links:
        1. https://doi.org/10.1007/s13042-019-01053-x

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.1, 0.5], percent of the best (p in the paper), default = 0.1
        + kf (float): [0.3, 0.8], knowledge factor that controls the total amount of gained and shared knowledge added from others to the current individual during generations, default = 0.5
        + kr (float): [0.5, 0.95], knowledge ratio, default = 0.9
        + kg (int): [3, 20], number of generations effect to D-dimension, default = 5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import GSKA    >>> import numpy as np
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
    >>> model = GSKA.OriginalGSKA(epoch=1000, pop_size=50, pb = 0.1, kf = 0.5, kr = 0.9, kg = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Mohamed, A.K., 2020. Gaining-sharing knowledge based algorithm for solving
    optimization problems: a novel nature-inspired algorithm. International Journal of Machine Learning and Cybernetics, 11(7), pp.1501-1529.
    """

    cdef public object pb
    cdef public object kf
    cdef public object kr
    cdef public object kg

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pb: float = 0.1,
        kf: float = 0.5,
        kr: float = 0.9,
        kg: int = 5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, n: pop_size, m: clusters
            pb (float): percent of the best   0.1%, 0.8%, 0.1% (p in the paper), default = 0.1
            kf (float): knowledge factor that controls the total amount of gained and shared knowledge added
                        from others to the current individual during generations, default = 0.5
            kr (float): knowledge ratio, default = 0.9
            kg (int): Number of generations effect to D-dimension, default = 5
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pb", "kf", "kr", "kg"],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pb = cy.validator(float, pb, (0, 1.0), "pb")
        self.kf = cy.validator(float, kf, (0, 1.0), "kf")
        self.kr = cy.validator(float, kr, (0, 1.0), "kr")
        self.kg = cy.validator(int, kg, [1, 1 + int(epoch / 2)], "kg")

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        F = np.asarray(pop.F)
        me = np.arange(n)
        dd = int(d * (1 - epoch_c / self.epoch) ** self.kg)
        prev, nxt = ops.neighbors(n)
        rand_idx = ops.exclude(rng.integers(0, n - 3, size=n), np.stack([prev, me, nxt], axis=1))
        id1 = int(self.pb * n)
        id2 = int(id1 + n * (1 - 2 * self.pb))
        best, worst, mid = (ops.pick_range(self, lo, hi, me) for lo, hi in ((0, id1), (id2, n), (id1, id2)))
        rb = ops.better(self, F[rand_idx], F)[:, None]
        mb = ops.better(self, F[mid], F)[:, None]
        junior = X + self.kf * (X[prev] - X[nxt] + np.where(rb, X[rand_idx] - X, X - X[rand_idx]))
        senior = X + self.kf * (X[best] - X[worst] + np.where(mb, X[mid] - X, X - X[mid]))
        pos = np.where(rng.uniform(size=(n, d)) <= self.kr, np.where((np.arange(d) < dd)[None, :], junior, senior), X)
        ops.step(self, pos)

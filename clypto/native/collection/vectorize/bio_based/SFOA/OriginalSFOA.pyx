#!/usr/bin/env python
# Created by "Thieu" at 22:37, 03/09/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalSFOA(LegacyNativeOptimizer):
    """
    The original version: Starfish Optimization Algorithm (SFOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/173735-starfish-optimization-algorithm-sfoa

    Notes:
        This algorithm claims to outperform 95 compared algorithms in accuracy and 97 algorithms in efficiency.
        However, it does not present any remarkable equations. Moreover, the provided MATLAB code does not
        include the standard CEC benchmark functions, but only simplified versions of them.
        Users should carefully consider this when validating the algorithm.
        Many new algorithms claim to be superior to other state-of-the-art methods,
        but it is evident that their implementations are often incorrect.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + gp (float): [0., 1] -> better [0.5, 0.7], the probablity for exploration

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import SFOA    >>> import numpy as np
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
    >>> model = SFOA.OriginalSFOA(epoch=1000, pop_size=50, gp = 0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    [1] Zhong, C., Li, G., Meng, Z., Li, H., Yildiz, A. R., & Mirjalili, S. (2025).
    Starfish optimization algorithm (SFOA): a bio-inspired metaheuristic algorithm for global
    optimization compared with 100 optimizers. Neural Computing and Applications, 37(5), 3641-3683.
    """

    cdef public object gp

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        gp: float = 0.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            gp (float): the exploration of starfish, default=0.5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "gp"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.gp = cy.validator(float, gp, [0, 1.0], "gp")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        me = np.arange(n)
        theta = np.pi / 2 * epoch / self.epoch
        tEO = (self.epoch - epoch) / self.epoch * np.cos(theta)
        if rng.random() < self.gp:  # exploration of starfish
            if d > 5:
                # five random dimensions of every agent move around the best
                pick = rng.random((n, d)).argsort(axis=1).argsort(axis=1) < 5
                pm = (2 * rng.random((n, d)) - 1) * np.pi
                pos = np.where(rng.random((n, d)) < self.gp, X + pm * (g - X) * np.cos(theta), X - pm * (g - X) * np.sin(theta))
                pos = np.where(pick & ((pos < lb) | (pos > ub)), X, np.where(pick, pos, X))
            else:
                # one random dimension moves with the help of two other agents
                jp = rng.integers(0, d, size=n)
                i1, i2 = ops.two_others(self, n, 1)
                diff1 = X[i1[:, 0], jp] - X[me, jp]
                diff2 = X[i2[:, 0], jp] - X[me, jp]
                v = tEO * X[me, jp] + (2 * rng.random(n) - 1) * diff1 + (2 * rng.random(n) - 1) * diff2
                v = np.where((v > ub[jp]) | (v < lb[jp]), X[me, jp], v)
                pos = np.array(X)
                pos[me, jp] = v
        else:  # exploitation of starfish
            df = rng.choice(n, 5, replace=False)
            dm = g - X[df]  # (5, d)
            kp = rng.random((n, 5)).argsort(axis=1)[:, :2]
            r = rng.random((n, 2, 1))
            pos = X + r[:, 0] * dm[kp[:, 0]] + r[:, 1] * dm[kp[:, 1]]
            pos[n - 1] = np.exp(-epoch * n / self.epoch) * X[n - 1]  # regeneration of starfish
        ops.step(self, pos)

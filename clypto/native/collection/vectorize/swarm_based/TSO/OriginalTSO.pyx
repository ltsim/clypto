#!/usr/bin/env python
# Created by "Thieu" at 17:52, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalTSO(LegacyNativeOptimizer):
    """
    The original version of: Tuna Swarm Optimization (TSO)

    Notes:
        1. Two variables that authors consider it as a constants (aa = 0.7 and zz = 0.05)
        2. https://www.hindawi.com/journals/cin/2021/9210050/
        3. https://www.mathworks.com/matlabcentral/fileexchange/101734-tuna-swarm-optimization

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import TSO    >>> import numpy as np
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
    >>> model = TSO.OriginalTSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Xie, L., Han, T., Zhou, H., Zhang, Z. R., Han, B., & Tang, A. (2021). Tuna swarm optimization: a novel swarm-based
    metaheuristic algorithm for global optimization. Computational intelligence and Neuroscience, 2021.
    """

    cdef public object aa
    cdef public object zz

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
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void initialize_variables(self):
        self.aa = 0.7
        self.zz = 0.05

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        C = epoch / self.epoch
        a1 = self.aa + (1 - self.aa) * C
        a2 = (1 - self.aa) - (1 - self.aa) * C
        tt = (1 - epoch / self.epoch) ** (epoch / self.epoch)
        prev = np.roll(X, 1, axis=0)
        prev[0] = X[0]
        rand_pos = lb + rng.random((n, d)) * (ub - lb)
        r1 = rng.random((n, 1))
        beta = np.exp(r1 * np.exp(3 * np.cos(np.pi * (self.epoch - epoch) / self.epoch))) * np.cos(2 * np.pi * r1)
        toward_best = (rng.random(n) < C)[:, None]
        spiral = np.where(
            toward_best,
            a1 * (g + beta * np.abs(g - X)) + a2 * prev,  # Eq. 8.4
            a1 * (rand_pos + beta * np.abs(rand_pos - X)) + a2 * prev,  # Eq. 8.2
        )
        tf = rng.choice([-1, 1], size=(n, 1))
        parabolic = np.where(
            (rng.random(n) < 0.5)[:, None],
            g + rng.random((n, d)) * (g - X) + tf * tt ** 2 * (g - X),  # Eq. 9.1
            tf * tt ** 2 * X,  # Eq. 9.2
        )
        pos = np.where((rng.random(n) > 0.5)[:, None], spiral, parabolic)
        pos = np.where((rng.random(n) < self.zz)[:, None], lb + rng.random((n, d)) * (ub - lb), pos)
        ops.replace(self, pos)

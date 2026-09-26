#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalDE(VectorizeOptimizer):
    """
    The original version of: Differential Evolution (DE)

    Links:
        1. https://doi.org/10.1016/j.swevo.2018.10.006

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wf (float): [-1., 1.0], weighting factor, default = 0.1
        + cr (float): [0.5, 0.95], crossover rate, default = 0.9
        + strategy (int): [0, 5], there are lots of variant version of DE algorithm,
            + 0: DE/current-to-rand/1/bin
            + 1: DE/best/1/bin
            + 2: DE/best/2/bin
            + 3: DE/rand/2/bin
            + 4: DE/current-to-best/1/bin
            + 5: DE/current-to-rand/1/bin

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.evolutionary_based import DE    >>> import numpy as np
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
    >>> model = DE.OriginalDE(epoch=1000, pop_size=50, wf = 0.7, cr = 0.9, strategy = 0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Jambi, K.M., 2019. Novel mutation strategy for enhancing SHADE and
    LSHADE algorithms for global numerical optimization. Swarm and Evolutionary Computation, 50, p.100455.
    """

    cdef public object wf
    cdef public object cr
    cdef public object strategy

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        wf: float = 0.1,
        cr: float = 0.9,
        strategy: int = 0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wf (float): weighting factor, default = 0.1
            cr (float): crossover rate, default = 0.9
            strategy (int): Different variants of DE, default = 0
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "wf", "cr", "strategy"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.wf = cy.validator(float, wf, (-3.0, 3.0), "wf")
        self.cr = cy.validator(float, cr, (0, 1.0), "cr")
        self.strategy = cy.validator(int, strategy, [0, 5], "strategy")

    def distinct_others__(self, n, k):
        """``k`` distinct random agent indices per agent, none of them the agent itself (n, k)."""
        keys = self.generator.random((n, n))
        keys[np.arange(n), np.arange(n)] = 2.0
        return np.argpartition(keys, k - 1, axis=1)[:, :k]

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        wf = self.wf
        if self.strategy == 0:
            i = self.distinct_others__(n, 3)
            pos = X[i[:, 0]] + wf * (X[i[:, 1]] - X[i[:, 2]])
        elif self.strategy == 1:
            i = self.distinct_others__(n, 2)
            pos = g + wf * (X[i[:, 0]] - X[i[:, 1]])
        elif self.strategy == 2:
            i = self.distinct_others__(n, 4)
            pos = g + wf * (X[i[:, 0]] - X[i[:, 1]]) + wf * (X[i[:, 2]] - X[i[:, 3]])
        elif self.strategy == 3:
            i = self.distinct_others__(n, 5)
            pos = X[i[:, 0]] + wf * (X[i[:, 1]] - X[i[:, 2]]) + wf * (X[i[:, 3]] - X[i[:, 4]])
        elif self.strategy == 4:
            i = self.distinct_others__(n, 2)
            pos = X + wf * (g - X) + wf * (X[i[:, 0]] - X[i[:, 1]])
        else:
            i = self.distinct_others__(n, 3)
            pos = X + wf * (X[i[:, 0]] - X) + wf * (X[i[:, 1]] - X[i[:, 2]])
        # binomial crossover
        ops.step(self, np.where(rng.random((n, d)) < self.cr, pos, X))

#!/usr/bin/env python
# Created by "Thieu" at 14:20, 15/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalSOS(LegacyNativeOptimizer):
    """
    The original version: Symbiotic Organisms Search (SOS)

    Links:
        1. https://doi.org/10.1016/j.compstruc.2014.03.007

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import SOS    >>> import numpy as np
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
    >>> model = SOS.OriginalSOS(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Cheng, M. Y., & Prayogo, D. (2014). Symbiotic organisms search: a new metaheuristic
    optimization algorithm. Computers & Structures, 139, 98-112.
    """

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        me = np.arange(n)
        # Mutualism phase: agent i and a random partner j both move towards the best (Eq. 3, 4)
        j = ops.others(self, n)[:, 0]
        mutual = (X + X[j]) / 2
        bf = rng.integers(1, 3, size=(n, 2, 1))
        xi_new = X + rng.random((n, 1)) * (g - bf[:, 0] * mutual)
        xj_new = X[j] + rng.random((n, 1)) * (g - bf[:, 1] * mutual)
        cand = pop.empty_like()
        cand.X[:] = self.correct_solution(xj_new)
        self.evaluate(cand, 0, n)
        ops.scatter(self, cand, j)  # the partners
        ops.step(self, xi_new)  # the agents
        # Commensalism phase (Eq. 5)
        X = pop.X
        j = ops.others(self, n)[:, 0]
        ops.step(self, X + rng.uniform(-1, 1, (n, 1)) * (g - X[j]))
        # Parasitism phase: a copy of a random agent with one random dimension re-initialized
        X = pop.X
        j = ops.others(self, n)[:, 0]
        pos = np.array(X[j])
        dim = rng.integers(0, d, size=n)
        pos[me, dim] = (lb + rng.random((n, d)) * (ub - lb))[me, dim]
        ops.step(self, pos)

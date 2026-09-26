#!/usr/bin/env python
# Created by "Thieu" at 11:10, 15/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalBMO(LegacyNativeOptimizer):
    """
    The original version: Barnacles Mating Optimizer (BMO)

    Links:
        1. https://ieeexplore.ieee.org/document/8441097

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pl (int): [1, pop_size - 1], barnacle’s threshold

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import BMO    >>> import numpy as np
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
    >>> model = BMO.OriginalBMO(epoch=1000, pop_size=50, pl = 4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2018. Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm
    for global optimisation problems. International journal of bio-inspired computation, 12(1), pp.1-22.
    """

    cdef public object pl

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        pl = 5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pl"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pl = cy.validator(int, pl, [1, self.pop_size - 1], "pl")

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t n = pop.n
        k1 = self.generator.permutation(self.pop_size)
        k2 = self.generator.permutation(self.pop_size)
        temp = np.abs(k1 - k2)
        p = self.generator.uniform(0, 1, n)[:, None]  # one draw per agent
        X = pop.X
        pos = np.where((temp <= self.pl)[:, None], p * X[k1] + (1 - p) * X[k2], p * X[k2])
        cand.X[:] = self.correct_solution(pos)
        self.evaluate(cand, 0, n)
        self.pop = cand

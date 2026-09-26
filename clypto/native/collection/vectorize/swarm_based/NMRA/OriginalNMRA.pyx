#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalNMRA(LegacyNativeOptimizer):
    """
    The original version of: Naked Mole-Rat Algorithm (NMRA)

    Links:
        1. https://www.doi.org10.1007/s00521-019-04464-7

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.5, 0.95], probability of breeding, default = 0.75

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import NMRA    >>> import numpy as np
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
    >>> model = NMRA.OriginalNMRA(epoch=1000, pop_size=50, pb = 0.75)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Salgotra, R. and Singh, U., 2019. The naked mole-rat algorithm. Neural Computing and Applications, 31(12), pp.8837-8857.
    """

    cdef public object pb
    cdef public object size_b

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pb: float = 0.75,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pb (float): probability of breeding, default = 0.75
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pb"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pb = cy.validator(float, pb, (0, 1.0), "pb")
        self.size_b = int(self.pop_size / 5)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        b = self.size_b
        m = n - b
        pos = np.array(X)
        # breeding operators
        alpha = rng.uniform(size=(b, 1))
        breed = (rng.uniform(size=(b, 1)) < self.pb)
        pos[:b] = np.where(breed, (1 - alpha) * X[:b] + alpha * (g - X[:b]), X[:b])
        # working operators: two distinct workers per agent
        t1 = b + rng.integers(0, m, size=m)
        t2 = b + (t1 - b + rng.integers(1, m, size=m)) % m
        pos[b:] = X[b:] + rng.uniform(size=(m, 1)) * (X[t1] - X[t2])
        ops.step(self, pos)

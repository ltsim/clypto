#!/usr/bin/env python
# Created by "Thieu" at 18:37, 28/05/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalCSA(LegacyNativeOptimizer):
    """
    The original version of: Cuckoo Search Algorithm (CSA)

    Links:
        1. https://doi.org/10.1109/NABIC.2009.5393690

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_a (float): [0.1, 0.7], probability a, default=0.3

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import CSA    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = CSA.OriginalCSA(epoch=1000, pop_size=50, p_a = 0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S. and Deb, S., 2009, December. Cuckoo search via Lévy flights. In 2009 World
    congress on nature & biologically inspired computing (NaBIC) (pp. 210-214). Ieee.
    """

    cdef public object p_a
    cdef public object n_cut

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        p_a: float = 0.3,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_a (float): probability a, default=0.3
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "p_a"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.p_a = cy.validator(float, p_a, (0, 1.0), "p_a")
        self.n_cut = int(self.p_a * self.pop_size)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        ## Generate levy-flight solution
        levy = self.get_levy_flight_step(multiplier=0.001, size=n, case=-1)
        k = 1.0 / np.sqrt(epoch) * np.sign(rng.random(n) - 0.5) * levy
        ops.step(self, X + k[:, None] * (X - g))
        ## Abandoned some worst nests
        ordered = pop.take(self.sorted_order(pop))
        new = self.new_population(rng.uniform(self.problem.lb, self.problem.ub, (self.n_cut, d)))
        self.pop = ordered.take(np.arange(self.pop_size - self.n_cut)).concat(new)

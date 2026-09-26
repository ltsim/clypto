#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native.fuzzy import FuzzySystem as FS
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class FuzzyGWO(LegacyNativeOptimizer):
    """
    The original version of: Fuzzy Hierarchical Operator - Grey Wolf Optimizer (FHO-GWO or FuzzyGWO or F-GWO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2017.03.048

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.FuzzyGWO(epoch=1000, pop_size=50, fuzzy_name="increase")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rodríguez, Luis, Oscar Castillo, José Soria, Patricia Melin, Fevrier Valdez, Claudia I. Gonzalez, Gabriela E. Martinez, and Jesus Soto. "A fuzzy hierarchical operator in the grey wolf optimizer algorithm." Applied Soft Computing 57 (2017): 315-328.
    """

    FUZZY_OPERATORS = ["increase", "decrease"]

    cdef public object fuzzy_name
    cdef public object fuzzy_system

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        fuzzy_name: str = "increase",
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            fuzzy_name (str): type of fuzzy operator to use, default = "increase"
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "fuzzy_name"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.fuzzy_name = cy.validator(str, fuzzy_name, FuzzyGWO.FUZZY_OPERATORS, "fuzzy_name")

    cdef void initialize_variables(self):
        self.fuzzy_system = FS(self.fuzzy_name)

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t n = pop.n, d = pop.d
        # linearly decreased from 2 to 0
        a = 2 - 2.0 * epoch / self.epoch
        best = pop.X[self.sorted_order(pop)[:3]][None]
        R = self.generator.random((n, 6, d))  # per agent: A1..A3 then C1..C3 draws
        A = a * (2 * R[:, :3] - 1)
        C = 2 * R[:, 3:]
        Xs = best - A * np.abs(C * best - pop.X[:, None, :])
        # Get fuzzy weights (they only depend on the epoch)
        FW_alpha, FW_beta, FW_delta = self.fuzzy_system.get_fuzzy_weights(epoch, self.epoch)
        total_weight = FW_alpha + FW_beta + FW_delta
        cand.X[:] = self.correct_solution((Xs[:, 0] * FW_alpha + Xs[:, 1] * FW_beta + Xs[:, 2] * FW_delta) / total_weight)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

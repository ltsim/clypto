#!/usr/bin/env python
# Created by "Thieu" at 18:08, 19/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalCEM(LegacyNativeOptimizer):
    """
    The original version of: Cross-Entropy Method (CEM)

    Links:
        1. https://github.com/clever-algorithms/CleverAlgorithms
        2. https://doi.org/10.1007/s10479-005-5724-z

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_best (int): N selected solutions as a samples for next evolution
        + alpha (float): weight factor for means and stdevs (normal distribution)

    Examples
    ~~~~~~~~
    >>> from clypto.collection.math_based import CEM    >>> import numpy as np
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
    >>> model = CEM.OriginalCEM(epoch=1000, pop_size=50, n_best = 20, alpha = 0.7)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] De Boer, P.T., Kroese, D.P., Mannor, S. and Rubinstein, R.Y., 2005. A tutorial on the
    cross-entropy method. Annals of operations research, 134(1), pp.19-67.
    """

    cdef public int n_best
    cdef public double alpha
    cdef public object means
    cdef public object stdevs

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_best: int = 10,
        alpha: float = 0.7,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_best (int): N selected solutions as a samples for next evolution
            alpha (float): weight factor for means and stdevs (normal distribution)
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "n_best", "alpha"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.n_best = cy.validator(int, n_best, [2, int(self.pop_size / 2)], "n_best")
        self.alpha = cy.validator(float, alpha, (0, 1.0), "alpha")

    cdef void initialize_variables(self):
        self.means = self.generator.uniform(self.problem.lb, self.problem.ub)
        self.stdevs = np.abs(self.problem.ub - self.problem.lb)

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t n = pop.n, d = pop.d
        ## Selected the best samples and update means and stdevs
        pos_list = np.ascontiguousarray(pop.X[: self.n_best])
        means_new = np.mean(pos_list, axis=0)
        means_new_repeat = np.repeat(means_new.reshape((1, -1)), self.n_best, axis=0)
        stdevs_new = np.mean((pos_list - means_new_repeat) ** 2, axis=0)
        self.means = self.alpha * self.means + (1.0 - self.alpha) * means_new
        self.stdevs = np.abs(self.alpha * self.stdevs + (1.0 - self.alpha) * stdevs_new)
        ## Create new population for next generation
        cand.X[:] = self.correct_solution(self.generator.normal(self.means, self.stdevs, (n, d)))
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

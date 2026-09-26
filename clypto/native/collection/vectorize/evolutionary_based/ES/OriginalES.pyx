#!/usr/bin/env python
# Created by "Thieu" at 18:14, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport _LegacyAgent


from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalES(LegacyNativeOptimizer):
    """
    The original version of: Evolution Strategies (ES)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): [0.5, 1.0], Percentage of child agents evolving in the next generation

    Examples
    ~~~~~~~~
    >>> from clypto.collection.evolutionary_based import ES    >>> import numpy as np
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
    >>> model = ES.OriginalES(epoch=1000, pop_size=50, lamda = 0.75)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Beyer, H.G. and Schwefel, H.P., 2002. Evolution strategies–a comprehensive introduction. Natural computing, 1(1), pp.3-52.
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        lamda: float = 0.75,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            lamda (float): Percentage of child agents evolving in the next generation, default=0.75
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "lamda"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.lamda = cy.validator(float, lamda, (0, 1.0), "lamda")
        self.n_child = int(self.lamda * self.pop_size)

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m):
        return [("S", d)]  # strategy (step size) of every agent

    cdef void initialize_variables(self):
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)

    cdef void init_fields(self, NativePopulation pop):
        pop.field("S")[:] = self.generator.uniform(0, self.distance, (pop.n, pop.d))

    def children__(self, NativePopulation pop, pos):
        """Children of the first ``n_child`` parents: their positions are ``pos``, their strategies are inherited with log-normal noise."""
        rng = self.generator
        d = pop.d
        nc = len(pos)
        tau = np.sqrt(2.0 * d) ** (-1.0)
        tau_p = np.sqrt(2.0 * np.sqrt(d)) ** (-1.0)
        kids = pop.take(np.arange(nc))
        kids.X[:] = self.correct_solution(pos)
        kids.field("S")[:] = np.exp(tau_p * rng.normal(0, 1.0, (nc, d)) + tau * rng.normal(0, 1.0, (nc, d)))
        self.evaluate(kids, 0, nc)
        return kids

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation kids, both
        nc = self.n_child
        X, S = np.asarray(pop.X), np.asarray(pop.field("S"))
        kids = self.children__(pop, X[:nc] + S[:nc] * self.generator.normal(0, 1.0, (nc, pop.d)))
        both = kids.concat(pop)
        self.pop = both.take(self.sorted_order(both)[:self.pop_size])

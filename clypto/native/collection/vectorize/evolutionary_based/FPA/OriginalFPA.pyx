#!/usr/bin/env python
# Created by "Thieu" at 19:34, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalFPA(LegacyNativeOptimizer):
    """
    The original version of: Flower Pollination Algorithm (FPA)

    Links:
        1. https://doi.org/10.1007/978-3-642-32894-7_27

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_s (float): [0.5, 0.95], switch probability, default = 0.8
        + levy_multiplier: [0.0001, 1000], mutiplier factor of Levy-flight trajectory, depends on the problem

    Examples
    ~~~~~~~~
    >>> from clypto.collection.evolutionary_based import FPA    >>> import numpy as np
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
    >>> model = FPA.OriginalFPA(epoch=1000, pop_size=50, p_s = 0.8, levy_multiplier = 0.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2012, September. Flower pollination algorithm for global optimization. In International
    conference on unconventional computing and natural computation (pp. 240-249). Springer, Berlin, Heidelberg.
    """

    cdef public object p_s
    cdef public object levy_multiplier

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        p_s: float = 0.8,
        levy_multiplier: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_s (float): switch probability, default = 0.8
            levy_multiplier (float): multiplier factor of Levy-flight trajectory, default = 0.2
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "p_s", "levy_multiplier"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.p_s = cy.validator(float, p_s, (0, 1.0), "p_s")
        self.levy_multiplier = cy.validator(float, levy_multiplier, (-10000, 10000), "levy_multiplier")

    cdef object amend_solution(self, object solution):
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        random_pos = self.problem.generate_solution()
        return np.where(condition, solution, random_pos)

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        for idx in range(0, self.pop_size):
            if self.generator.uniform() < self.p_s:
                levy = self.get_levy_flight_step(
                    multiplier=self.levy_multiplier, size=self.problem.n_dims, case=-1
                )
                pos_new = Xp[idx] + 1.0 / np.sqrt(epoch) * levy * (
                        Xp[idx] - g_best
                )
            else:
                id1, id2 = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
                )
                pos_new = Xp[idx] + self.generator.uniform() * (
                        Xp[id1] - Xp[id2]
                )
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np

from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.native.collection.vectorize.swarm_based.PSO.P_PSO cimport P_PSO


cdef class HPSO_TVAC(P_PSO):
    """
    The original version of: Hierarchical PSO Time-Varying Acceleration (HPSO-TVAC)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + ci (float): [0.3, 1.0], c initial, default = 0.5
        + cf (float): [0.0, 0.3], c final, default = 0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import PSO    >>> import numpy as np
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
    >>> model = PSO.HPSO_TVAC(epoch=1000, pop_size=50, ci=0.5, cf=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ghasemi, M., Aghaei, J. and Hadipour, M., 2017. New self-organising hierarchical PSO with
    jumping time-varying acceleration coefficients. Electronics Letters, 53(20), pp.1360-1362.
    """

    cdef public double ci
    cdef public double cf

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        ci: float = 0.5,
        cf: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            ci: c initial, default = 0.5
            cf: c final, default = 0.0
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "ci", "cf"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.ci = cy.validator(float, ci, [0.3, 1.0], "ci")
        self.cf = cy.validator(float, cf, [0, 0.3], "cf")

    cdef void evolve(self, int epoch):
        # Sequential: reads pop[idx_k] after earlier agents updated it and draws a
        # data-dependent number of normals, so agents are processed one by one.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        cdef double c_it = ((self.cf - self.ci) * (epoch / <double>self.epoch)) + self.ci
        X, V, P = pop.X, pop.field("V"), pop.field("P")
        for idx in range(n):
            idx_k = self.generator.integers(0, self.pop_size)
            w = self.generator.normal()
            while np.abs(w - 1.0) < 0.01:
                w = self.generator.normal()
            c1_it = np.abs(w) ** (c_it * w)
            c2_it = np.abs(1 - w) ** (c_it / (1 - w))
            #################### HPSO
            v_new = c1_it * self.generator.uniform(0, 1, d) * (P[idx] - X[idx]) \
                + c2_it * self.generator.uniform(0, 1, d) * (self.g_best_x() + P[idx_k] - 2 * X[idx])
            v_new = np.where(
                v_new == 0,
                np.sign(0.5 - self.generator.uniform()) * self.generator.uniform() * self.v_max,
                v_new,
            )
            v_new = np.sign(v_new) * np.minimum(np.abs(v_new), self.v_max)
            #########################
            v_new = np.minimum(np.maximum(v_new, -self.v_max), self.v_max)
            cand.X[idx] = self.correct_solution(X[idx] + v_new)
            V[idx] = v_new
            self.evaluate(cand, idx, idx + 1)
            self.accept(cand, idx, idx + 1)

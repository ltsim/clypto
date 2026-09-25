#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np

from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.collection.swarm_based.PSO._base cimport _PSOBase


cdef class AIW_PSO(_PSOBase):
    """
    The original version of: Adaptive Inertia Weight Particle Swarm Optimization (AIW-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + alpha (float): [0., 1.0], The positive constant, default = 0.4

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import PSO    >>> import numpy as np
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
    >>> model = PSO.AIW_PSO(epoch=1000, pop_size=50, c1=2.05, c2=20.5, alpha=0.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Qin, Z., Yu, F., Shi, Z., Wang, Y. (2006). Adaptive Inertia Weight Particle Swarm Optimization. In: Rutkowski, L.,
    Tadeusiewicz, R., Zadeh, L.A., Żurada, J.M. (eds) Artificial Intelligence and Soft Computing – ICAISC 2006. ICAISC 2006.
    Lecture Notes in Computer Science(), vol 4029. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11785231_48
    """

    cdef public double c1
    cdef public double c2
    cdef public double alpha

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 2.05,
        c2: float = 2.05,
        alpha: float = 0.4,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
            alpha: The positive constant, default = 0.4
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "c1", "c2", "alpha"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.c1 = cy.validator(float, c1, (0, 5.0), "c1")
        self.c2 = cy.validator(float, c2, (0, 5.0), "c2")
        self.alpha = cy.validator(float, alpha, [0.0, 1.0], "alpha")

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t start, stop, n = pop.n, d = pop.d
        current_best = pop.X[self.sorted_order(pop)[0]].copy()
        R = self.generator.random((n, 3, d))
        X, V, P = pop.X, pop.field("V"), pop.field("P")
        for start, stop in self.chunks(n):
            g = np.array(self.g_best_x(), dtype=float)
            Xs, Ps = np.ascontiguousarray(X[start:stop]), np.ascontiguousarray(P[start:stop])
            denom = np.abs(Ps - current_best)
            denom = np.where(denom == 0, 1e-6, denom)
            isa = np.abs(Xs - Ps) / denom  # individual search ability
            w = 1 - self.alpha * (1.0 / (1.0 + np.exp(-isa)))
            cognitive = self.c1 * R[start:stop, 0] * (Ps - Xs)
            social = self.c2 * R[start:stop, 1] * (g - Xs)
            velocity = w * V[start:stop] + cognitive + social
            V[start:stop] = np.clip(velocity, self.v_min, self.v_max)
            pos = self.amend_random(Xs + V[start:stop], R[start:stop, 2])
            cand.X[start:stop] = self.problem.correct_solutions(pos)
            self.evaluate(cand, start, stop)
            self.accept(cand, start, stop)

#!/usr/bin/env python
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
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


cdef class AdaptiveBA(LegacyNativeOptimizer):
    """
    The original version of: Adaptive Bat-inspired Algorithm (ABA)

    Notes
    ~~~~~
    + The value of A and r are changing after each iteration

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + loudness_min (float): A_min - loudness, default=1.0
        + loudness_max (float): A_max - loudness, default=2.0
        + pr_min (float): pulse rate / emission rate min, default = 0.15
        + pr_max (float): pulse rate / emission rate max, default = 0.85
        + pf_min (float): pulse frequency min, default = 0
        + pf_max (float): pulse frequency max, default = 10

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import BA    >>> import numpy as np
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
    >>> model = BA.AdaptiveBA(epoch=1000, pop_size=50, loudness_min = 1.0, loudness_max = 2.0, pr_min = -2.5, pr_max = 0.85, pf_min = 0.1, pf_max = 10.)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2010. A new metaheuristic bat-inspired algorithm. In Nature inspired cooperative
    strategies for optimization (NICSO 2010) (pp. 65-74). Springer, Berlin, Heidelberg.
    """

    cdef public object loudness_min
    cdef public object loudness_max
    cdef public object pr_min
    cdef public object pr_max
    cdef public object pf_min
    cdef public object pf_max
    cdef public object alpha
    cdef public object gamma

    cdef public object velocity
    cdef public object loudness_v
    cdef public object pulse_rate_v
    def __init__(
        self,
        epoch: int = 10000,
        pop_size: object = 100,
        loudness_min: float = 1.0,
        loudness_max: float = 2.0,
        pr_min: float = 0.15,
        pr_max: float = 0.85,
        pf_min: float = -10.0,
        pf_max: float = 10.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            loudness_min (float): A_min - loudness, default=1.0
            loudness_max (float): A_max - loudness, default=2.0
            pr_min (float): pulse rate / emission rate min, default = 0.15
            pr_max (float): pulse rate / emission rate max, default = 0.85
            pf_min (float): pulse frequency min, default = 0
            pf_max (float): pulse frequency max, default = 10
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "loudness_min",
                "loudness_max",
                "pr_min",
                "pr_max",
                "pf_min",
                "pf_max",
            ],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.loudness_min = cy.validator(float, loudness_min, [0.5, 1.5], "loudness_min")
        self.loudness_max = cy.validator(float, loudness_max, [1.5, 3.0], "loudness_max")
        self.pr_min = cy.validator(float, pr_min, [-10.0, 10.0], "pr_min")
        self.pr_max = cy.validator(float, pr_max, [-10.0, 10.0], "pr_max")
        self.pf_min = cy.validator(float, pf_min, [-10.0, 10.0], "pf_min")
        self.pf_max = cy.validator(float, pf_max, [0.0, 10.0], "pf_max")
        self.alpha = self.gamma = 0.9

    cdef void initialization(self):
        LegacyNativeOptimizer.initialization(self)
        n, d = self.pop.n, self.pop.d
        self.velocity = self.generator.uniform(self.problem.lb, self.problem.ub, (n, d))
        self.loudness_v = self.generator.uniform(self.loudness_min, self.loudness_max, n)
        self.pulse_rate_v = self.generator.uniform(self.pr_min, self.pr_max, n)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        mean_a = np.mean(self.loudness_v)
        pf = rng.uniform(self.pf_min, self.pf_max, (n, 1))
        x_new = X + self.velocity + pf * (X - g)
        x_new = np.where((rng.random(n) > self.pulse_rate_v)[:, None], g + mean_a * rng.normal(-1, 1, (n, 1)), x_new)
        cand = pop.empty_like()
        cand.X[:] = self.correct_solution(x_new)
        self.evaluate(cand, 0, n)
        ok = ops.better(self, np.asarray(cand.F), np.asarray(pop.F)) & (rng.random(n) < self.loudness_v)
        pop.buf[ok] = cand.buf[ok]
        self.loudness_v = np.where(ok, self.alpha * self.loudness_v, self.loudness_v)
        self.pulse_rate_v = np.where(ok, self.pulse_rate_v * (1 - np.exp(-self.gamma * epoch_c)), self.pulse_rate_v)

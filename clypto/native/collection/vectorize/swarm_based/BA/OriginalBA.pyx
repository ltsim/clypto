#!/usr/bin/env python
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer._native.agent cimport _LegacyAgent


from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalBA(LegacyNativeOptimizer):
    """
    The original version of: Bat-inspired Algorithm (BA)

    Notes
    ~~~~~
    + The value of A and r parameters are constant

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + loudness (float): (1.0, 2.0), loudness, default = 0.8
        + pulse_rate (float): (0.15, 0.85), pulse rate / emission rate, default = 0.95
        + pulse_frequency (list, tuple): (pf_min, pf_max) -> ([0, 3], [5, 20]), pulse frequency, default = (0, 10)

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import BA    >>> import numpy as np
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
    >>> model = BA.OriginalBA(epoch=1000, pop_size=50, loudness=0.8, pulse_rate=0.95, pf_min=0.1, pf_max=10.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2010. A new metaheuristic bat-inspired algorithm. In Nature inspired cooperative
    strategies for optimization (NICSO 2010) (pp. 65-74). Springer, Berlin, Heidelberg.
    """

    cdef public object loudness
    cdef public object pulse_rate
    cdef public object pf_min
    cdef public object pf_max
    cdef public object alpha
    cdef public object gamma

    cdef public object velocity
    cdef public object pulse_frequency
    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        loudness: float = 0.8,
        pulse_rate: float = 0.95,
        pf_min: float = 0.0,
        pf_max: float = 10.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            loudness (float): (A_min, A_max): loudness, default = 0.8
            pulse_rate (float): (r_min, r_max): pulse rate / emission rate, default = 0.95
            pf_min (float): pulse frequency min, default = 0
            pf_max (float): pulse frequency max, default = 10
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "loudness", "pulse_rate", "pf_min", "pf_max"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.loudness = cy.validator(float, loudness, (0, 1.0), "loudness")
        self.pulse_rate = cy.validator(float, pulse_rate, (0, 1.0), "pulse_rate")
        self.pf_min = cy.validator(float, pf_min, [0.0, 3.0], "pf_min")
        self.pf_max = cy.validator(float, pf_max, [5.0, 20.0], "pf_max")
        self.alpha = self.gamma = 0.9

    cdef void initialization(self):
        LegacyNativeOptimizer.initialization(self)
        n, d = self.pop.n, self.pop.d
        self.velocity = self.generator.uniform(self.problem.lb, self.problem.ub, (n, d))
        self.pulse_frequency = self.pf_min + (self.pf_max - self.pf_min) * self.generator.uniform(size=(n, 1))

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        x_new = X + self.velocity
        ## Local search around g_best position
        x_new = np.where((rng.random(n) > self.pulse_rate)[:, None], g + 0.001 * rng.normal(d, 1.0, (n, 1)), x_new)
        cand = pop.empty_like()
        cand.X[:] = self.correct_solution(x_new)
        self.evaluate(cand, 0, n)
        ## Replace the old position by the new one when it is better (and the sound is quiet enough)
        ok = ops.better(self, np.asarray(cand.F), np.asarray(pop.F)) & (rng.random(n) < self.loudness)
        pop.buf[ok] = cand.buf[ok]

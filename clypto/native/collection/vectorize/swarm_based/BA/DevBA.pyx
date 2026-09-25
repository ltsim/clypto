#!/usr/bin/env python
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class DevBA(LegacyNativeOptimizer):
    """
    The original version of: Developed Bat-inspired Algorithm (DBA)

    Notes
    ~~~~~
    + A (loudness) parameter is removed
    + Flow is changed:
        + 1st: the exploration phase is proceed (using frequency)
        + 2nd: If new position has better fitness, replace the old position
        + 3rd: Otherwise, proceed exploitation phase (using finding around the best position so far)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pulse_rate (float): [0.7, 1.0], pulse rate / emission rate, default = 0.95
        + pulse_frequency (tuple, list): (pf_min, pf_max) -> ([0, 3], [5, 20]), pulse frequency, default = (0, 10)

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
    >>> model = BA.DevBA(epoch=1000, pop_size=50, pulse_rate = 0.95, pf_min = 0., pf_max = 10.)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    cdef public object pulse_rate
    cdef public object pf_min
    cdef public object pf_max
    cdef public object alpha
    cdef public object gamma
    cdef public object dyn_list_velocity

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        pulse_rate = 0.95,
        pf_min = 0.0,
        pf_max = 10.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pulse_rate", "pf_min", "pf_max"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pulse_rate = cy.validator(float, pulse_rate, (0, 1.0), "pulse_rate")
        self.pf_min = cy.validator(float, pf_min, [0, 2], "pf_min")
        self.pf_max = cy.validator(float, pf_max, [2, 10], "pf_max")
        self.alpha = self.gamma = 0.9

    cdef void initialize_variables(self):
        self.dyn_list_velocity = np.zeros((self.pop_size, self.problem.n_dims))

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand, child
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        pf = self.pf_min + (self.pf_max - self.pf_min) * rng.uniform(size=(n, 1))  # Eq. 2
        self.dyn_list_velocity = rng.uniform(size=(n, 1)) * self.dyn_list_velocity + (g - X) * pf  # Eq. 3
        cand = pop.empty_like()
        cand.X[:] = self.correct_solution(X + self.dyn_list_velocity)  # Eq. 4
        self.evaluate(cand, 0, n)
        # agents whose move did not improve them try a local search around the best
        retry = np.flatnonzero(~ops.better(self, np.asarray(cand.F), np.asarray(pop.F)) & (rng.random(n) > self.pulse_rate))
        if len(retry):
            child = pop.take(retry)
            child.X[:] = self.correct_solution(g + 0.01 * rng.uniform(lb, ub, (len(retry), d)))
            self.evaluate(child, 0, len(retry))
            ops.scatter(self, child, retry, dst=cand)
        self.pop = cand

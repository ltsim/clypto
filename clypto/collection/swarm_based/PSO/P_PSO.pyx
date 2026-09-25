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


cdef class P_PSO(_PSOBase):
    """
    The original version of: Phasor Particle Swarm Optimization (P-PSO)

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
    >>> model = PSO.P_PSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ghasemi, M., Akbari, E., Rahimnejad, A., Razavi, S.E., Ghavidel, S. and Li, L., 2019.
    Phasor particle swarm optimization: a simple and efficient variant of PSO. Soft Computing, 23(19), pp.9701-9718.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.dyn_delta_list = self.generator.uniform(0, 2 * np.pi, self.pop_size)

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, start, stop, n = pop.n
        span = self.problem.ub - self.problem.lb
        # Per-agent scalars, computed with the same NumPy scalar operations as the
        # classic loop. Agent i+1 clips with the v_max left by agent i.
        aa, bb, ee, tt, v_limit = [], [], [], [], []
        for idx in range(n):
            delta = self.dyn_delta_list[idx]
            a_i = 2 * (np.sin(delta))
            b_i = 2 * (np.cos(delta))
            aa.append(a_i)
            bb.append(b_i)
            ee.append(np.abs(np.cos(delta)) ** a_i)
            tt.append(np.abs(np.sin(delta)) ** b_i)
            v_limit.append(self.v_max)
            self.dyn_delta_list[idx] += np.abs(a_i + b_i) * (2 * np.pi)
            self.v_max = (np.abs(np.cos(self.dyn_delta_list[idx])) ** 2) * span
        ee = np.array(ee)[:, None]
        tt = np.array(tt)[:, None]
        v_limit = np.array([np.broadcast_to(v, span.shape) for v in v_limit])
        X, V, P = pop.X, pop.field("V"), pop.field("P")
        for start, stop in self.chunks(n):
            g = np.array(self.g_best_x(), dtype=float)
            Xs, lim = X[start:stop], v_limit[start:stop]
            v_new = ee[start:stop] * (P[start:stop] - Xs) + tt[start:stop] * (g - Xs)
            v_new = np.minimum(np.maximum(v_new, -lim), lim)
            V[start:stop] = v_new
            cand.X[start:stop] = self.correct_solution(Xs + v_new)
            self.evaluate(cand, start, stop)
            self.accept(cand, start, stop)

#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent cimport LegacyNativeAgent
from clypto.optimizer.native.target cimport NativeTarget


cdef class SADE(LegacyNativeOptimizer):
    """
    The original version of: Self-Adaptive Differential Evolution (SADE)

    Links:
        1. https://doi.org/10.1109/CEC.2005.1554904

    Examples
    ~~~~~~~~
    >>> from clypto.collection.evolutionary_based import DE    >>> import numpy as np
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
    >>> model = DE.SADE(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Qin, A.K. and Suganthan, P.N., 2005, September. Self-adaptive differential evolution algorithm for
    numerical optimization. In 2005 IEEE congress on evolutionary computation (Vol. 2, pp. 1785-1791). IEEE.
    """

    cdef public object loop_probability
    cdef public object loop_cr
    cdef public object ns1
    cdef public object ns2
    cdef public object nf1
    cdef public object nf2
    cdef public object crm
    cdef public object p1
    cdef public object dyn_list_cr
    cdef public object objs

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
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void initialize_variables(self):
        self.loop_probability = 50
        self.loop_cr = 5
        self.ns1 = self.ns2 = self.nf1 = self.nf2 = 0
        self.crm = 0.5
        self.p1 = 0.5
        self.dyn_list_cr = list()

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = np.array(pop.X)
        me = np.arange(n)
        g = np.array(self.g_best_x())
        # adaptive parameters: cr ~ N(crm, 0.1), f ~ N(0.5, 0.3) re-drawn while negative, capped at 1
        cr = np.clip(rng.normal(self.crm, 0.1, n), 0, 1)
        f = rng.normal(0.5, 0.3, n)
        while np.any(f < 0):
            bad = f < 0
            f[bad] = rng.normal(0.5, 0.3, int(bad.sum()))
        f = np.minimum(f, 1.0)[:, None]
        i = ops.k_others(self, n, 3)
        strategy1 = rng.random(n) < self.p1
        x_new = np.where(strategy1[:, None], X[i[:, 0]] + f * (X[i[:, 1]] - X[i[:, 2]]),
                         X + f * (g - X) + f * (X[i[:, 0]] - X[i[:, 1]]))
        pos = np.where(rng.random((n, d)) < cr[:, None], x_new, X)
        j_rand = rng.integers(0, d, size=n)
        pos[me, j_rand] = x_new[me, j_rand]
        before = np.array(pop.F)
        ops.step(self, pos)
        ok = ops.better(self, np.asarray(self.pop.F), before)
        self.ns1 += int((ok & strategy1).sum())
        self.nf1 += int((~ok & strategy1).sum())
        self.ns2 += int((ok & ~strategy1).sum())
        self.nf2 += int((~ok & ~strategy1).sum())
        self.dyn_list_cr.extend(cr[ok & ~strategy1].tolist())
        # update cr and p1 periodically
        if epoch_c % self.loop_cr == 0 and len(self.dyn_list_cr):
            self.crm = np.mean(self.dyn_list_cr)
            self.dyn_list_cr = list()
        if epoch_c % self.loop_probability == 0:
            den = self.ns2 * (self.ns1 + self.nf1) + self.ns1 * (self.ns2 + self.nf2)
            if den > 0:
                self.p1 = self.ns1 * (self.ns2 + self.nf2) / den
            self.ns1 = self.ns2 = self.nf1 = self.nf2 = 0

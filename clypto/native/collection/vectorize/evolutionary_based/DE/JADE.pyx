#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from scipy.stats import cauchy
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent cimport LegacyNativeAgent
from clypto.optimizer.native.target cimport NativeTarget


cdef class JADE(LegacyNativeOptimizer):
    """
    The original version of: Differential Evolution (JADE)

    Links:
        1. https://doi.org/10.1109/TEVC.2009.2014613

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial adaptive f, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial adaptive cr, default = 0.5
        + pt (float): [0.05, 0.2], The percent of top best agents (p in the paper), default = 0.1
        + ap (float): [0.05, 0.2], The Adaptation Parameter control value of f and cr (c in the paper), default=0.1

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
    >>> model = DE.JADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5, pt = 0.1, ap = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhang, J. and Sanderson, A.C., 2009. JADE: adaptive differential evolution with optional
    external archive. IEEE Transactions on evolutionary computation, 13(5), pp.945-958.
    """

    cdef public object miu_f
    cdef public object miu_cr
    cdef public object pt
    cdef public object ap
    cdef public object dyn_miu_cr
    cdef public object dyn_miu_f
    cdef public object dyn_pop_archive
    cdef public object objs

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        miu_f: float = 0.5,
        miu_cr: float = 0.5,
        pt: float = 0.1,
        ap: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial adaptive f, default = 0.5
            miu_cr (float): initial adaptive cr, default = 0.5
            pt (float): The percent of top best agents (p in the paper), default = 0.1
            ap (float): The Adaptation Parameter control value of f and cr (c in the paper), default=0.1
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "miu_f", "miu_cr", "pt", "ap"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.miu_f = cy.validator(float, miu_f, (0, 1.0), "miu_f")
        self.miu_cr = cy.validator(float, miu_cr, (0, 1.0), "miu_cr")
        self.pt = cy.validator(float, pt, (0, 1.0), "pt")
        self.ap = cy.validator(float, ap, (0, 1.0), "ap")

    cdef void initialize_variables(self):
        self.dyn_miu_cr = self.miu_cr
        self.dyn_miu_f = self.miu_f
        self.dyn_pop_archive = np.empty((0, self.problem.n_dims))

    def lehmer_mean(self, list_objects):
        temp = np.sum(list_objects)
        return 0 if temp == 0 else np.sum(list_objects**2) / temp

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = np.array(pop.X)
        me = np.arange(n)
        # adaptive parameters: cr ~ N(miu_cr, 0.1), f ~ Cauchy(miu_f, 0.1) re-drawn while negative, capped at 1
        cr = np.clip(rng.normal(self.dyn_miu_cr, 0.1, n), 0, 1)
        f = self.dyn_miu_f + 0.1 * rng.standard_cauchy(n)
        while np.any(f < 0):
            bad = f < 0
            f[bad] = self.dyn_miu_f + 0.1 * rng.standard_cauchy(int(bad.sum()))
        f = np.minimum(f, 1.0)
        top = int(n * self.pt)
        x_best = X[self.sorted_order(pop)[:top]][rng.integers(0, top, size=n)]
        r1 = ops.others(self, n)[:, 0]
        union = np.vstack([X, self.dyn_pop_archive])
        r2 = ops.exclude(rng.integers(0, len(union) - 2, size=n), np.stack([me, r1], axis=1))
        f_ = f[:, None]
        x_new = X + f_ * (x_best - X) + f_ * (X[r1] - union[r2])
        pos = np.where(rng.random((n, d)) < cr[:, None], x_new, X)
        j_rand = rng.integers(0, d, size=n)
        pos[me, j_rand] = x_new[me, j_rand]
        before = np.array(pop.F)
        ops.step(self, pos)
        improved = ops.better(self, np.asarray(self.pop.F), before)
        # the replaced parents go to the archive (kept at most pop_size long), the parameters that worked adapt the means
        archive = np.vstack([self.dyn_pop_archive, X[improved]])
        extra = len(archive) - n
        if extra > 0:
            archive = np.delete(archive, rng.choice(len(archive), extra, replace=False), axis=0)
        self.dyn_pop_archive = archive
        list_cr, list_f = cr[improved], f[improved]
        if len(list_cr) == 0:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * 0.5
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * 0.5
        else:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * np.mean(list_cr)
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * self.lehmer_mean(list_f)

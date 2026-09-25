#!/usr/bin/env python
# Created by "Thieu" at 08:37, 17/06/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.stats import cauchy
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class L_SHADE(LegacyNativeOptimizer):
    """
    The original version of: Linear Population Size Reduction Success-History Adaptation Differential Evolution (LSHADE)

    Links:
        1. https://metahack.org/CEC2014-Tanabe-Fukunaga.pdf

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial weighting factor, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial cross-over probability, default = 0.5

    Examples
    ~~~~~~~~
    >>> from clypto.collection.evolutionary_based import SHADE    >>> import numpy as np
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
    >>> model = SHADE.L_SHADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Tanabe, R. and Fukunaga, A.S., 2014, July. Improving the search performance of SHADE using
    linear population size reduction. In 2014 IEEE congress on evolutionary computation (CEC) (pp. 1658-1665). IEEE.
    """

    cdef public object miu_f
    cdef public object miu_cr
    cdef public object dyn_miu_f
    cdef public object dyn_miu_cr
    cdef public object dyn_pop_archive
    cdef public object dyn_pop_size
    cdef public object k_counter
    cdef public object n_min

    def __init__(
        self,
        epoch: int = 750,
        pop_size: int = 100,
        miu_f: float = 0.5,
        miu_cr: float = 0.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial weighting factor, default = 0.5
            miu_cr (float): initial cross-over probability, default = 0.5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "miu_f", "miu_cr"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.miu_f = cy.validator(float, miu_f, (0, 1.0), "miu_f")
        self.miu_cr = cy.validator(float, miu_cr, (0, 1.0), "miu_cr")

    cdef void initialize_variables(self):
        self.dyn_miu_f = self.miu_f * np.ones(self.pop_size)  # memory of the successful f,
        self.dyn_miu_cr = self.miu_cr * np.ones(self.pop_size)  # memory of the successful cr,
        self.dyn_pop_archive = np.empty((0, self.problem.n_dims))
        self.dyn_pop_size = self.pop_size
        self.k_counter = 0
        self.n_min = int(self.pop_size / 5)

    def weighted_lehmer_mean(self, list_objects, list_weights):
        up = np.sum(list_weights * list_objects ** 2)
        down = np.sum(list_weights * list_objects)
        return up / down if down != 0 else 0.5

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = np.array(pop.X)
        me = np.arange(n)
        mem = rng.integers(0, n, size=n)
        cr = np.clip(rng.normal(self.dyn_miu_cr[mem], 0.1), 0, 1)
        f = self.dyn_miu_f[mem] + 0.1 * rng.standard_cauchy(n)
        while np.any(f < 0):
            bad = f < 0
            f[bad] = self.dyn_miu_f[mem][bad] + 0.1 * rng.standard_cauchy(int(bad.sum()))
        f = np.minimum(f, 1.0)
        top = np.ceil(self.dyn_pop_size * rng.uniform(0.15, 0.2, n)).astype(int)
        x_best = X[self.sorted_order(pop)][(rng.random(n) * top).astype(int)]
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
        pop = self.pop
        after = np.asarray(pop.F)
        improved = ops.better(self, after, before)
        archive = np.vstack([self.dyn_pop_archive, np.asarray(pop.X)[improved]])
        extra = len(archive) - n
        if extra > 0:
            archive = np.delete(archive, rng.choice(len(archive), extra, replace=False), axis=0)
        self.dyn_pop_archive = archive
        if improved.any():
            delta = np.abs(after[improved] - before[improved])
            total = delta.sum()
            w = np.zeros(len(delta)) if total == 0 else delta / total
            self.dyn_miu_cr[self.k_counter] = np.sum(w * cr[improved])
            self.dyn_miu_f[self.k_counter] = self.weighted_lehmer_mean(f[improved], w)
            self.k_counter += 1
            if self.k_counter >= self.dyn_pop_size:
                self.k_counter = 0
        self.dyn_pop_size = round(self.pop_size + epoch_c * ((self.n_min - self.pop_size) / self.epoch))

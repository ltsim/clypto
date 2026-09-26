#!/usr/bin/env python
# Created by "Thieu" at 13:59, 24/06/2021 ----------%
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


cdef class OriginalCOA(LegacyNativeOptimizer):
    """
    The original version of: Coyote Optimization Algorithm (COA)

    Links:
        1. https://ieeexplore.ieee.org/document/8477769
        2. https://github.com/jkpir/COA/blob/master/COA.py  (Old version Mealpy < 1.2.2)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_coyotes (int): [3, 15], number of coyotes per group, default=5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import COA    >>> import numpy as np
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
    >>> model = COA.OriginalCOA(epoch=1000, pop_size=50, n_coyotes = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Pierezan, J. and Coelho, L.D.S., 2018, July. Coyote optimization algorithm: a new metaheuristic
    for global optimization problems. In 2018 IEEE congress on evolutionary computation (CEC) (pp. 1-8). IEEE.
    """

    cdef public object n_coyotes
    cdef public object n_packs
    cdef public object pop_group
    cdef public object ps
    cdef public object p_leave

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_coyotes: int = 5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_coyotes (int): number of coyotes per group, default=5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "n_coyotes"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.n_coyotes = cy.validator(int, n_coyotes, [2, int(self.pop_size / 2)], "n_coyotes")
        self.n_packs = int(pop_size / self.n_coyotes)

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m):
        return [("AGE", 1)]  # age of every coyote

    cdef void init_fields(self, NativePopulation pop):
        pop.field("AGE")[:] = 1

    cdef void initialization(self):
        LegacyNativeOptimizer.initialization(self)
        self.ps = 1.0 / self.problem.n_dims
        self.p_leave = 0.005 * (self.n_coyotes**2)  # Probability of leaving a pack

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation pup
        cdef Py_ssize_t n = pop.n, d = pop.d, npk = self.n_packs, nc = self.n_coyotes, m = npk * nc
        cdef object rng = self.generator
        # every pack (a block of nc rows) is sorted by fitness, its best coyote leads
        F = np.asarray(pop.F)[:m].reshape(npk, nc)
        order = np.argsort(F, axis=1)
        if self.problem.minmax == "max":
            order = order[:, ::-1]
        pop = self.pop = pop.take(np.concatenate([(order + nc * np.arange(npk)[:, None]).ravel(), np.arange(m, n)]))
        X3 = np.array(pop.X[:m]).reshape(npk, nc, d)
        tendency = X3.mean(axis=(1, 2))[:, None, None]  # (the classic code takes the mean of all the elements of the pack)
        # social condition: towards the alpha and the pack tendency (Eq. 12)
        rc = np.stack([ops.others(self, nc, 2) for _ in range(npk)])  # (npk, nc, 2) random pack mates
        pack = np.arange(npk)[:, None]
        pos = X3 + rng.random((npk, nc, 1)) * (X3[:, :1] - X3[pack, rc[:, :, 0]]) + rng.random((npk, nc, 1)) * (tendency - X3[pack, rc[:, :, 1]])
        ops.step(self, pos.reshape(m, d), stop=m)
        # birth of a pup per pack from two random parents (Eq. 7)
        pop = self.pop
        X3 = np.array(pop.X[:m]).reshape(npk, nc, d)
        F = np.asarray(pop.F)[:m].reshape(npk, nc)
        dad = rng.integers(0, nc, size=npk)
        mom = (dad + rng.integers(1, nc, size=npk)) % nc
        prob1 = (1.0 - self.ps) / 2.0
        p = np.arange(npk)
        pups = rng.normal(0, 1, (npk, 1)) * np.where(rng.random((npk, d)) < prob1, X3[p, dad], X3[p, mom])
        pup = pop.take(np.arange(npk))
        pup.X[:] = self.correct_solution(pups)
        self.evaluate(pup, 0, npk)
        pup.field("AGE")[:] = 1
        worst = F.max(axis=1) if self.problem.minmax == "min" else F.min(axis=1)
        survive = ops.better(self, np.asarray(pup.F), worst)
        ages = np.asarray(pop.field("AGE"))[:m, 0].reshape(npk, nc)
        victim = ages.argmax(axis=1) if self.problem.minmax == "min" else ages.argmin(axis=1)  # the oldest (the youngest for max)
        rows = (p * nc + victim)[survive]
        pop.buf[rows] = pup.buf[np.flatnonzero(survive)]
        # a coyote can leave its pack and join another one (Eq. 4)
        if npk > 1 and rng.random() < self.p_leave:
            pk1, pk2 = rng.choice(npk, 2, replace=False)
            i1, i2 = rng.choice(nc, 2, replace=False)
            a, b = pk1 * nc + i1, pk2 * nc + i2
            tmp = np.array(pop.buf[a])
            pop.buf[a] = pop.buf[b]
            pop.buf[b] = tmp
        pop.field("AGE")[:m] += 1

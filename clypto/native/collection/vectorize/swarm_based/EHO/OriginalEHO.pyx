#!/usr/bin/env python
# Created by "Thieu" at 18:41, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalEHO(LegacyNativeOptimizer):
    """
    The original version of: Elephant Herding Optimization (EHO)

    Links:
        1. https://doi.org/10.1109/ISCBI.2015.8

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.3, 0.8], a factor that determines the influence of the best in each clan, default=0.5
        + beta (float): [0.3, 0.8], a factor that determines the influence of the x_center, default=0.5
        + n_clans (int): [3, 10], the number of clans, default=5

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import EHO    >>> import numpy as np
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
    >>> model = EHO.OriginalEHO(epoch=1000, pop_size=50, alpha = 0.5, beta = 0.5, n_clans = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2015, December. Elephant herding optimization.
    In 2015 3rd international symposium on computational and business intelligence (ISCBI) (pp. 1-5). IEEE.
    """

    cdef public object alpha
    cdef public object beta
    cdef public object n_clans
    cdef public object n_individuals
    cdef public object pop_group

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        alpha: float = 0.5,
        beta: float = 0.5,
        n_clans: int = 5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): a factor that determines the influence of the best in each clan, default=0.5
            beta (float): a factor that determines the influence of the x_center, default=0.5
            n_clans (int): the number of clans, default=5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "alpha", "beta", "n_clans"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.alpha = cy.validator(float, alpha, (0, 3.0), "alpha")
        self.beta = cy.validator(float, beta, (0, 1.0), "beta")
        self.n_clans = cy.validator(int, n_clans, [2, int(self.pop_size / 5)], "n_clans")
        self.n_individuals = int(self.pop_size / self.n_clans)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation fresh
        cdef Py_ssize_t n = pop.n, d = pop.d, nc = self.n_clans, ni = self.n_individuals, m = nc * ni  # (m <= n rows belong to a clan)
        cdef object rng = self.generator
        # clan updating operator: clans are consecutive blocks of rows, the first one of a block is its best
        X3 = np.array(pop.X[:m]).reshape(nc, ni, d)
        pos = np.empty_like(X3)
        pos[:, 0] = self.beta * X3.mean(axis=1)
        pos[:, 1:] = X3[:, 1:] + self.alpha * rng.random((nc, ni - 1, 1)) * (X3[:, :1] - X3[:, 1:])
        ops.step(self, pos.reshape(m, d), stop=m)
        # every clan is sorted by fitness and its worst member is replaced by a new random agent (separating operator)
        pop = self.pop
        F = np.asarray(pop.F)[:m].reshape(nc, ni)
        order = np.argsort(F, axis=1)
        if self.problem.minmax == "max":
            order = order[:, ::-1]
        pop = pop.take(np.concatenate([(order + ni * np.arange(nc)[:, None]).ravel(), np.arange(m, n)]))
        fresh = pop.take(ni * np.arange(1, nc + 1) - 1)
        fresh.X[:] = self.correct_solution(rng.uniform(self.problem.lb, self.problem.ub, (nc, d)))
        self.evaluate(fresh, 0, nc)
        pop.buf[ni * np.arange(1, nc + 1) - 1] = fresh.buf
        self.pop = pop

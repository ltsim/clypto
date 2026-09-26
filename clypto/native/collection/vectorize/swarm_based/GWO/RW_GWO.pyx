#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class RW_GWO(LegacyNativeOptimizer):
    """
    The original version of: Random Walk Grey Wolf Optimizer (RW-GWO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.RW_GWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Gupta, S. and Deep, K., 2019. A novel random walk grey wolf optimizer. Swarm and evolutionary computation, 44, pp.101-112.
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

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t n = pop.n, d = pop.d
        # linearly decreased from 2 to 0
        a = 2.0 - 2.0 * epoch / self.epoch
        order = self.sorted_order(pop)
        g_best = np.array(self.g_best_x())  # position of g_best before this epoch's updates
        leaders = pop.take(order[:3])

        ## Random walk of the three leaders (Cauchy steps drawn leader by leader)
        walk = np.array([leaders.X[k] + a * self.generator.standard_cauchy(d) for k in range(3)])
        leaders_new = self.new_population(self.correct_solution(walk))
        ops.accept(self, leaders_new, dst=leaders)

        ## Update other wolves (Eq. 3 and 4)
        R = self.generator.random((n, 6, d))  # per agent: miu1..3 then c1..3 draws
        miu = a * (2 * R[:, :3] - 1)
        c = 2 * R[:, 3:]
        Xs = leaders.X[None] - miu * np.abs(c * g_best - pop.X[:, None, :])
        cand.X[:] = self.correct_solution((Xs[:, 0] + Xs[:, 1] + Xs[:, 2]) / 3.0)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

        merged = pop.concat(leaders)
        self.pop = merged.take(self.sorted_order(merged)[:self.pop_size])

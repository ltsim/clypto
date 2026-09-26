#!/usr/bin/env python
# Created by "Thieu" at 17:22, 29/05/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevSSA(VectorizeOptimizer):
    """
    The developed version: Sparrow Search Algorithm (SSA)

    Notes:
        + First, the population is sorted to find g-best and g-worst
        + In Eq. 4, the self.generator.normal() gaussian distribution is used instead of A+ and L

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + ST (float): ST in [0.5, 1.0], safety threshold value, default = 0.8
        + PD (float): number of producers (percentage), default = 0.2
        + SD (float): number of sparrows who perceive the danger, default = 0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import SSA    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = SSA.DevSSA(epoch=1000, pop_size=50, ST = 0.8, PD = 0.2, SD = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Xue, J. and Shen, B., 2020. A novel swarm intelligence optimization approach:
    sparrow search algorithm. Systems Science & Control Engineering, 8(1), pp.22-34.
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        ST: float = 0.8,
        PD: float = 0.2,
        SD: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ST (float): ST in [0.5, 1.0], safety threshold value, default = 0.8
            PD (float): number of producers (percentage), default = 0.2
            SD (float): number of sparrows who perceive the danger, default = 0.1
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "ST", "PD", "SD"],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.ST = cy.validator(float, ST, (0, 1.0), "ST")
        self.PD = cy.validator(float, PD, (0, 1.0), "PD")
        self.SD = cy.validator(float, SD, (0, 1.0), "SD")
        self.n1 = int(self.PD * self.pop_size)
        self.n2 = int(self.SD * self.pop_size)

    cdef object _amend_solution(self, object solution):
        condition = np.logical_and(
            self.problem.bounds.low <= solution, solution <= self.problem.bounds.up
        )
        pos_rand = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up, size=np.shape(solution))
        return np.where(condition, solution, pos_rand)

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        me = np.arange(n)
        r2 = rng.uniform()  # R2 in [0, 1], the alarm value
        best_x = np.array(X[ops.best_row(self, self.pop)])
        F = np.asarray(pop.F)
        worst_x = np.array(X[int(F.argmax() if self.problem.sense == "min" else F.argmin())])
        # producers (the first n1 sparrows) and scroungers
        des = epoch_c / (rng.uniform(size=(n, 1)) * self.epoch + self.EPSILON)
        des = np.where(des > 5, rng.normal(size=(n, 1)), des)
        prod = np.where(r2 < self.ST, X * np.exp(des), X + rng.normal(size=(n, 1)))
        scr_far = np.broadcast_to(np.where((me > int(n / 2))[:, None], rng.normal(size=(n, 1)) * np.exp((worst_x - X) / ((me + 1) ** 2)[:, None]), 0.0), (n, d))
        scr_near = best_x + np.abs(X - best_x) * rng.normal(size=(n, 1))
        pos = np.where((me < self.n1)[:, None], prod, np.where((me > int(n / 2))[:, None], scr_far, scr_near))
        ops.step(self, pos)
        # the population is sorted, the last sparrows move around the best (or away from the worst)
        pop = self.pop = self.pop.take(self.sorted_order(self.pop))
        X = pop.X
        F = np.asarray(pop.F)
        best_x = np.array(X[0])
        n2, m2 = self.n2, n - self.n2
        worst_x = np.array(X[n - 1])
        gb_fit, gw_fit = F[0], F[n - 1]
        X2, F2 = X[n2:], F[n2:]
        risky = ops.better(self, F[:m2], gb_fit)[:, None]
        pos2 = np.where(risky, X2 + rng.uniform(-1, 1, (m2, 1)) * (np.abs(X2 - worst_x) / (F2 - gw_fit + self.EPSILON)[:, None]),
                        best_x + rng.normal(size=(m2, 1)) * np.abs(X2 - best_x))
        ops.step(self, pos2, start=n2, stop=n)

#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class CG_GWO(LegacyNativeOptimizer):
    """
    The original version of: Cauchy‑Gaussian mutation and improved search strategy GWO (CG‑GWO)

    Notes:
        + This algorithm can't be parallelized because of the 'single' update mode.
        + Meaning that the updating of the pack is based on order and sequence of the wolves.

    Links:
        1. https://doi.org/10.1038/s41598-022-23713-9

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.CG_GWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Li, K., Li, S., Huang, Z. et al. Grey Wolf Optimization algorithm based on Cauchy-Gaussian mutation and improved search strategy. Sci Rep 12, 18961 (2022).
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
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def cauchy_gaussian_mutation(self, best_fit, leader_fit, leader_pos, epoch):
        # Calculate dynamic parameters (equations 11 and 12)
        eps2 = (epoch / self.epoch) ** 2
        eps1 = 1 - eps2
        # Calculate sigma (equation 9)
        if abs(best_fit) > 1e-10:
            sigma = np.exp((leader_fit - best_fit) / abs(best_fit))
        else:
            sigma = 1.0
        # Generate Cauchy and Gaussian random variables
        c_rand = self.generator.standard_cauchy(size=self.problem.n_dims) * sigma**2 + 0
        g_rand = self.generator.normal(loc=0, scale=sigma**2, size=self.problem.n_dims)
        # Apply mutation (equation 8)
        return leader_pos * (1 + eps1 * c_rand + eps2 * g_rand)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand, sub
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        lb, ub = self.problem.lb, self.problem.ub
        a = 2 - 2.0 * epoch_c / self.epoch  # linearly decreased from 2 to 0
        order = self.sorted_order(pop)[:3]
        best_pos = np.array(X[order])
        best_fit = np.array(pop.F[order])
        # Cauchy-Gaussian mutation of the three leaders, then greedy selection
        lead = pop.take(order)
        lead.X[:] = self.correct_solution(np.array([self.cauchy_gaussian_mutation(best_fit[0], best_fit[k], best_pos[k], epoch_c) for k in range(3)]))
        self.evaluate(lead, 0, 3)
        win = ops.better(self, lead.F, best_fit)
        best_pos[win] = lead.X[win]
        # improved search strategy (equation 13): around a random wolf or the alpha wolf
        R = rng.random((n, 5, 1))
        x_rand = X[ops.others(self, n)[:, 0]]
        x_avg = np.mean(np.ascontiguousarray(X), axis=0)
        pos_e = np.where(R[:, 4] >= 0.5, x_rand - R[:, 0] * np.abs(x_rand - 2 * R[:, 1] * X),
                         (best_pos[0] - x_avg) - R[:, 2] * (lb + R[:, 3] * (ub - lb)))
        cand = pop.empty_like()
        cand.X[:] = self.correct_solution(pos_e)
        self.evaluate(cand, 0, n)
        # where it is not an improvement: the original GWO update
        fail = np.flatnonzero(ops.better(self, pop.F, cand.F))
        if len(fail):
            m = len(fail)
            G = rng.random((m, 6, d))
            Xf = X[fail][:, None, :]
            Xs = best_pos[None] - (a * (2 * G[:, :3] - 1)) * np.abs(2 * G[:, 3:] * best_pos[None] - Xf)
            sub = pop.take(fail)
            sub.X[:] = self.correct_solution(Xs.sum(axis=1) / 3.0)
            self.evaluate(sub, 0, m)
            cand.buf[fail] = sub.buf
        ops.greedy(self, cand)

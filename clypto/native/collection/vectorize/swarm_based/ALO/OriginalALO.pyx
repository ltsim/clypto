#!/usr/bin/env python
# Created by "Thieu" at 12:01, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalALO(LegacyNativeOptimizer):
    """
    The original version of: Ant Lion Optimizer (ALO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/49920-ant-lion-optimizer-alo
        2. https://dx.doi.org/10.1016/j.advengsoft.2015.01.010

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import ALO    >>> import numpy as np
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
    >>> model = ALO.OriginalALO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2015. The ant lion optimizer. Advances in engineering software, 83, pp.80-98.
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
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def random_walk_antlion__(self, solution, current_epoch, steps, column):
        """Bounded random walks of every agent around ``solution`` (n, d): the value of step ``column`` of a walk with ``steps`` steps."""
        n, d = solution.shape
        I = 1  # I is the ratio in Equations (2.10) and (2.11)
        if current_epoch > self.epoch / 10:
            I = 1 + 100 * (current_epoch / self.epoch)
        if current_epoch > self.epoch / 2:
            I = 1 + 1000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * (3 / 4):
            I = 1 + 10000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.9:
            I = 1 + 100000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.95:
            I = 1 + 1000000 * (current_epoch / self.epoch)
        rng = self.generator
        # Decrease boundaries to converge towards antlion (Eq. 2.10), move the interval around it (Eqs. 2.8, 2.9)
        sign = np.where(rng.random((n, 2, 1)) < 0.5, 1.0, -1.0)
        lb = sign[:, 0] * (self.problem.lb / I) + solution
        ub = sign[:, 1] * (self.problem.ub / I) + solution
        out = np.empty((n, d))
        block = max(1, 4000000 // max(1, d * steps))  # agents per block: bounded (block, d, steps) walks
        for i0 in range(0, n, block):
            i1 = min(n, i0 + block)
            X = np.cumsum(2 * (rng.random((i1 - i0, d, steps)) > 0.5) - 1, axis=2, dtype=np.int32)
            a = X.min(axis=2)
            b = X.max(axis=2)
            out[i0:i1] = ((X[:, :, column] - a) * (ub[i0:i1] - lb[i0:i1])) / (b - a) + lb[i0:i1]  # Eq. (2.7)
        return out

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand, both
        cdef Py_ssize_t n = pop.n
        cdef object rng = self.generator
        X = pop.X
        elite = self.current_g_best()
        elite_x = np.array(elite.solution)
        fits = np.array(pop.F)
        # Select ant lions based on their fitness (the better antlion the higher chance of catching an ant)
        if np.ptp(fits) == 0:
            selected = rng.integers(0, n, size=n)
        else:
            f = fits - fits.min() if np.any(fits < 0) else fits
            f = f.max() - f if self.problem.minmax == "min" else f
            selected = rng.choice(n, size=n, p=f / f.sum())
        RA = self.random_walk_antlion__(X[selected], epoch, self.epoch, epoch - 1)
        RE = self.random_walk_antlion__(np.broadcast_to(elite_x, (n, pop.d)), epoch, self.epoch, epoch - 1)
        cand = pop.empty_like()
        cand.X[:] = self.correct_solution((RA + RE) / 2)  # Equation(2.13)
        self.evaluate(cand, 0, n)
        # an ant fitter than an antlion is caught by it: the antlion moves to its position
        both = pop.concat(cand)
        pop = self.pop = both.take(self.sorted_order(both)[:n])
        # Keep the elite in the population
        ops.set_row(pop, n - 1, elite_x, elite.target)

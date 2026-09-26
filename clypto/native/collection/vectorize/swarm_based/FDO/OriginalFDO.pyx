#!/usr/bin/env python
# Created by "Thieu" at 10:01, 16/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalFDO(VectorizeOptimizer):
    """
    The original version of: Fitness Dependent Optimizer (FDO)

    Notes:
        + https://doi.org/10.1109/ACCESS.2019.2907012
        + Inspired by the bee swarming reproductive process, this algorithm optimizes solutions based on their fitness values.
        + This algorithm mainly relies on Lévy flight techniques. Thanks to this method of generating random numbers
        according to the Lévy distribution, it is able to converge. However, in the design of the fitness weight
        condition, it is almost impossible for an update to occur when the fitness weight equals 1. This is the main drawback.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import FDO    >>> import numpy as np
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
    >>> model = FDO.OriginalFDO(epoch=1000, pop_size=50, weight_factor=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdullah, J. M., & Ahmed, T. (2019).
    Fitness dependent optimizer: inspired by the bee swarming reproductive process. IEEe Access, 7, 43473-43486.
    """

    cdef public object weight_factor
    cdef public object pop_pace

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        weight_factor = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            weight_factor (float): factor to adjust the fitness weight calculation, default = 0.1
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "weight_factor"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.weight_factor = cy.validator(float, weight_factor, [0.0, 1.0], "weight_factor")

    def get_fit_weight(self, best_fit, current_fit, weight_factor=0.1):
        """Fitness weight of every agent (vectorized over ``current_fit``)."""
        current_fit = np.asarray(current_fit, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = best_fit / current_fit
        if self.problem.sense == "min":
            fw = np.where(best_fit < 0.05 * current_fit, 0.2, ratio - weight_factor)
        else:
            fw = np.where(best_fit > 0.05 * current_fit, 0.2, weight_factor - ratio)
        return np.zeros_like(current_fit) if best_fit == 0 else fw

    def get_into_levy_bound(self, pos_new):
        levy = self._get_levy_flight_step(beta=1.5, multiplier=0.01, size=pos_new.shape, case=-1)
        return np.select(
            [pos_new > self.problem.bounds.up, pos_new < self.problem.bounds.low],
            [self.problem.bounds.up * np.abs(levy), self.problem.bounds.low * np.abs(levy)],
            default=pos_new,
        )

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        X = np.array(pop.X)
        g = np.array(self.g_best_x())
        gb_fit = self.current_g_best().target.fitness
        fw = self.get_fit_weight(gb_fit, np.asarray(pop.F), self.weight_factor)[:, None]
        dist = g - X
        levy = self._get_levy_flight_step(beta=1.5, multiplier=0.01, size=(n, d), case=-1)
        pace = np.where(fw == 1, X * levy, np.where(fw == 0, dist * levy, dist * fw * np.sign(levy)))
        # three attempts per agent, each one only for the agents the previous attempt did not improve
        pos1 = self._correct_solution(self.get_into_levy_bound(X + pace))
        F0 = np.array(pop.F)
        ops.step(self, pos1)
        todo = np.flatnonzero(~ops.better(self, np.asarray(self.pop.F), F0))
        if len(todo):
            pos2 = pos1[todo] + (g - pos1[todo]) * fw[todo] + pace[todo]
            cand = self.pop.take(todo)
            cand.X[:] = self._correct_solution(self.get_into_levy_bound(pos2))
            self.evaluate(cand, 0, len(todo))
            F1 = np.array(self.pop.F)
            ops.scatter(self, cand, todo)
            todo = todo[~ops.better(self, np.asarray(self.pop.F)[todo], F1[todo])]
        if len(todo):
            Xt = np.array(self.pop.X[todo])
            levy = self._get_levy_flight_step(beta=1.5, multiplier=0.01, size=(len(todo), d), case=-1)
            cand = self.pop.take(todo)
            cand.X[:] = self._correct_solution(self.get_into_levy_bound(Xt + Xt * levy))
            self.evaluate(cand, 0, len(todo))
            ops.scatter(self, cand, todo)

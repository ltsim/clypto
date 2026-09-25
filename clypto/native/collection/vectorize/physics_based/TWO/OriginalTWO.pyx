#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np



from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalTWO(LegacyNativeOptimizer):
    """
    The original version of: Tug of War Optimization (TWO)

    Links:
        1. https://www.researchgate.net/publication/332088054_Tug_of_War_Optimization_Algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import TWO    >>> import numpy as np
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
    >>> model = TWO.OriginalTWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kaveh, A., 2017. Tug of war optimization. In Advances in metaheuristic algorithms for
    optimal design of structures (pp. 451-487). Springer, Cham.
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
        self.muy_s = 1
        self.muy_k = 1
        self.delta_t = 1
        self.alpha = 0.99
        self.beta = 0.1

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m):
        return [("W", 1)]  # weight of each team

    cdef void init_fields(self, NativePopulation pop):
        pop.field("W")[:] = 0.0

    cdef void initialization(self):
        LegacyNativeOptimizer.initialization(self)
        self.update_weight__(self.pop)

    def update_weight__(self, NativePopulation teams):
        list_fits = np.array(teams.F)
        maxx, minn = np.max(list_fits), np.min(list_fits)
        if maxx == minn:
            list_fits = self.generator.uniform(0.0, 1.0, self.pop_size)
        list_weights = np.exp(-(list_fits - maxx) / (maxx - minn))
        list_weights = list_weights / np.sum(list_weights) + 0.1
        teams.field("W")[:self.pop_size, 0] = list_weights
        return teams

    def forces__(self, NativePopulation pop, epoch):
        """Loop 1: the teams pull each other; positions are updated in place, one team after the other
        (the classic candidate list shares its agents with the population)."""
        cdef Py_ssize_t idx, jdx
        lb, ub = self.problem.lb, self.problem.ub
        Xp, W = pop.X, pop.field("W")[:, 0]
        for idx in range(self.pop_size):
            pos_new = Xp[idx].copy().astype(float)
            for jdx in range(self.pop_size):
                if W[idx] < W[jdx]:
                    force = max(W[idx] * self.muy_s, W[jdx] * self.muy_s)
                    resultant_force = force - W[idx] * self.muy_k
                    g = Xp[jdx] - Xp[idx]
                    acceleration = resultant_force * g / (W[idx] * self.muy_k)
                    delta_x = 0.5 * acceleration + np.power(self.alpha, epoch) * self.beta * (ub - lb) * self.generator.normal(0, 1, pop.d)
                    pos_new += delta_x
            Xp[idx] = pos_new

    def bound__(self, NativePopulation pop, Py_ssize_t idx, epoch):
        """Loop 2 body: the classic bound handling of team ``idx`` (returns the uncorrected position)."""
        cdef Py_ssize_t jdx
        lb, ub = self.problem.lb, self.problem.ub
        Xp = pop.X
        pos_new = Xp[idx].copy().astype(float)
        for jdx in range(pop.d):
            if pos_new[jdx] < lb[jdx] or pos_new[jdx] > ub[jdx]:
                if self.generator.random() <= 0.5:
                    g_best = self.g_best_x()  # aliased to the best row: sees the updates made so far
                    pos_new[jdx] = g_best[jdx] + self.generator.standard_normal() / epoch * (g_best[jdx] - pos_new[jdx])
                    if pos_new[jdx] < lb[jdx] or pos_new[jdx] > ub[jdx]:
                        pos_new[jdx] = Xp[idx][jdx]
                else:
                    if pos_new[jdx] < lb[jdx]:
                        pos_new[jdx] = lb[jdx]
                    if pos_new[jdx] > ub[jdx]:
                        pos_new[jdx] = ub[jdx]
        return pos_new

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativeTarget tar
        cdef Py_ssize_t idx
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        self.forces__(pop, epoch)
        for idx in range(self.pop_size):
            pos_new = self.correct_solution(self.bound__(pop, idx, epoch))
            if swarm:
                pop.X[idx] = pos_new
            else:
                # the candidate is the population's own agent: it is always replaced
                ops.set_row(pop, idx, pos_new, self.get_target(pos_new))
        if swarm:
            self.evaluate(pop, 0, pop.n)
        self.update_weight__(pop)

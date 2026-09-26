#!/usr/bin/env python
# Created by "Thieu" at 16:10, 08/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalArchOA(LegacyNativeOptimizer):
    """
    The original version of: Archimedes Optimization Algorithm (ArchOA)

    Links:
        1. https://doi.org/10.1007/s10489-020-01893-z

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (int): factor, default belongs to [1, 2]
        + c2 (int): factor, Default belongs to [2, 4, 6]
        + c3 (int): factor, Default belongs to [1, 2]
        + c4 (float): factor, Default belongs to [0.5, 1]
        + acc_max (float): acceleration max, Default 0.9
        + acc_min (float): acceleration min, Default 0.1

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import ArchOA    >>> import numpy as np
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
    >>> model = ArchOA.OriginalArchOA(epoch=1000, pop_size=50, c1 = 2, c2 = 5, c3 = 2, c4 = 0.5, acc_max = 0.9, acc_min = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F.A., Hussain, K., Houssein, E.H., Mabrouk, M.S. and Al-Atabany, W., 2021. Archimedes optimization
    algorithm: a new metaheuristic algorithm for solving optimization problems. Applied Intelligence, 51(3), pp.1531-1551.
    """

    cdef public double c1
    cdef public double c2
    cdef public double c3
    cdef public double c4
    cdef public double acc_max
    cdef public double acc_min

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 2,
        c2: float = 6,
        c3: float = 2,
        c4: float = 0.5,
        acc_max: float = 0.9,
        acc_min: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): factor, default belongs [1, 2]
            c2 (float): factor, Default belongs [2, 4, 6]
            c3 (float): factor, Default belongs [1, 2]
            c4 (float): factor, Default belongs [0.5, 1]
            acc_max (float): acceleration max, Default 0.9
            acc_min (float): acceleration min, Default 0.1
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "c1", "c2", "c3", "c4", "acc_max", "acc_min"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.c1 = cy.validator(float, c1, [1, 3], "c1")
        self.c2 = cy.validator(float, c2, [2, 6], "c2")
        self.c3 = cy.validator(float, c3, [1, 3], "c3")
        self.c4 = cy.validator(float, c4, (0, 1.0), "c4")
        self.acc_max = cy.validator(float, acc_max, (0.3, 1.0), "acc_max")
        self.acc_min = cy.validator(float, acc_min, (0, 0.3), "acc_min")

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m):
        return [("DEN", d), ("VOL", d), ("ACC", d)]  # density, volume, acceleration

    cdef void init_fields(self, NativePopulation pop):
        lb, ub = self.problem.lb, self.problem.ub
        R = self.generator.random((pop.n, 3, pop.d))  # per agent: den, vol, acc draws
        pop.field("DEN")[:] = lb + (ub - lb) * R[:, 0]  # Density
        pop.field("VOL")[:] = lb + (ub - lb) * R[:, 1]  # Volume
        pop.field("ACC")[:] = lb + (lb + (ub - lb) * R[:, 2]) * (ub - lb)  # Acceleration

    cdef void evolve(self, int epoch):
        # Densities and volumes are updated in place and read by the agents after them, so the
        # first loop runs on the buffer rows; in sequential mode so does the position loop.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef NativeTarget tar
        cdef Py_ssize_t idx, b, n = pop.n
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        DEN, VOL, ACC = pop.field("DEN"), pop.field("VOL"), pop.field("ACC")
        # g_best: an alias of the best row (its density/volume change in place), a copy in epoch 1
        b = self._g_best_row
        if b >= 0:
            g_den, g_vol, g_acc = DEN[b], VOL[b], ACC[b]
        else:
            b = self.sorted_order(pop)[0]
            g_den, g_vol, g_acc = DEN[b].copy(), VOL[b].copy(), ACC[b].copy()
        g_best = np.array(self.g_best_x())
        ## Transfer operator Eq. 8
        tf = np.exp(epoch / self.epoch)
        ## Density decreasing factor Eq. 9
        ddf = np.exp(1.0 - epoch / self.epoch) - epoch / self.epoch
        list_acc = []
        ## Calculate new density, volume and acceleration
        for idx in range(0, self.pop_size):
            # Update density and volume of each object using Eq. 7
            new_den = DEN[idx] + self.generator.uniform() * (g_den - DEN[idx])
            new_vol = VOL[idx] + self.generator.uniform() * (g_vol - VOL[idx])
            # Exploration phase
            if tf <= 0.5:
                # Update acceleration using Eq. 10 and normalize acceleration using Eq. 12
                id_rand = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                new_acc = (DEN[id_rand] + VOL[id_rand] * ACC[id_rand]) / (new_den * new_vol)
            else:
                new_acc = (g_den + g_vol * g_acc) / (new_den * new_vol)
            list_acc.append(new_acc)
            DEN[idx] = new_den
            VOL[idx] = new_vol
        min_acc = np.min(list_acc)
        max_acc = np.max(list_acc)
        ## Normalize acceleration using Eq. 12
        for idx in range(0, self.pop_size):
            ACC[idx] = (
                self.acc_max
                * (list_acc[idx] - min_acc)
                / (max_acc - min_acc + self.EPSILON)
                + self.acc_min
            )
        cand.buf[:] = pop.buf  # candidates carry the agent's fields
        for idx in range(0, self.pop_size):
            if tf <= 0.5:  # update position using Eq. 13
                id_rand = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                pos_new = Xp[idx] + self.c1 * self.generator.uniform() * ACC[idx] * ddf * (Xp[id_rand] - Xp[idx])
            else:
                p = 2 * self.generator.random() - self.c4
                f = 1 if p <= 0.5 else -1
                t = self.c3 * tf
                pos_new = g_best + f * self.c2 * self.generator.random() * ACC[idx] * ddf * (t * g_best - Xp[idx])
            pos_c = self.correct_solution(pos_new)
            if swarm:
                cand.X[idx] = pos_c
            else:
                # the classic sequential path evaluates the *uncorrected* position
                tar = self.get_target(pos_new)
                if self.compare_fitness(tar.fitness, pop.F[idx], self.problem.minmax):
                    ops.set_row(pop, idx, pos_c, tar)
        if swarm:
            ops.finish(self, cand, 0, n)

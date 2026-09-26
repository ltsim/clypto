#!/usr/bin/env python
# Created by "Thieu" at 16:31, 13/09/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalMSO(LegacyNativeOptimizer):
    """
    The original version of: Mirage Search Optimization (MSO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/180042-mirage-search-optimization

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.physics_based import MSO    >>> import numpy as np
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
    >>> model = MSO.OriginalMSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] He, J., Zhao, S., Ding, J., & Wang, Y. (2025). Mirage search optimization: Application to
    path planning and engineering design problems. Advances in Engineering Software, 203, 103883.
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
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")

    def sind(self, x):
        return np.sin(np.deg2rad(x))

    def cosd(self, x):
        return np.cos(np.deg2rad(x))

    def tand(self, x):
        x = np.asarray(x, dtype=float)
        bad = (np.mod(x, 90) == 0) & (np.mod(x, 180) != 0)
        x[bad] = x[bad] - self.EPSILON
        return np.tan(np.deg2rad(x))

    def atand(self, x):
        return np.rad2deg(np.arctan(x))

    def asind(self, x):
        x = np.clip(x, -1, 1)  # asin bound [-1, 1]
        val = np.rad2deg(np.arcsin(x))
        return val

    def atanh(self, x):
        if np.abs(x) >= 1:
            return 1.0
        return np.arctanh(x)

    cdef void evolve(self, int epoch_c):
        # The classic code reads the evaluation counter while it builds the candidates (every
        # agent is evaluated as soon as it is built), so each agent is evaluated inline.
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation new, merged
        cdef NativeTarget tar
        cdef Py_ssize_t i, idx, k, n = pop.n, d = pop.d
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        # Random permutation for agent selection
        ac = self.generator.permutation(self.pop_size - 1) + 1
        # Selection of individuals for Superior mirage search
        cv = int(np.ceil((self.pop_size * (2 / 3)) * ((self.epoch - self.nf_counter + 1) / self.epoch)))

        # Superior mirage search
        chosen = ac[:cv]
        new = pop.take(np.zeros(len(chosen), dtype=int))
        for i in range(len(chosen)):
            idx = chosen[i]
            pos_new = np.zeros(d)
            for k in range(d):
                h = (g_best[k] - Xp[idx][k]) * self.generator.random()
                cmax = 1
                hmax = 5 * self.atanh(-(self.nf_counter / self.epoch) + 1) + cmax
                if h > hmax:
                    h = hmax
                if h < cmax:
                    h = cmax
                zf = self.generator.choice([-1, 1])
                a = self.generator.random() * 20
                b = self.generator.random() * (45 - a / 2)
                z = self.generator.integers(1, 3)
                A = B = C = D = 90
                if z == 1:
                    C = b + 90
                    D = 180 - C - a
                    B = 180 - 2 * D
                    A = 180 - B + a - 90
                elif z == 2 and a < b:
                    C = 90 - b
                    D = 90 + a - b
                    B = 180 - 2 * D
                    A = 180 - B - a - 90
                elif z == 2 and a > b:
                    C = 90 - b
                    D = 180 - C - a
                    B = 180 - 2 * D
                    A = 180 - B - 90 + a
                else:
                    zf = 0
                dx = (self.sind(B) * h * self.sind(C)) / (self.sind(D) * self.sind(A))
                dx = dx * zf
                pos_new[k] = Xp[idx][k] + dx
            # Bound the variables
            pos_new = self.correct_solution(pos_new)
            ops.set_row(new, i, pos_new, self.get_target(pos_new))
        merged = pop.concat(new)
        pop = merged.take(self.sorted_order(merged)[:self.pop_size])
        self.pop = pop
        Xp = pop.X

        # Inferior mirage search
        new = pop.empty_like()
        for idx in range(self.pop_size):
            if np.allclose(g_best, Xp[idx], atol=1e-6):
                hh = np.ones(d) * 0.05 * self.generator.choice([-1, 1])
            else:
                hh = g_best - Xp[idx]
            zf = np.sign(hh)
            hh = np.abs(hh * self.generator.random(d))
            gama = self.generator.random(d) * 90 * ((self.epoch - self.nf_counter * 0.99) / self.epoch)
            amax = self.atand(1.0 / (2 * self.tand(gama)))
            amin = self.atand((self.sind(gama) * self.cosd(gama)) / (1 + (self.sind(gama)) ** 2))
            fai = (amax - amin) * self.generator.random() + amin
            omg = self.asind(self.generator.random() * self.sind(fai + gama))
            x = (hh / self.tand(gama)) - (
                    (
                            (hh / self.sind(gama))
                            - (hh * self.sind(fai)) / (self.cosd(fai + gama))
                    )
                    * self.cosd(omg)
            ) / self.cosd(omg - gama)
            pos_new = self.correct_solution(Xp[idx] + x * zf)
            ops.set_row(new, idx, pos_new, self.get_target(pos_new))
        merged = pop.concat(new)
        self.pop = merged.take(self.sorted_order(merged)[:self.pop_size])

#!/usr/bin/env python
# Created by "Thieu" at 17:29, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalINFO(LegacyNativeOptimizer):
    """
    The original version of: weIghted meaN oF vectOrs (INFO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0957417422000173
        2. https://aliasgharheidari.com/INFO.html
        3. https://doi.org/10.1016/j.eswa.2022.116516

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.math_based import INFO    >>> import numpy as np
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
    >>> model = INFO.OriginalINFO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ahmadianfar, I., Heidari, A. A., Noshadian, S., Chen, H., & Gandomi, A. H. (2022). INFO: An efficient optimization
    algorithm based on weighted mean of vectors. Expert Systems with Applications, 195, 116516.
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

    cdef void evolve(self, int epoch):
        # A different number of draws per agent depending on the branches, so candidates
        # are built agent by agent (same draw order); the old population is only read.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        alpha = 2 * np.exp(-4 * (epoch / self.epoch))  # Eqs.(5.1) - Eq.(9.1)
        idx_better = self.generator.integers(2, 6)
        Xp, Xc = pop.X, cand.X
        gb = self.current_g_best()
        g_best_pos, g_best_fit = gb.solution, gb.target.fitness

        for idx in range(0, self.pop_size):
            ## Updating rule stage
            delta = 2 * self.generator.random() * alpha - alpha  # Eq. (5)
            sigma = 2 * self.generator.random() * alpha - alpha  # Eq. (9)
            ## Select three random solution
            a, b, c = self.generator.choice(range(0, self.pop_size), 3, replace=False)
            e1 = 1e-25
            epsilon = e1 * self.generator.random()
            fit_a = pop.F[a]
            fit_b = pop.F[b]
            fit_c = pop.F[c]
            omg1 = np.max([fit_a, fit_b, fit_c])
            MM1 = np.array([fit_a - fit_b, fit_a - fit_c, fit_b - fit_c])

            w1 = np.cos(MM1[0] + np.pi) * np.exp(-np.abs(MM1[0] / omg1))  # Eq. (4.2)
            w2 = np.cos(MM1[1] + np.pi) * np.exp(-np.abs(MM1[1] / omg1))  # Eq. (4.3)
            w3 = np.cos(MM1[2] + np.pi) * np.exp(-np.abs(MM1[2] / omg1))  # Eq. (4.4)
            Wt1 = np.sum([w1, w2, w3])
            WM1 = (
                    delta
                    * (
                            w1 * (Xp[a] - Xp[b])  # Eq.(4.1)
                            + w2 * (Xp[a] - Xp[c])
                            + w3 * (Xp[b] - Xp[c])
                    )
                    / (Wt1 + 1)
                    + epsilon
            )

            fit_1 = g_best_fit
            fit_2 = pop.F[idx_better]
            fit_3 = pop.F[n - 1]
            omg2 = np.max([fit_1, fit_2, fit_3])
            MM2 = np.array([fit_1 - fit_2, fit_1 - fit_3, fit_2 - fit_3])
            w4 = np.cos(MM2[0] + np.pi) * np.exp(-np.abs(MM2[0] / omg2))  # Eq. (4.7)
            w5 = np.cos(MM2[1] + np.pi) * np.exp(-np.abs(MM2[1] / omg2))  # Eq. (4.8)
            w6 = np.cos(MM2[2] + np.pi) * np.exp(-np.abs(MM2[2] / omg2))  # Eq. (4.9)
            Wt2 = np.sum([w4, w5, w6])
            WM2 = (
                    delta
                    * (
                            w4 * (g_best_pos - Xp[idx_better])  # Eq. (4.6)
                            + w5 * (g_best_pos - Xp[n - 1])
                            + w6 * (Xp[idx_better] - Xp[n - 1])
                    )
                    / (Wt2 + 1)
                    + epsilon
            )
            ## Determine MeanRule
            r = self.generator.uniform(0.1, 0.5)
            mean_rule = r * WM1 + (1 - r) * WM2  # Eq. (4)
            if self.generator.random() < 0.5:  # Eq. (8)
                z1 = (
                        Xp[idx]
                        + sigma * (self.generator.random() * mean_rule)
                        + self.generator.random()
                        * (g_best_pos - Xp[a])
                        / (fit_1 - fit_a + 1)
                )
                z2 = (
                        g_best_pos
                        + sigma * (self.generator.random() * mean_rule)
                        + self.generator.random()
                        * (Xp[a] - Xp[b])
                        / (fit_a - fit_b + 1)
                )
            else:
                z1 = (
                        Xp[a]
                        + sigma * (self.generator.random() * mean_rule)
                        + self.generator.random()
                        * (Xp[b] - Xp[c])
                        / (fit_b - fit_c + 1)
                )
                z2 = (
                        Xp[idx_better]
                        + sigma * (self.generator.random() * mean_rule)
                        + self.generator.random()
                        * (Xp[a] - Xp[b])
                        / (fit_a - fit_b + 1)
                )
            ## Vector combining stage
            mu = 0.05 * self.generator.random(d)
            u1 = z1 + mu * np.abs(z1 - z2)  # Eq. (10.1)
            u2 = z2 + mu * np.abs(z1 - z2)  # Eq. (10.2)
            cond1 = self.generator.random(d) < 0.05
            cond2 = self.generator.random(d) < 0.05
            x1 = np.where(cond1, u1, u2)
            pos_new = np.where(cond2, x1, Xp[idx])  # Eq. (10.3)
            ## Local search stage
            if self.generator.random() < 0.5:
                L = int(self.generator.random() < 0.5)  # 0 or 1
                v1 = (1 - L) * 2 * self.generator.random() + L  # Eqs. (11.5)
                v2 = self.generator.random() * L + (1 - L)  # Eq. (11.6)
                x_avg = (
                                Xp[a] + Xp[b] + Xp[c]
                        ) / 3  # Eq. (11.4)
                phi = self.generator.random()
                x_rand = phi * x_avg + (1 - phi) * (
                        phi * Xp[idx_better] + (1 - phi) * g_best_pos
                )  # Eq. (11.3)
                n_rand = (
                        L * self.generator.random(d)
                        + (1 - L) * self.generator.random()
                )
                if self.generator.random() < 0.5:  # Eq. (11.1)
                    pos_new = g_best_pos + n_rand * (
                            mean_rule
                            + self.generator.random()
                            * (g_best_pos - Xp[a])
                    )
                else:  # Eq. (11.2)
                    pos_new = x_rand + n_rand * (
                            mean_rule
                            + self.generator.random()
                            * (v1 * g_best_pos - v2 * x_rand)
                    )
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        self.pop = cand

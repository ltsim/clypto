#!/usr/bin/env python
# Created by "Thieu" at 17:07, 02/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalGBO(VectorizeOptimizer):
    """
    The original version of: Gradient-Based Optimizer (GBO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pr (float): [0.2, 0.8], Probability Parameter, default = 0.5
        + beta_min (float): Fixed parameter (no name in the paper), default = 0.2
        + beta_max (float): Fixed parameter (no name in the paper), default = 1.2

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.math_based import GBO    >>> import numpy as np
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
    >>> model = GBO.OriginalGBO(epoch=1000, pop_size=50, pr = 0.5, beta_min = 0.2, beta_max = 1.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ahmadianfar, I., Bozorg-Haddad, O. and Chu, X., 2020. Gradient-based optimizer:
    A new metaheuristic optimization algorithm. Information Sciences, 540, pp.131-159.
    """

    cdef public double pr
    cdef public double beta_min
    cdef public double beta_max

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pr: float = 0.5,
        beta_min: float = 0.2,
        beta_max: float = 1.2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pr (float): Probability Parameter, default = 0.5
            beta_min (float): Fixed parameter (no name in the paper), default = 0.2
            beta_max (float): Fixed parameter (no name in the paper), default = 1.2
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pr", "beta_min", "beta_max"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pr = cy.validator(float, pr, (0, 1.0), "pr")
        self.beta_min = cy.validator(float, beta_min, (0, 2.0), "beta_min")
        self.beta_max = cy.validator(float, beta_max, (0, 5.0), "beta_max")

    def _evolve(self, int epoch):
        # Agents read the population they just updated (random members), so in sequential
        # mode the loop runs on the buffer rows; swarm/parallel modes batch the evaluation.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef NativeTarget tar
        cdef Py_ssize_t idx
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        g_worst = self.g_worst.solution
        # Eq.(14.2), Eq.(14.1)
        beta = (
                self.beta_min
                + (self.beta_max - self.beta_min) * (1 - (<object>(epoch / self.epoch)) ** 3) ** 2
        )
        alpha = np.abs(beta * np.sin(3 * np.pi / 2 + np.sin(beta * 3 * np.pi / 2)))

        for idx in range(0, self.pop_size):
            p1 = 2 * self.generator.random() * alpha - alpha
            p2 = 2 * self.generator.random() * alpha - alpha
            #  Four positions randomly selected from population
            r1, r2, r3, r4 = self.generator.choice(
                list(set(range(0, self.pop_size)) - {idx}), 4, replace=False
            )
            # Average of Four positions randomly selected from population
            r0 = (
                         Xp[r1]
                         + Xp[r2]
                         + Xp[r3]
                         + Xp[r4]
                 ) / 4
            # Randomization Epsilon
            epsilon = 5e-3 * self.generator.random()
            delta = 2 * self.generator.random() * np.abs(r0 - Xp[idx])
            step = (g_best - Xp[r1] + delta) / 2
            delta_x = self.generator.choice(range(0, self.pop_size)) * np.abs(step)
            x1 = (
                    Xp[idx]
                    - self.generator.normal()
                    * p1
                    * 2
                    * delta_x
                    * Xp[idx]
                    / (g_worst - g_best + epsilon)
                    + self.generator.random()
                    * p2
                    * (g_best - Xp[idx])
            )
            z = Xp[idx] - self.generator.normal() * 2 * delta_x * Xp[idx] / (
                        g_worst - g_best + epsilon
                )
            y_p = self.generator.random() * (
                    (z + Xp[idx]) / 2 + self.generator.random() * delta_x
            )
            y_q = self.generator.random() * (
                    (z + Xp[idx]) / 2 - self.generator.random() * delta_x
            )
            x2 = (
                    g_best
                    - self.generator.normal()
                    * p1
                    * 2
                    * delta_x
                    * Xp[idx]
                    / (y_p - y_q + epsilon)
                    + self.generator.random()
                    * p2
                    * (Xp[r1] - Xp[r2])
            )

            x3 = Xp[idx] - p1 * (x2 - x1)
            ra = self.generator.random()
            rb = self.generator.random()
            pos_new = ra * (rb * x1 + (1 - rb) * x2) + (1 - ra) * x3

            # Local escaping operator
            if self.generator.random() < self.pr:
                f1 = self.generator.uniform(-1, 1)
                f2 = self.generator.normal(0, 1)
                L1 = np.round(1 - self.generator.random())
                u1 = L1 * 2 * self.generator.random() + (1 - L1)
                u2 = L1 * self.generator.random() + (1 - L1)
                u3 = L1 * self.generator.random() + (1 - L1)
                L2 = np.round(1 - self.generator.random())
                x_rand = self.problem.generate_solution()
                x_p = Xp[self.generator.choice(range(0, self.pop_size))]
                x_m = L2 * x_p + (1 - L2) * x_rand
                if self.generator.random() < 0.5:
                    pos_new = (
                            pos_new
                            + f1 * (u1 * g_best - u2 * x_m)
                            + f2
                            * p1
                            * (
                                    u3 * (x2 - x1)
                                    + u2 * (Xp[r1] - Xp[r2])
                            )
                            / 2
                    )
                else:
                    pos_new = (
                            g_best
                            + f1 * (u1 * g_best - u2 * x_m)
                            + f2
                            * p1
                            * (
                                    u3 * (x2 - x1)
                                    + u2 * (Xp[r1] - Xp[r2])
                            )
                            / 2
                    )
            # Check if solutions go outside the search space and bring them back
            pos_new = self._correct_solution(pos_new)
            if swarm:
                Xc[idx] = pos_new
            else:
                tar = self._get_target(pos_new)
                if self._compare_fitness(tar.fitness, pop.F[idx], self.problem.sense):
                    ops.set_row(pop, idx, pos_new, tar)
        if swarm:
            self.evaluate(cand, 0, pop.n)
            ops.accept(self, cand)
        order = self.sorted_order(pop)
        self.g_best, self.g_worst = pop.agent(order[0]), pop.agent(order[len(order) - 1])

#!/usr/bin/env python
# Created by "Thieu" at 21:42, 13/09/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalTHRO(LegacyNativeOptimizer):
    """
    The original version of: Tianji's Horse Racing Optimization (THRO)

    Links:
        + https://www.mathworks.com/matlabcentral/fileexchange/181341-tianji-s-horse-racing-optimization-thro

    Notes:
        + This algorithm has many drawbacks, especially in training, where such cases almost never occur.
        As a result, scenarios 3, 4, and 5 will practically never happen, since the situation where
        two solutions have exactly the same fitness value is very rare in practice.

    Examples
    ~~~~~~~~
    >>> from clypto.collection.game_based import THRO    >>> import numpy as np
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
    >>> model = THRO.OriginalTHRO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, L., Du, H., Zhang, Z., Hu, G., Mirjalili, S., Khodadadi, N., Hussien, A.G., Liao, Y. and Zhao, W., 2025.
    Tianji’s horse racing optimization (THRO): a new metaheuristic inspired by ancient wisdom and its
    engineering optimization applications. Artificial Intelligence Review, 58(9), p.282.
    """

    cdef public object n_pop
    cdef public object pop_tianji
    cdef public object pop_king

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

    cdef void before_main_loop(self):
        # Split to two groups: tianji and king (50%-50%)
        self.n_pop = self.pop_size // 2

    cdef void evolve(self, int epoch):
        # Sequential: horses are compared and replaced one after another, so the loops
        # run on the buffer rows of the two sub-populations.
        cdef NativePopulation pop, tian, king
        cdef NativeTarget tar
        cdef Py_ssize_t idx, jdx
        ### """Main racing phase with five scenarios"""
        # Randomly shuffle and redistribute populations
        n = self.pop.n
        pop = self.pop.take(self.rng.sample(range(n), n))
        # Sort populations by fitness
        tian = pop.take(np.arange(self.n_pop))
        tian = tian.take(self.sorted_order(tian))
        king = pop.take(np.arange(self.n_pop, n))
        king = king.take(self.sorted_order(king))

        # Generate binary matrices T_B and K_B
        t_b = np.zeros((self.n_pop, self.problem.n_dims))
        k_b = np.zeros((self.n_pop, self.problem.n_dims))

        for idx in range(self.n_pop):
            # For Tianji
            rand_dim = self.generator.permutation(self.problem.n_dims)
            rand_num = int(
                np.ceil(
                    np.sin(np.pi / 2 * self.generator.random()) * self.problem.n_dims
                )
            )
            t_b[idx, rand_dim[:rand_num]] = 1

            # For King
            rand_dim = self.generator.permutation(self.problem.n_dims)
            rand_num = int(
                np.ceil(
                    np.sin(np.pi / 2 * self.generator.random()) * self.problem.n_dims
                )
            )
            k_b[idx, rand_dim[:rand_num]] = 1

        # Weight parameter
        p = 1 - epoch / self.epoch

        # Current horse indices
        tianji_slowest_id = self.n_pop - 1
        tianji_fastest_id = 0
        king_slowest_id = self.n_pop - 1
        king_fastest_id = 0

        # Racing scenarios
        for idx in range(self.n_pop):
            tianji_alpha = (
                    1
                    + np.round(0.5 * (0.5 + self.generator.random()))
                    * self.generator.standard_normal()
            )
            t_beta = (
                    np.round(0.5 * (0.1 + self.generator.random()))
                    * self.generator.standard_normal()
            )
            king_alpha = (
                    1
                    + np.round(0.5 * (0.5 + self.generator.random()))
                    * self.generator.standard_normal()
            )
            k_beta = (
                    np.round(0.5 * (0.1 + self.generator.random()))
                    * self.generator.standard_normal()
            )

            tianji_r = (
                    self.get_levy_flight_step(beta=1.5, multiplier=1, size=None, case=-1)
                    * t_b[idx]
            )
            king_r = (
                    self.get_levy_flight_step(beta=1.5, multiplier=1, size=None, case=-1)
                    * k_b[idx]
            )

            fit_t = tian.F[tianji_slowest_id]
            fit_k = king.F[king_slowest_id]
            if self.problem.minmax == "min":
                if fit_t < fit_k:
                    case = 1
                elif fit_t > fit_k:
                    case = 2
                else:
                    case = 3
            else:
                if fit_t > fit_k:
                    case = 1
                elif fit_t < fit_k:
                    case = 2
                else:
                    case = 3

            tianji_mean = np.mean(np.ascontiguousarray(tian.X), axis=0)
            king_mean = np.mean(np.ascontiguousarray(king.X), axis=0)

            # Scenario 1: Tianji's slowest < King's slowest
            if case == 1:
                # Update Tianji's slowest horse
                pos_new = (
                                  (
                                          p * tian.X[tianji_slowest_id]
                                          + (1 - p) * tian.X[0]
                                  )
                                  + tianji_r
                                  * (
                                          tian.X[0]
                                          - tian.X[tianji_slowest_id]
                                          + p * (tianji_mean - king_mean)
                                  )
                          ) * tianji_alpha + t_beta
                pos_new = self.correct_solution(pos_new)
                tar = self.get_target(pos_new)
                if self.compare_fitness(tar.fitness, tian.F[tianji_slowest_id], self.problem.minmax):
                    ops.set_row(tian, tianji_slowest_id, pos_new, tar)

                # Update King's slowest horse
                pos_new = (
                                  (
                                          p * king.X[king_slowest_id]
                                          + (1 - p) * tian.X[tianji_slowest_id]
                                  )
                                  + king_r
                                  * (
                                          tian.X[tianji_slowest_id]
                                          - king.X[king_slowest_id]
                                          + p * (tianji_mean - king_mean)
                                  )
                          ) * king_alpha + k_beta
                pos_new = self.correct_solution(pos_new)
                tar = self.get_target(pos_new)
                if self.compare_fitness(tar.fitness, king.F[king_slowest_id], self.problem.minmax):
                    ops.set_row(king, king_slowest_id, pos_new, tar)

                tianji_slowest_id = max(0, tianji_slowest_id - 1)
                king_slowest_id = max(0, king_slowest_id - 1)

            # Scenario 2: Tianji's slowest > King's slowest
            elif case == 2:
                tr1 = self.generator.choice(
                    list(set(range(self.n_pop)) - {tianji_slowest_id})
                )
                # Update Tianji's slowest horse
                pos_new = (
                                  (
                                          p * tian.X[tianji_slowest_id]
                                          + (1 - p) * tian.X[tr1]
                                  )
                                  + tianji_r
                                  * (
                                          tian.X[tr1]
                                          - tian.X[tianji_slowest_id]
                                          + p * (tianji_mean - king_mean)
                                  )
                          ) * tianji_alpha + t_beta
                pos_new = self.correct_solution(pos_new)
                tar = self.get_target(pos_new)
                if self.compare_fitness(tar.fitness, tian.F[tianji_slowest_id], self.problem.minmax):
                    ops.set_row(tian, tianji_slowest_id, pos_new, tar)

                # Update King's fastest horse
                pos_new = (
                                  (
                                          p * king.X[king_fastest_id]
                                          + (1 - p) * king.X[0]
                                  )
                                  + king_r
                                  * (
                                          king.X[0]
                                          - king.X[king_fastest_id]
                                          + p * (tianji_mean - king_mean)
                                  )
                          ) * king_alpha + k_beta
                pos_new = self.correct_solution(pos_new)
                tar = self.get_target(pos_new)
                if self.compare_fitness(tar.fitness, king.F[king_fastest_id], self.problem.minmax):
                    ops.set_row(king, king_fastest_id, pos_new, tar)

                tianji_slowest_id = max(0, tianji_slowest_id - 1)
                king_fastest_id = min(self.n_pop - 1, king_fastest_id + 1)

            else:  # Equal slowest speeds
                fit_t = tian.F[tianji_fastest_id]
                fit_k = king.F[king_fastest_id]
                if self.problem.minmax == "min":
                    if fit_t < fit_k:
                        case_fast = 1
                    elif fit_t > fit_k:
                        case_fast = 2
                    else:
                        case_fast = 3
                else:
                    if fit_t > fit_k:
                        case_fast = 1
                    elif fit_t < fit_k:
                        case_fast = 2
                    else:
                        case_fast = 3

                # Scenario 3: Tianji's fastest < King's fastest
                if case_fast == 1:
                    # Update Tianji's fastest horse
                    pos_new = (
                                      (
                                              p * tian.X[tianji_fastest_id]
                                              + (1 - p) * tian.X[0]
                                      )
                                      + tianji_r
                                      * (
                                              tian.X[0]
                                              - tian.X[tianji_fastest_id]
                                              + p * (tianji_mean - king_mean)
                                      )
                              ) * tianji_alpha + t_beta
                    pos_new = self.correct_solution(pos_new)
                    tar = self.get_target(pos_new)
                    if self.compare_fitness(tar.fitness, tian.F[tianji_fastest_id], self.problem.minmax):
                        ops.set_row(tian, tianji_fastest_id, pos_new, tar)

                    # Update King's fastest horse
                    pos_new = (
                                      (
                                              p * king.X[king_fastest_id]
                                              + (1 - p) * tian.X[tianji_fastest_id]
                                      )
                                      + king_r
                                      * (
                                              tian.X[tianji_fastest_id]
                                              - king.X[king_fastest_id]
                                              + p * (tianji_mean - king_mean)
                                      )
                              ) * king_alpha + k_beta
                    pos_new = self.correct_solution(pos_new)
                    tar = self.get_target(pos_new)
                    if self.compare_fitness(tar.fitness, king.F[king_fastest_id], self.problem.minmax):
                        ops.set_row(king, king_fastest_id, pos_new, tar)

                    tianji_fastest_id = min(self.n_pop - 1, tianji_fastest_id + 1)
                    king_fastest_id = min(self.n_pop - 1, king_fastest_id + 1)

                # Scenario 4: Tianji's fastest > King's fastest
                elif case_fast == 2:
                    tr2 = self.generator.choice(
                        list(set(range(self.n_pop)) - {tianji_fastest_id})
                    )

                    # Update Tianji's slowest horse
                    pos_new = (
                                      (
                                              p * tian.X[tianji_slowest_id]
                                              + (1 - p) * tian.X[tr2]
                                      )
                                      + tianji_r
                                      * (
                                              tian.X[tr2]
                                              - tian.X[tianji_slowest_id]
                                              + p * (tianji_mean - king_mean)
                                      )
                              ) * tianji_alpha + t_beta
                    pos_new = self.correct_solution(pos_new)
                    tar = self.get_target(pos_new)
                    if self.compare_fitness(tar.fitness, tian.F[tianji_slowest_id], self.problem.minmax):
                        ops.set_row(tian, tianji_slowest_id, pos_new, tar)

                    # Update King's fastest horse
                    pos_new = (
                                      (
                                              p * king.X[king_fastest_id]
                                              + (1 - p) * king.X[0]
                                      )
                                      + king_r
                                      * (
                                              king.X[0]
                                              - king.X[king_fastest_id]
                                              + p * (tianji_mean - king_mean)
                                      )
                              ) * king_alpha + k_beta
                    pos_new = self.correct_solution(pos_new)
                    tar = self.get_target(pos_new)
                    if self.compare_fitness(tar.fitness, king.F[king_fastest_id], self.problem.minmax):
                        ops.set_row(king, king_fastest_id, pos_new, tar)

                    tianji_slowest_id = max(0, tianji_slowest_id - 1)
                    king_fastest_id = min(self.n_pop - 1, king_fastest_id + 1)

                # Scenario 5: Equal fastest speeds
                else:
                    tr3 = self.generator.choice(
                        list(set(range(self.n_pop)) - {tianji_slowest_id})
                    )

                    # Update Tianji's slowest horse
                    pos_new = (
                                      (
                                              p * tian.X[tianji_slowest_id]
                                              + (1 - p) * tian.X[tr3]
                                      )
                                      + tianji_r
                                      * (
                                              tian.X[tr3]
                                              - tian.X[tianji_slowest_id]
                                              + p * (tianji_mean - king_mean)
                                      )
                              ) * tianji_alpha + t_beta
                    pos_new = self.correct_solution(pos_new)
                    tar = self.get_target(pos_new)
                    if self.compare_fitness(tar.fitness, tian.F[tianji_slowest_id], self.problem.minmax):
                        ops.set_row(tian, tianji_slowest_id, pos_new, tar)

                    # Update King's fastest horse
                    pos_new = (
                                      (
                                              p * king.X[king_fastest_id]
                                              + (1 - p) * king.X[0]
                                      )
                                      + king_r
                                      * (
                                              king.X[0]
                                              - king.X[king_fastest_id]
                                              + p * (tianji_mean - king_mean)
                                      )
                              ) * king_alpha + k_beta
                    pos_new = self.correct_solution(pos_new)
                    tar = self.get_target(pos_new)
                    if self.compare_fitness(tar.fitness, king.F[king_fastest_id], self.problem.minmax):
                        ops.set_row(king, king_fastest_id, pos_new, tar)

                    tianji_slowest_id = max(0, tianji_slowest_id - 1)
                    king_fastest_id = min(self.n_pop - 1, king_fastest_id + 1)

        # Training phase. The classic agents share their solution arrays (get_best_agent
        # copies the agent but not its array) and the moves below edit them in place, so the
        # phase runs on lists of arrays to keep those aliasing effects.
        sol_t, sol_k = [row.copy() for row in tian.X], [row.copy() for row in king.X]
        best_tianji = sol_t[self.sorted_order(tian)[0]]
        best_king = sol_k[self.sorted_order(king)[0]]

        for idx in range(self.n_pop):
            # Training for Tianji's population
            pos_new = sol_t[idx]
            for jdx in range(self.problem.n_dims):
                if self.generator.random() > 0.5:  # Levy flight based training
                    tr4, tr5 = self.generator.choice(
                        list(set(range(self.n_pop)) - {idx}), size=2, replace=False
                    )
                    lt = self.get_levy_flight_step(
                        beta=1.5, multiplier=0.2, size=None, case=-1
                    )
                    pos_new[jdx] = pos_new[jdx] + lt * (
                            sol_t[tr4][jdx]
                            - sol_t[tr5][jdx]
                    )
                else:  # Best-guided training
                    mt = 0.5 * (
                            1
                            + 0.001
                            * (<object>(1 - epoch / self.epoch)) ** 2
                            * np.sin(np.pi * self.generator.random())
                    )
                    pos_new[jdx] = best_tianji[jdx] + mt * (
                            best_tianji[jdx] - pos_new[jdx]
                    )
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)
            if self.compare_fitness(tar.fitness, tian.F[idx], self.problem.minmax):
                sol_t[idx] = pos_new
                tian.F[idx] = tar.fitness
                tian.O[idx] = tar.objectives

            # Training for King's population
            pos_new = sol_t[idx]
            for jdx in range(self.problem.n_dims):
                if self.generator.random() > 0.5:  # Levy flight based training
                    kr1, kr2 = self.generator.choice(
                        list(set(range(self.n_pop)) - {idx}), size=2, replace=False
                    )
                    lk = self.get_levy_flight_step(
                        beta=1.5, multiplier=0.2, size=None, case=-1
                    )
                    pos_new[jdx] = pos_new[jdx] + lk * (
                            sol_k[kr1][jdx]
                            - sol_k[kr2][jdx]
                    )
                else:  # Best-guided training
                    mk = 0.5 * (
                            1
                            + 0.001
                            * (<object>(1 - epoch / self.epoch)) ** 2
                            * np.sin(np.pi * self.generator.random())
                    )
                    pos_new[jdx] = best_king[jdx] + mk * (
                            best_king[jdx] - pos_new[jdx]
                    )
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)
            if self.compare_fitness(tar.fitness, king.F[idx], self.problem.minmax):
                sol_k[idx] = pos_new
                king.F[idx] = tar.fitness
                king.O[idx] = tar.objectives

        tian.X[:] = sol_t
        king.X[:] = sol_k
        # Merge populations back
        self.pop = tian.concat(king)

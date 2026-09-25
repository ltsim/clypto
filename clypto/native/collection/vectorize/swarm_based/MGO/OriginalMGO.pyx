#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalMGO(LegacyNativeOptimizer):
    """
    The original version of: Mountain Gazelle Optimizer (MGO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0965997822001831
        2. https://www.mathworks.com/matlabcentral/fileexchange/118680-mountain-gazelle-optimizer

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import MGO    >>> import numpy as np
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
    >>> model = MGO.OriginalMGO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Gharehchopogh, F. S., Khodadadi, N., & Mirjalili, S. (2022). Mountain gazelle optimizer: a new
    nature-inspired metaheuristic algorithm for global optimization problems. Advances in Engineering Software, 174, 103282.
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

    def coefficient_vector__(self, n_dims, epoch, max_epoch):
        a2 = -1.0 + epoch * (-1.0 / max_epoch)
        u = self.generator.standard_normal(n_dims)
        v = self.generator.standard_normal(n_dims)
        cofi = np.zeros((4, n_dims))
        cofi[0, :] = self.generator.random(n_dims)
        cofi[1, :] = (a2 + 1) + self.generator.random()
        cofi[2, :] = a2 * self.generator.standard_normal(n_dims)
        cofi[3, :] = u * np.power(v, 2) * np.cos((self.generator.random() * 2) * u)
        return cofi

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.take(np.zeros(4 * pop.n, dtype=int))
        cdef Py_ssize_t idx, n = pop.n
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        for idx in range(0, self.pop_size):
            idxs_rand = self.generator.permutation(self.pop_size)[: int(np.ceil(self.pop_size / 3))]
            pos_list = np.array([Xp[mm] for mm in idxs_rand])
            idx_rand = self.generator.integers(int(np.ceil(self.pop_size / 3)), self.pop_size)
            M = Xp[idx_rand] * np.floor(self.generator.normal()) + np.mean(pos_list, axis=0) * np.ceil(self.generator.normal())

            # Calculate the vector of coefficients
            cofi = self.coefficient_vector__(self.problem.n_dims, epoch, self.epoch)
            A = self.generator.standard_normal(self.problem.n_dims) * np.exp(2 - epoch * (2.0 / self.epoch))
            D = (np.abs(Xp[idx]) + np.abs(g_best)) * (2 * self.generator.random() - 1)

            # Update the location
            x2 = (
                    g_best
                    - np.abs(
                (
                        self.generator.integers(1, 3) * M
                        - self.generator.integers(1, 3) * Xp[idx]
                )
                * A
            )
                    * cofi[self.generator.integers(0, 4), :]
            )
            x3 = (
                    M
                    + cofi[self.generator.integers(0, 4), :]
                    + (
                            self.generator.integers(1, 3) * g_best
                            - self.generator.integers(1, 3)
                            * Xp[self.generator.integers(self.pop_size)]
                    )
                    * cofi[self.generator.integers(0, 4), :]
            )
            x4 = (
                    Xp[idx]
                    - D
                    + (
                            self.generator.integers(1, 3) * g_best
                            - self.generator.integers(1, 3) * M
                    )
                    * cofi[self.generator.integers(0, 4), :]
            )

            x1 = self.problem.generate_solution()
            Xc[4 * idx] = self.correct_solution(x1)
            Xc[4 * idx + 1] = self.correct_solution(x2)
            Xc[4 * idx + 2] = self.correct_solution(x3)
            Xc[4 * idx + 3] = self.correct_solution(x4)
        self.evaluate(cand, 0, 4 * n)
        both = pop.concat(cand)
        self.pop = both.take(self.sorted_order(both)[:n])

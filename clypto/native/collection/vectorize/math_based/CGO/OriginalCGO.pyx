#!/usr/bin/env python
# Created by "Thieu" at 22:24, 02/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalCGO(LegacyNativeOptimizer):
    """
    The original version of: Chaos Game Optimization (CGO)

    Links:
        1. https://doi.org/10.1007/s10462-020-09867-w

    Notes:
        + 4th seed is mutation process, but it is not clear mutation on multiple variables or 1 variable
        + There is no usage of the variable alpha 4th in the paper
        + The replacement of the worst solutions by generated seed are not clear (Lots of grammar errors in this section)

    Examples
    ~~~~~~~~

    >>> from clypto.collection.math_based import CGO    >>> import numpy as np
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
    >>> model = CGO.OriginalCGO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Talatahari, S. and Azizi, M., 2021. Chaos Game Optimization: a novel metaheuristic algorithm.
    Artificial Intelligence Review, 54(2), pp.917-1004.
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

    cdef void evolve(self, int epoch):
        # Agents read the population they just updated (three random members), so the
        # loop stays sequential on the buffer rows.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation seeds
        cdef Py_ssize_t idx
        minmax = self.problem.minmax
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        for idx in range(self.pop_size):
            s1, s2, s3 = self.generator.choice(range(0, self.pop_size), 3, replace=False)
            MG = (Xp[s1] + Xp[s2] + Xp[s3]) / 3
            ## Calculating alpha based on Eq. 7
            alpha1 = self.generator.random()
            alpha2 = 2 * self.generator.random()
            alpha3 = 1 + self.generator.random() * self.generator.random()
            esp = self.generator.random()
            # There is no usage of this variable in the paper
            alpha4 = esp + esp * self.generator.random()
            beta = self.generator.integers(0, 2, 3)
            gama = self.generator.integers(0, 2, 3)
            ## The seed4 is mutation process, but not sure k is multiple variables or 1 variable.
            ## In the text said, multiple variables, but the defination of k is 1 variable. So confused
            k = self.generator.integers(0, self.problem.n_dims)
            k_idx = self.generator.choice(range(0, self.problem.n_dims), k, replace=False)
            seed1 = Xp[idx] + alpha1 * (beta[0] * g_best - gama[0] * MG)  # Eq. 3
            seed2 = g_best + alpha2 * (beta[1] * Xp[idx] - gama[1] * MG)  # Eq. 4
            seed3 = MG + alpha3 * (beta[2] * Xp[idx] - gama[2] * g_best)  # Eq. 5
            seed4 = Xp[idx].copy().astype(float)
            seed4[k_idx] += self.generator.uniform(0, 1, k)
            # Check if solutions go outside the search space and bring them back
            seeds = self.new_population(self.correct_solution(np.array([seed1, seed2, seed3, seed4])))
            ## Lots of grammar errors in this section, so confused to understand which strategy they are using
            best = self.sorted_order(seeds)[0]
            if self.compare_fitness(seeds.F[best], pop.F[idx], minmax):
                pop.buf[idx] = seeds.buf[best]

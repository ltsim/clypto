#!/usr/bin/env python
# Created by "Thieu" at 14:20, 15/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalSOS(LegacyNativeOptimizer):
    """
    The original version: Symbiotic Organisms Search (SOS)

    Links:
        1. https://doi.org/10.1016/j.compstruc.2014.03.007

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import SOS    >>> import numpy as np
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
    >>> model = SOS.OriginalSOS(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Cheng, M. Y., & Prayogo, D. (2014). Symbiotic organisms search: a new metaheuristic
    optimization algorithm. Computers & Structures, 139, 98-112.
    """

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
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

    cdef void evolve(self, int epoch_c):
        # Every organism updates its own row and its partner's, and the next ones read them; the best
        # organism is aliased by g_best, so it is read live: sequential on the buffer rows.
        cdef NativePopulation pop = self.pop
        cdef NativeTarget xi_target, xj_target
        cdef Py_ssize_t idx, n = pop.n
        minmax = self.problem.minmax
        Xp = pop.X
        for idx in range(0, self.pop_size):
            ## Mutualism Phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            mutual_vector = (Xp[idx] + Xp[jdx]) / 2
            bf1, bf2 = self.generator.integers(1, 3, 2)
            xi_new = Xp[idx] + self.generator.random() * (self.g_best_x() - bf1 * mutual_vector)
            xj_new = Xp[jdx] + self.generator.random() * (self.g_best_x() - bf2 * mutual_vector)
            xi_new = self.correct_solution(xi_new)
            xj_new = self.correct_solution(xj_new)
            xi_target = self.get_target(xi_new)
            xj_target = self.get_target(xj_new)
            if self.compare_fitness(xi_target.fitness, pop.F[idx], minmax):
                ops.set_row(pop, idx, xi_new, xi_target)
            if self.compare_fitness(xj_target.fitness, pop.F[jdx], minmax):
                ops.set_row(pop, jdx, xj_new, xj_target)
            ## Commensalism phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            xi_new = Xp[idx] + self.generator.uniform(-1, 1) * (self.g_best_x() - Xp[jdx])
            xi_new = self.correct_solution(xi_new)
            xi_target = self.get_target(xi_new)
            if self.compare_fitness(xi_target.fitness, pop.F[idx], minmax):
                ops.set_row(pop, idx, xi_new, xi_target)
            ## Parasitism phase
            jdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            temp_idx = self.generator.integers(0, self.problem.n_dims)
            xi_new = Xp[jdx].copy()
            xi_new[temp_idx] = self.problem.generate_solution()[temp_idx]
            xi_new = self.correct_solution(xi_new)
            xi_target = self.get_target(xi_new)
            if self.compare_fitness(xi_target.fitness, pop.F[idx], minmax):
                ops.set_row(pop, idx, xi_new, xi_target)

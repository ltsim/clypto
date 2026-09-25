#!/usr/bin/env python
# Created by "Thieu" at 20:22, 12/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class DevSMA(LegacyNativeOptimizer):
    """
    The developed version: Slime Mould Algorithm (SMA)

    Notes:
        + Selected 2 unique and random solution to create new solution (not to create variable)
        + Check bound and compare old position with new position to get the best one

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_t (float): (0, 1.0) -> better [0.01, 0.1], probability threshold (z in the paper)

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import SMA    >>> import numpy as np
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
    >>> model = SMA.DevSMA(epoch=1000, pop_size=50, p_t = 0.03)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        p_t: float = 0.03,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_t (float): probability threshold (z in the paper), default = 0.03
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "p_t"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.p_t = cy.validator(float, p_t, (0, 1.0), "p_t")

    cdef void initialize_variables(self):
        self.weights = np.zeros((self.pop_size, self.problem.n_dims))

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        gb_fit = self.current_g_best().target.fitness
        # plus eps to avoid denominator zero
        ss = gb_fit - pop.F[-1] + self.EPSILON
        # calculate the fitness weight of each slime mold
        for idx in range(0, self.pop_size):
            # Eq.(2.5)
            if idx <= int(self.pop_size / 2):
                self.weights[idx] = 1 + self.generator.uniform(
                    0, 1, self.problem.n_dims
                ) * np.log10(
                    (gb_fit - pop.F[idx]) / ss + 1
                )
            else:
                self.weights[idx] = 1 - self.generator.uniform(
                    0, 1, self.problem.n_dims
                ) * np.log10(
                    (gb_fit - pop.F[idx]) / ss + 1
                )
        a = np.arctanh(1 - epoch / self.epoch)  # Eq.(2.4)
        b = 1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            # Update the Position of search agent
            if self.generator.uniform() < self.p_t:  # Eq.(2.7)
                pos_new = self.problem.generate_solution()
            else:
                p = np.tanh(
                    np.abs(pop.F[idx] - gb_fit)
                )  # Eq.(2.2)
                vb = self.generator.uniform(-a, a, self.problem.n_dims)  # Eq.(2.3)
                vc = self.generator.uniform(-b, b, self.problem.n_dims)
                # two positions randomly selected from population, apply for the whole problem size instead of 1 variable
                id_a, id_b = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
                )
                pos_1 = g_best + vb * (
                        self.weights[idx] * Xp[id_a]
                        - Xp[id_b]
                )
                pos_2 = vc * Xp[idx]
                condition = self.generator.random(self.problem.n_dims) < p
                pos_new = np.where(condition, pos_1, pos_2)
            # Check bound and re-calculate fitness after each individual move
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)

#!/usr/bin/env python
# Created by "Thieu" at 10:55, 02/12/2019 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalSHO(LegacyNativeOptimizer):
    """
    The original version of: Spotted Hyena Optimizer (SHO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2017.05.014

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + h_factor (float): default = 5, coefficient linearly decreased from 5 to 0
        + n_trials (int): default = 10

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import SHO    >>> import numpy as np
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
    >>> model = SHO.OriginalSHO(epoch=1000, pop_size=50, h_factor = 5.0, n_trials = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Dhiman, G. and Kumar, V., 2017. Spotted hyena optimizer: a novel bio-inspired based metaheuristic
    technique for engineering applications. Advances in Engineering Software, 114, pp.48-70.
    """

    cdef public object h_factor
    cdef public object n_trials

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        h_factor: float = 5.0,
        n_trials: int = 10,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            h_factor (float): default = 5, coefficient linearly decreased from 5.0 to 0
            n_trials (int): default = 10,
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "h_factor", "n_trials"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.h_factor = cy.validator(float, h_factor, (0.5, 10.0), "h_factor")
        self.n_trials = cy.validator(int, n_trials, (1, float("inf")), "n_trials")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef NativeTarget tar
        cdef Py_ssize_t idx
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        gb_fit = self.current_g_best().target.fitness
        for idx in range(0, self.pop_size):
            hh = self.h_factor - epoch * (self.h_factor / self.epoch)
            rd1 = self.generator.uniform(0, 1, self.problem.n_dims)
            rd2 = self.generator.uniform(0, 1, self.problem.n_dims)
            B = 2 * rd1
            E = 2 * hh * rd2 - hh

            if self.generator.random() < 0.5:
                D_h = np.abs(np.dot(B, g_best) - Xp[idx])
                pos_new = g_best - np.dot(E, D_h)
            else:
                N = 1
                for _ in range(0, self.n_trials):
                    pos_temp = g_best + self.generator.normal(
                        0, 1, self.problem.n_dims
                    ) * self.generator.uniform(self.problem.lb, self.problem.ub)
                    pos_new = self.correct_solution(pos_temp)
                    tar = self.get_target(pos_new)
                    if self.compare_fitness(tar.fitness, gb_fit, self.problem.minmax):
                        N += 1
                        break
                    N += 1
                circle_list = []
                idx_list = self.generator.choice(
                    range(0, self.pop_size), N, replace=False
                )
                for j in range(0, N):
                    D_h = np.abs(np.dot(B, g_best) - Xp[idx_list[j]])
                    p_k = g_best - np.dot(E, D_h)
                    circle_list.append(p_k)
                pos_new = np.mean(np.array(circle_list), axis=0)
            pos_new = self.correct_solution(pos_new)
            ops.commit(self, pop, cand, idx, pos_new, swarm, True)
        if swarm:
            ops.finish(self, cand, 0, pop.n)

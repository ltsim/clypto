#!/usr/bin/env python
# Created by "Thieu" at 20:22, 12/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.bio_based.SMA.DevSMA cimport DevSMA
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalSMA(DevSMA):
    """
    The original version of: Slime Mould Algorithm (SMA)

    Links:
        1. https://doi.org/10.1016/j.future.2020.03.055
        2. https://www.researchgate.net/publication/340431861_Slime_mould_algorithm_A_new_method_for_stochastic_optimization

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
    >>> model = SMA.OriginalSMA(epoch=1000, pop_size=50, p_t = 0.03)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Li, S., Chen, H., Wang, M., Heidari, A.A. and Mirjalili, S., 2020. Slime mould algorithm: A new method for
    stochastic optimization. Future Generation Computer Systems, 111, pp.300-323.
    """

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        p_t = 0.03,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): number of population size, default = 100
            p_t (float): probability threshold (z in the paper), default = 0.03
        """
        super().__init__(epoch, pop_size, p_t, name=name, mode=mode)

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

        aa = np.arctanh(-(epoch / self.epoch) + 1)  # Eq.(2.4)
        bb = 1 - epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            # Update the Position of search agent
            pos_new = Xp[idx].copy()
            if self.generator.uniform() < self.p_t:  # Eq.(2.7)
                pos_new = self.problem.generate_solution()
            else:
                p = np.tanh(
                    np.abs(pop.F[idx] - gb_fit)
                )  # Eq.(2.2)
                vb = self.generator.uniform(-aa, aa, self.problem.n_dims)  # Eq.(2.3)
                vc = self.generator.uniform(-bb, bb, self.problem.n_dims)
                for jdx in range(0, self.problem.n_dims):
                    # two positions randomly selected from population
                    id_a, id_b = self.generator.choice(
                        list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
                    )
                    if self.generator.uniform() < p:  # Eq.(2.1)
                        pos_new[jdx] = g_best[jdx] + vb[jdx] * (
                                self.weights[idx, jdx] * Xp[id_a][jdx]
                                - Xp[id_b][jdx]
                        )
                    else:
                        pos_new[jdx] = vc[jdx] * pos_new[jdx]
            pos_new = self.correct_solution(pos_new)
            if swarm:
                Xc[idx] = pos_new
            else:
                # every agent is replaced by its candidate, and the next ones read it
                ops.set_row(pop, idx, pos_new, self.get_target(pos_new))
        if swarm:
            self.evaluate(cand, 0, n)
            self.pop = cand

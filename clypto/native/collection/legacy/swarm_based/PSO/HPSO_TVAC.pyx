#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.native.collection.legacy.swarm_based.PSO.P_PSO cimport P_PSO


cdef class HPSO_TVAC(P_PSO):
    """
    The original version of: Hierarchical PSO Time-Varying Acceleration (HPSO-TVAC)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + ci (float): [0.3, 1.0], c initial, default = 0.5
        + cf (float): [0.0, 0.3], c final, default = 0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import PSO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "sense": "min",
    >>> }
    >>>
    >>> model = PSO.HPSO_TVAC(epoch=1000, pop_size=50, ci=0.5, cf=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ghasemi, M., Aghaei, J. and Hadipour, M., 2017. New self-organising hierarchical PSO with
    jumping time-varying acceleration coefficients. Electronics Letters, 53(20), pp.1360-1362.
    """

    def __init__(self, epoch=10000, pop_size=100, ci=0.5, cf=0.1, **kwargs):
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            ci: c initial, default = 0.5
            cf: c final, default = 0.0
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.ci = self.validator.check_float("ci", ci, [0.3, 1.0])
        self.cf = self.validator.check_float("cf", cf, [0, 0.3])
        self._set_parameters(["epoch", "pop_size", "ci", "cf"])
        self.sort_flag = False

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        c_it = ((self.cf - self.ci) * (epoch / self.epoch)) + self.ci
        for idx in range(0, self.pop_size):
            idx_k = self.generator.integers(0, self.pop_size)
            w = self.generator.normal()
            while np.abs(w - 1.0) < 0.01:
                w = self.generator.normal()
            c1_it = np.abs(w) ** (c_it * w)
            c2_it = np.abs(1 - w) ** (c_it / (1 - w))
            #################### HPSO
            v_new = c1_it * self.generator.uniform(0, 1, self.problem.n_dims) * (
                self.pop[idx].local_solution - self.pop[idx].solution
            ) + c2_it * self.generator.uniform(0, 1, self.problem.n_dims) * (
                self.g_best.solution
                + self.pop[idx_k].local_solution
                - 2 * self.pop[idx].solution
            )
            v_new = np.where(
                v_new == 0,
                np.sign(0.5 - self.generator.uniform())
                * self.generator.uniform()
                * self.v_max,
                v_new,
            )
            v_new = np.sign(v_new) * np.minimum(np.abs(v_new), self.v_max)
            #########################
            v_new = np.minimum(np.maximum(v_new, -self.v_max), self.v_max)
            pos_new = self.pop[idx].solution + v_new
            pos_new = self._correct_solution(pos_new)
            self.pop[idx].velocity = v_new
            target = self._get_target(pos_new)
            if self._compare_target(target, self.pop[idx].target, self.problem.sense):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self._compare_target(
                target, self.pop[idx].local_target, self.problem.sense
            ):
                self.pop[idx].update(
                    local_solution=pos_new.copy(), local_target=target.copy()
                )

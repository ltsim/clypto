#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 14:01, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.collection.swarm_based.FOA.OriginalFOA cimport OriginalFOA
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class WhaleFOA(OriginalFOA):
    """
    The original version of: Whale Fruit-fly Optimization Algorithm (WFOA)

    Links:
        1. https://doi.org/10.1016/j.eswa.2020.113502

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import FOA    >>> import numpy as np
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
    >>> model = FOA.WhaleFOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Fan, Y., Wang, P., Heidari, A.A., Wang, M., Zhao, X., Chen, H. and Li, C., 2020. Boosted hunting-based
    fruit fly optimization and advances in real-world problems. Expert Systems with Applications, 159, p.113502.
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
        super().__init__(epoch, pop_size, name=name, mode=mode)

    def evolve_agents(self, epoch):
        a = 2.0 - 2.0 * epoch / self.epoch  # linearly decreased from 2 to 0
        pop_new = []
        for idx in range(0, self.pop_size):
            r = self.generator.random()
            A = 2 * a * r - a
            C = 2 * r
            l = self.generator.uniform(-1, 1)
            p = 0.5
            b = 1
            if self.generator.random() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best.solution - self.objs[idx].solution)
                    pos_new = self.g_best.solution - A * D
                else:
                    # select random 1 position in pop
                    x_rand = self.objs[self.generator.integers(self.pop_size)]
                    D = np.abs(C * x_rand.solution - self.objs[idx].solution)
                    pos_new = x_rand.solution - A * D
            else:
                D1 = np.abs(self.g_best.solution - self.objs[idx].solution)
                pos_new = (
                    D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + self.g_best.solution
                )
            smell = self.norm_consecutive_adjacent__(pos_new)
            pos_new = self.correct_solution(smell)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                pop_new, self.objs, self.problem.minmax
            )

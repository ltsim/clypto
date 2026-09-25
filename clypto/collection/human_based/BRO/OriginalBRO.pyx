#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 09:17, 09/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.collection.human_based.BRO.DevBRO cimport DevBRO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class OriginalBRO(DevBRO):
    """
    The original version of: Battle Royale Optimization (BRO)

    Links:
        1. https://doi.org/10.1007/s00521-020-05004-4

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + threshold (int): [2, 5], dead threshold, default=3

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import BRO    >>> import numpy as np
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
    >>> model = BRO.OriginalBRO(epoch=1000, pop_size=50, threshold = 3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rahkar Farshi, T., 2021. Battle royale optimization algorithm. Neural Computing and Applications, 33(4), pp.1139-1157.
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        threshold: float = 3,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            threshold (int): dead threshold, default=3
        """
        super().__init__(epoch, pop_size, threshold, name=name, mode=mode)
        self.sort_flag = False
        self.is_parallelizable = False

    def evolve_agents(self, epoch):
        for idx in range(self.pop_size):
            # Compare ith soldier with nearest one (jth)
            jdx = self.find_idx_min_distance__(self.objs[idx].solution, self.objs)
            dam, vic = (
                idx,
                jdx,
            )  ## This error in the algorithm's flow in the paper, But in the matlab code, he changed.
            if self.compare_target(
                self.objs[idx].target, self.objs[jdx].target, self.problem.minmax
            ):
                dam, vic = jdx, idx  ## The mistake also here in the paper.
            if self.objs[dam].damage < self.threshold:
                pos_new = self.generator.uniform(0, 1, self.problem.n_dims) * (
                    np.maximum(self.objs[dam].solution, self.g_best.solution)
                    - np.minimum(self.objs[dam].solution, self.g_best.solution)
                ) + np.maximum(self.objs[dam].solution, self.g_best.solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                agent.damage = self.objs[dam].damage + 1
                self.objs[dam] = agent
                self.objs[vic].damage = 0
            else:
                pos_new = self.generator.uniform(
                    self.lb_updated, self.ub_updated
                )
                agent = self.generate_agent(pos_new)
                self.objs[dam] = agent
        if epoch >= self.dyn_delta:
            pos_list = np.array(
                [self.objs[idx].solution for idx in range(0, self.pop_size)]
            )
            pos_std = np.std(pos_list, axis=0)
            lb = self.g_best.solution - pos_std
            ub = self.g_best.solution + pos_std
            self.lb_updated = np.clip(
                lb, self.lb_updated, self.ub_updated
            )
            self.ub_updated = np.clip(
                ub, self.lb_updated, self.ub_updated
            )
            self.dyn_delta += round(self.dyn_delta / 2)

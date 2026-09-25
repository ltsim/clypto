#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 10:14, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.human_based.TLO.DevTLO cimport DevTLO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class OriginalTLO(DevTLO):
    """
    The original version of: Teaching Learning-based Optimization (TLO)

    Notes:
        + Third loops are removed
        + This version is inspired from above link
        + https://github.com/andaviaco/tblo

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import TLO    >>> import numpy as np
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
    >>> model = TLO.OriginalTLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R.V., Savsani, V.J. and Vakharia, D.P., 2011. Teaching–learning-based optimization: a novel method
    for constrained mechanical design optimization problems. Computer-aided design, 43(3), pp.303-315.
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
        self.sort_flag = False
        self.is_parallelizable = False

    def evolve_agents(self, epoch):
        for idx in range(0, self.pop_size):
            ## Teaching Phrase
            TF = self.generator.integers(1, 3)  # 1 or 2 (never 3)
            #### Remove third loop here
            list_pos = np.array([agent.solution for agent in self.objs])
            pos_new = self.objs[idx].solution + self.generator.uniform(
                0, 1, self.problem.n_dims
            ) * (self.g_best.solution - TF * np.mean(list_pos, axis=0))
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(
                    agent.target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx] = agent
            ## Learning Phrase
            id_partner = self.generator.choice(
                np.setxor1d(np.array(range(self.pop_size)), np.array([idx]))
            )
            #### Remove third loop here
            if self.compare_target(
                    self.objs[idx].target, self.objs[id_partner].target, self.problem.minmax
            ):
                diff = self.objs[idx].solution - self.objs[id_partner].solution
            else:
                diff = self.objs[id_partner].solution - self.objs[idx].solution
            pos_new = (
                    self.objs[idx].solution
                    + self.generator.uniform(0, 1, self.problem.n_dims) * diff
            )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(
                    agent.target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx] = agent

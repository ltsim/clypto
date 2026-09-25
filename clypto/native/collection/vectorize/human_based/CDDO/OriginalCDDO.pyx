#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 23:41, 15/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class OriginalCDDO(AgentListOptimizer):
    """
    The original version of: Child Drawing Development Optimization (CCDO)

    Notes:
        + This source code was converted from the original Matlab implementation in the paper into Python.
        The Matlab code itself has many issues, for example, parameters are defined but never used.
        Several variables are declared, such as p1, p2, p3. Parameters like child skill rate and child level
        rate are initialized as hyperparameters at the beginning, but inside the loop they are randomly generated,
        which is inconsistent with the paper.

        + Moreover, the biggest flaw of this algorithm lies in the if–else condition during the update process.
        There is a high chance that neither condition will be executed, because the golden ratio is not necessarily
        within the interval [1.5, 2], as it is computed based on a random position. In addition, when comparing
        the position with a random integer T (hand pressure), it is unclear why this is done. It is highly likely
        that the algorithm will only execute that single condition.

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import CDDO    >>> import numpy as np
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
    >>> model = CDDO.OriginalCDDO(epoch=1000, pop_size=50, pattern_size=10, creativity_rate=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdulhameed, S., Rashid, T.A. Child Drawing Development Optimization Algorithm Based on
    Child’s Cognitive Development. Arab J Sci Eng 47, 1337–1351 (2022). https://doi.org/10.1007/s13369-021-05928-6
    """

    cdef public object pattern_size
    cdef public object creativity_rate
    cdef public object LR
    cdef public object SR
    cdef public object pop_local
    cdef public object list_gr

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pattern_size = 10,
        creativity_rate = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pattern_size (int): size of the pattern matrix, default = 10
            creativity_rate (float): creativity rate, default = 0.1
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pattern_size", "creativity_rate"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pattern_size = cy.validator(int, pattern_size, [1, 1000], "pattern_size")
        self.creativity_rate = cy.validator(float, creativity_rate, [0.0, 1.0], "creativity_rate")

    cdef void before_main_loop(self):
        self.LR = self.generator.uniform(0.1, 1.0)  # Child level rate
        self.SR = self.generator.uniform(0.1, 1.0)  # Child Skill Rate
        self.pop_local = self.objs.copy()
        # Golden ratio
        self.list_gr = []
        for idx in range(self.pop_size):
            p1 = self.generator.integers(0, self.problem.n_dims)
            p2 = self.generator.integers(0, self.problem.n_dims)
            if self.objs[idx].solution[p1] == 0:
                self.list_gr.append(self.objs[idx].solution[p2])
            else:
                self.list_gr.append(
                    self.objs[idx].solution[p1]
                    + self.objs[idx].solution[p2] / self.objs[idx].solution[p1]
                )

    def evolve_agents(self, epoch):
        # Pattern matrix
        _, pattern, _ = self.get_special_agents(
            self.objs, n_best=self.pattern_size, minmax=self.problem.minmax
        )
        for idx in range(0, self.pop_size):
            hand_pressure = self.generator.integers(
                self.problem.lb[0], self.problem.ub[0] + 1
            )
            pp = self.generator.integers(0, self.problem.n_dims)
            pos_new = self.objs[idx].solution.copy()
            if self.objs[idx].solution[pp] <= hand_pressure:
                # Update the drawings
                pos_new = (
                        self.list_gr[idx]
                        + self.SR
                        * self.generator.random(self.problem.n_dims)
                        * (self.pop_local[idx].solution - self.objs[idx].solution)
                        + self.LR
                        * self.generator.random(self.problem.n_dims)
                        * (self.g_best.solution - self.objs[idx].solution)
                )
                self.LR = self.generator.integers(6, 11) / 10
                self.SR = self.generator.integers(6, 11) / 10
            elif 1.5 < self.list_gr[idx] < 2:
                # Consider the learnt patterns
                pos_new = (
                        pattern[self.generator.integers(0, self.pattern_size)].solution
                        - self.creativity_rate * self.pop_local[idx].solution
                )
                self.LR = self.generator.integers(0, 6) / 10
                self.SR = self.generator.integers(0, 6) / 10
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            self.objs[idx] = agent
            if self.mode not in self.AVAILABLE_MODES:
                self.objs[idx].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            self.objs = self.update_target_for_population(self.objs)
        # Update the local information
        self.pop_local = self.greedy_selection_population(
            self.pop_local, self.objs, self.problem.minmax
        )

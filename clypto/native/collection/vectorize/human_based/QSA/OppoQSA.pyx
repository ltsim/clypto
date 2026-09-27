#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 10:21, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.native.collection.vectorize.human_based.QSA.DevQSA cimport DevQSA
import numpy as np
cimport clypto.core as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class OppoQSA(DevQSA):
    """
    The opposition-based learning version: Queuing Search Algorithm (OQSA)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import QSA    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = QSA.OppoQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
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
        self.sort_flag = True

    def opposition_based__(self, pop=None, g_best=None):
        pop = cy.sort_agents(pop, self.problem.sense)
        pop_new = []
        for idx in range(0, self.pop_size):
            # opposition-based candidate around g_best
            pos_new = self.correct_solution(
                self.problem.bounds.low + self.problem.bounds.up - g_best.solution
                + self.generator.uniform() * (g_best.solution - pop[idx].solution)
            )
            pos_new = self.correct_solution(pos_new)
            agent = self.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                pop_new[-1] = cy.get_better_agent(agent, pop[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.evaluate_agents(pop_new)
            pop_new = cy.greedy_agents(pop, pop_new, self.problem.sense)
        return pop_new

    def evolve_agents(self, epoch):
        pop = self.update_business_1__(self.objs, epoch)
        pop = self.update_business_2__(pop)
        pop = self.update_business_3__(pop, self.g_best)
        self.objs = self.opposition_based__(pop, self.g_best)

#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 10:21, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.native.collection.vectorize.human_based.QSA.DevQSA cimport DevQSA
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class OriginalQSA(DevQSA):
    """
    The original version of: Queuing Search Algorithm (QSA)

    Links:
       1. https://www.sciencedirect.com/science/article/abs/pii/S0307904X18302890

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
    >>> model = QSA.OriginalQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhang, J., Xiao, M., Gao, L. and Pan, Q., 2018. Queuing search algorithm: A novel metaheuristic algorithm
    for solving engineering optimization problems. Applied Mathematical Modelling, 63, pp.464-490.
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

    def update_business_3__(self, pop, g_best):
        pr = [idx / self.pop_size for idx in range(1, self.pop_size + 1)]
        pop_new = []
        for idx in range(self.pop_size):
            pos_new = pop[idx].solution.copy()
            for jdx in range(self.problem.n_dims):
                if self.generator.random() > pr[idx]:
                    i1, i2 = self.generator.choice(self.pop_size, 2, replace=False)
                    e = self.generator.exponential(0.5)
                    X1 = pop[i1].solution
                    X2 = pop[i2].solution
                    pos_new[jdx] = X1[jdx] + e * (X2[jdx] - pop[idx].solution[jdx])
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                pop_new[-1] = self._get_better_agent(
                    agent, pop[idx], self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            pop_new = self._greedy_selection_population(
                pop, pop_new, self.problem.sense
            )
        return pop_new

    def _evolve_agents(self, epoch):
        pop = self.update_business_1__(self.objs, epoch)
        pop = self.update_business_2__(pop)
        self.objs = self.update_business_3__(pop, self.g_best)

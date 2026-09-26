#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class OriginalSFO(AgentListOptimizer):
    """
    The original version of: SailFish Optimizer (SFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.01.001

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        + AP (float): coefficient for decreasing the value of Attack Power linearly from AP to 0
        + epsilon (float): should be 0.0001, 0.001

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import SFO    >>> import numpy as np
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
    >>> model = SFO.OriginalSFO(epoch=1000, pop_size=50, pp = 0.1, AP = 4.0, epsilon = 0.0001)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shadravan, S., Naji, H.R. and Bardsiri, V.K., 2019. The Sailfish Optimizer: A novel nature-inspired metaheuristic
    algorithm for solving constrained engineering optimization problems. Engineering Applications of Artificial Intelligence, 80, pp.20-34.
    """

    cdef public object pp
    cdef public object AP
    cdef public object epsilon
    cdef public object s_size
    cdef public object s_pop
    cdef public object s_gbest

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pp: float = 0.1,
        AP: float = 4.0,
        epsilon: float = 0.0001,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, SailFish pop size
            pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
            AP (float): coefficient for decreasing the value of Power Attack linearly from AP to 0
            epsilon (float): should be 0.0001, 0.001
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pp", "AP", "epsilon"],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pp = cy.validator(float, pp, (0, 1.0), "pp")
        self.AP = cy.validator(float, AP, (0, 100), "AP")
        self.epsilon = cy.validator(float, epsilon, (0, 0.1), "epsilon")
        self.s_size = int(self.pop_size / self.pp)

    def _initialization(self):
        AgentListOptimizer._initialization(self)
        if self.objs is None:
            self.objs = self._generate_agents(self.pop_size)  # pop = sailfish
        self.s_pop = self._generate_agents(self.s_size)
        self.s_gbest = self._get_best_agent(
            self.s_pop, self.problem.sense
        )  # s_pop = sardines
        self.pop = self.mirror__()

    def _evolve_agents(self, epoch):
        ## Calculate lamda_i using Eq.(7)
        ## Update the position of sailfish using Eq.(6)
        pop_new = []
        PD = 1 - self.pop_size / (self.pop_size + self.s_size)
        for idx in range(0, self.pop_size):
            lamda_i = 2 * self.generator.uniform() * PD - PD
            pos_new = self.s_gbest.solution - lamda_i * (
                    self.generator.uniform()
                    * (self.objs[idx].solution + self.s_gbest.solution)
                    / 2
                    - self.objs[idx].solution
            )
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.objs[idx] = self._get_better_agent(
                    self.objs[idx], agent, self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            self.objs = self._greedy_selection_population(
                self.objs, pop_new, self.problem.sense
            )
        ## Calculate AttackPower using Eq.(10)
        AP = self.AP * (1.0 - 2.0 * epoch * self.epsilon)
        if AP < 0.5:
            alpha = int(self.s_size * np.abs(AP))
            beta = int(self.problem.n_dims * np.abs(AP))
            ### Random self.generator.choice number of sardines which will be updated their position
            list1 = self.generator.choice(range(0, self.s_size), alpha)
            for idx in range(0, self.s_size):
                if idx in list1:
                    #### Random self.generator.choice number of dimensions in sardines updated, remove third loop by numpy vector computation
                    pos_new = self.s_pop[idx].solution.copy()
                    list2 = self.generator.choice(
                        range(0, self.problem.n_dims), beta, replace=False
                    )
                    pos_new[list2] = (
                            self.generator.uniform(0, 1, self.problem.n_dims)
                            * (self.s_gbest.solution - self.s_pop[idx].solution + AP)
                    )[list2]
                    pos_new = self._correct_solution(pos_new)
                    agent = self._generate_empty_agent(pos_new)
                    if self.mode not in self.AVAILABLE_MODES:
                        agent.target = self._get_target(pos_new)
                        self.s_pop[idx] = agent
        else:
            ### Update the position of all sardine using Eq.(9)
            for idx in range(0, self.s_size):
                pos_new = self.generator.uniform() * (
                        self.g_best.solution - self.s_pop[idx].solution + AP
                )
                pos_new = self._correct_solution(pos_new)
                agent = self._generate_empty_agent(pos_new)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self._get_target(pos_new)
                    self.s_pop[idx] = agent
        ## Recalculate the fitness of all sardine
        self.s_pop = self._update_target_for_population(self.s_pop)
        ## Sort the population of sailfish and sardine (for reducing computational cost)
        self.objs = self._get_sorted_and_trimmed_population(
            self.objs, self.pop_size, self.problem.sense
        )
        self.s_pop = self._get_sorted_and_trimmed_population(
            self.s_pop, len(self.s_pop), self.problem.sense
        )
        for idx in range(0, self.pop_size):
            for jdx in range(0, self.s_size):
                ### If there is a better position in sardine population.
                if self._compare_target(
                        self.s_pop[jdx].target, self.objs[idx].target, self.problem.sense
                ):
                    self.objs[idx] = self.s_pop[jdx].copy()
                    del self.s_pop[jdx]
                break  #### This simple keyword helped reducing ton of comparing operation.
                #### Especially when sardine pop size >> sailfish pop size
        temp = self.s_size - len(self.s_pop)
        if temp == 1:
            self.s_pop = self.s_pop + [self._generate_agent()]
        else:
            self.s_pop = self.s_pop + self._generate_agents(
                self.s_size - len(self.s_pop)
            )
        self.s_gbest = self._get_best_agent(self.s_pop, self.problem.sense)

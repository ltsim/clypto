#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
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


cdef class ImprovedNMRA(AgentListOptimizer):
    """
    The developed version of: Improved Naked Mole-Rat Algorithm (I-NMRA)

    Notes:
        + Use mutation probability idea
        + Use crossover operator
        + Use Levy-flight technique

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.5, 0.95], probability of breeding, default = 0.75
        + pm (float): [0.01, 0.1], probability of mutation, default = 0.01

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import NMRA    >>> import numpy as np
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
    >>> model = NMRA.ImprovedNMRA(epoch=1000, pop_size=50, pb = 0.75, pm = 0.01)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    cdef public object pb
    cdef public object pm
    cdef public object size_b

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        pb = 0.75,
        pm = 0.01,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pb (float): breeding probability, default = 0.75
            pm (float): probability of mutation, default = 0.01
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pb", "pm"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pb = cy.validator(float, pb, (0, 1.0), "pb")
        self.pm = cy.validator(float, pm, (0, 1.0), "pm")
        self.size_b = int(self.pop_size / 5)

    def crossover_random__(self, pop, g_best):
        start_point = self.generator.integers(0, self.problem.n_dims / 2)
        id1 = start_point
        id2 = int(start_point + self.problem.n_dims / 3)
        id3 = int(self.problem.n_dims)

        partner = pop[self.generator.integers(0, self.pop_size)].solution
        new_temp = g_best.solution.copy()
        new_temp[0:id1] = g_best.solution[0:id1]
        new_temp[id1:id2] = partner[id1:id2]
        new_temp[id2:id3] = g_best.solution[id2:id3]
        return new_temp

    def evolve_agents(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            # Exploration
            if idx < self.size_b:  # breeding operators
                if self.generator.uniform() < self.pb:
                    pos_new = self.objs[idx].solution + self.generator.normal(
                        0, 1, self.problem.n_dims
                    ) * (self.g_best.solution - self.objs[idx].solution)
                else:
                    levy_step = self.get_levy_flight_step(
                        beta=1, multiplier=0.001, case=-1
                    )
                    pos_new = self.objs[idx].solution + 1.0 / np.sqrt(epoch) * np.sign(
                        self.generator.random() - 0.5
                    ) * levy_step * (self.objs[idx].solution - self.g_best.solution)
            # Exploitation
            else:  # working operators
                if self.generator.uniform() < 0.5:
                    t1, t2 = self.generator.choice(
                        range(self.size_b, self.pop_size), 2, replace=False
                    )
                    pos_new = self.objs[idx].solution + self.generator.normal(
                        0, 1, self.problem.n_dims
                    ) * (self.objs[t1].solution - self.objs[t2].solution)
                else:
                    pos_new = self.crossover_random__(self.objs, self.g_best)
            # Mutation
            temp = self.generator.uniform(self.problem.lb, self.problem.ub)
            condition = self.generator.uniform(0, 1, self.problem.n_dims) < self.pm
            pos_new = np.where(condition, temp, pos_new)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    self.objs[idx], agent, self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )

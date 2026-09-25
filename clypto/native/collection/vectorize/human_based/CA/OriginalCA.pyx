#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 12:09, 02/03/2021 ----------%
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


cdef class OriginalCA(AgentListOptimizer):
    """
    The original version of: Culture Algorithm (CA)

    Links:
        1. https://github.com/clever-algorithms/CleverAlgorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + accepted_rate (float): [0.1, 0.5], probability of accepted rate, default: 0.15

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import CA    >>> import numpy as np
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
    >>> model = CA.OriginalCA(epoch=1000, pop_size=50, accepted_rate = 0.15)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Chen, B., Zhao, L. and Lu, J.H., 2009, April. Wind power forecast using RBF network and culture algorithm.
    In 2009 International Conference on Sustainable Power Generation and Supply (pp. 1-6). IEEE.
    """

    cdef public object accepted_rate
    cdef public object dyn_belief_space
    cdef public object dyn_accepted_num

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        accepted_rate: float = 0.15,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            accepted_rate (float): probability of accepted rate, default: 0.15
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "accepted_rate"],
            sort_flag=True,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.accepted_rate = cy.validator(float, accepted_rate, (0, 1.0), "accepted_rate")

    cdef void initialize_variables(self):
        ## Dynamic variables
        self.dyn_belief_space = {
            "lb": self.problem.lb,
            "ub": self.problem.ub,
        }
        self.dyn_accepted_num = int(self.accepted_rate * self.pop_size)

    def create_faithful__(self, lb, ub):
        pos = self.generator.uniform(lb, ub)
        return self.generate_agent(pos)

    def update_belief_space__(self, belief_space, pop_accepted):
        pos_list = np.array([agent.solution for agent in pop_accepted])
        belief_space["lb"] = np.min(pos_list, axis=0)
        belief_space["ub"] = np.max(pos_list, axis=0)
        return belief_space

    def evolve_agents(self, epoch):
        # create next generation
        pop_child = [
            self.create_faithful__(
                self.dyn_belief_space["lb"], self.dyn_belief_space["ub"]
            )
            for _ in range(0, self.pop_size)
        ]
        # select next generation
        pop_new = []
        pop_full = self.objs + pop_child
        size_new = len(pop_full)
        for _ in range(0, self.pop_size):
            id1, id2 = self.generator.choice(list(range(0, size_new)), 2, replace=False)
            agent = self.get_better_agent(
                pop_full[id1], pop_full[id2], self.problem.minmax
            )
            pop_new.append(agent)
        self.objs = self.get_sorted_population(pop_new, self.problem.minmax)
        # Get accepted faithful
        accepted = self.objs[: self.dyn_accepted_num]
        # Update belief_space
        self.dyn_belief_space = self.update_belief_space__(
            self.dyn_belief_space, accepted
        )

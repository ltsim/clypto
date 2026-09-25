#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 18:41, 08/04/2020 ----------%
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


cdef class OriginalEHO(AgentListOptimizer):
    """
    The original version of: Elephant Herding Optimization (EHO)

    Links:
        1. https://doi.org/10.1109/ISCBI.2015.8

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.3, 0.8], a factor that determines the influence of the best in each clan, default=0.5
        + beta (float): [0.3, 0.8], a factor that determines the influence of the x_center, default=0.5
        + n_clans (int): [3, 10], the number of clans, default=5

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import EHO    >>> import numpy as np
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
    >>> model = EHO.OriginalEHO(epoch=1000, pop_size=50, alpha = 0.5, beta = 0.5, n_clans = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2015, December. Elephant herding optimization.
    In 2015 3rd international symposium on computational and business intelligence (ISCBI) (pp. 1-5). IEEE.
    """

    cdef public object alpha
    cdef public object beta
    cdef public object n_clans
    cdef public object n_individuals
    cdef public object pop_group

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        alpha: float = 0.5,
        beta: float = 0.5,
        n_clans: int = 5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): a factor that determines the influence of the best in each clan, default=0.5
            beta (float): a factor that determines the influence of the x_center, default=0.5
            n_clans (int): the number of clans, default=5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "alpha", "beta", "n_clans"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.alpha = cy.validator(float, alpha, (0, 3.0), "alpha")
        self.beta = cy.validator(float, beta, (0, 1.0), "beta")
        self.n_clans = cy.validator(int, n_clans, [2, int(self.pop_size / 5)], "n_clans")
        self.n_individuals = int(self.pop_size / self.n_clans)

    cdef void initialization(self):
        AgentListOptimizer.initialization(self)
        self.pop_group = self.generate_group_population(
            self.objs, self.n_clans, self.n_individuals
        )
        self.pop = self.mirror__()

    def evolve_agents(self, epoch):
        # Clan updating operator
        pop_new = []
        for idx in range(0, self.pop_size):
            clan_idx = int(idx / self.n_individuals)
            pos_clan_idx = int(idx % self.n_individuals)
            if (
                    pos_clan_idx == 0
            ):  # The best in clan, because all clans are sorted based on fitness
                center = np.mean(
                    np.array([agent.solution for agent in self.pop_group[clan_idx]]),
                    axis=0,
                )
                pos_new = self.beta * center
            else:
                pos_new = self.pop_group[clan_idx][
                              pos_clan_idx
                          ].solution + self.alpha * self.generator.random() * (
                                  self.pop_group[clan_idx][0].solution
                                  - self.pop_group[clan_idx][pos_clan_idx].solution
                          )
            pos_new = self.correct_solution(pos_new)
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
                self.objs, pop_new, self.problem.minmax
            )
        self.pop_group = self.generate_group_population(
            self.objs, self.n_clans, self.n_individuals
        )
        # Separating operator
        for idx in range(0, self.n_clans):
            self.pop_group[idx] = self.get_sorted_population(
                self.pop_group[idx], self.problem.minmax
            )
            self.pop_group[idx][-1] = self.generate_agent()
        self.objs = [agent for pack in self.pop_group for agent in pack]

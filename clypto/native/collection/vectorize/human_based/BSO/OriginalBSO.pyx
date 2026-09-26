#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 07:44, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.human_based.BSO.ImprovedBSO cimport ImprovedBSO
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class OriginalBSO(ImprovedBSO):
    """
    The original version of: Brain Storm Optimization (BSO)

    Links:
        1. https://doi.org/10.1007/978-3-642-21515-5_36

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + m_clusters (int): [3, 10], number of clusters (m in the paper)
        + p1 (float): [0.1, 0.5], probability
        + p2 (float): [0.5, 0.95], probability
        + p3 (float): [0.2, 0.8], probability
        + p4 (float): [0.2, 0.8], probability
        + slope (int): [10, 15, 20, 25], changing logsig() function's slope (k: in the paper)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import BSO    >>> import numpy as np
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
    >>> model = BSO.OriginalBSO(epoch=1000, pop_size=50, m_clusters = 5, p1 = 0.2, p2 = 0.8, p3 = 0.4, p4 = 0.5, slope = 20)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shi, Y., 2011, June. Brain storm optimization algorithm. In International
    conference in swarm intelligence (pp. 303-309). Springer, Berlin, Heidelberg.
    """

    cdef public object slope

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        m_clusters: int = 5,
        p1: float = 0.2,
        p2: float = 0.8,
        p3: float = 0.4,
        p4: float = 0.5,
        slope: int = 20,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            m_clusters (int): number of clusters (m in the paper)
            p1 (float): probability
            p2 (float): probability
            p3 (float): probability
            p4 (float): probability
            slope (int): changing logsig() function's slope (k: in the paper)
        """
        super().__init__(epoch, pop_size, m_clusters, p1, p2, p3, p4, name=name, mode=mode)
        self._params_name_ordered = tuple(["epoch", "pop_size", "m_clusters", "p1", "p2", "p3", "p4", "slope"])
        self.slope = cy.validator(int, slope, [10, 50], "slope")

    cdef object _amend_solution(self, object solution):
        rp = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
        condition = np.logical_and(
            self.problem.bounds.low <= solution, solution <= self.problem.bounds.up
        )
        return np.where(condition, solution, rp)

    def _evolve_agents(self, epoch):
        x = (0.5 * self.epoch - epoch) / self.slope
        epsilon = self.generator.uniform() * (1 / (1 + np.exp(-x)))
        if self.generator.random() < self.p1:  # p_5a
            idx = self.generator.integers(0, self.m_clusters)
            self.centers[idx] = self._generate_agent()
        pop_group = self.pop_group
        for idx in range(0, self.pop_size):  # Generate new individuals
            cluster_id = int(idx / self.m_solution)
            location_id = int(idx % self.m_solution)
            if self.generator.uniform() < self.p2:  # p_6b
                if self.generator.uniform() < self.p3:  # p_6i
                    cluster_id = self.generator.integers(0, self.m_clusters)
                if self.generator.uniform() < self.p3:
                    pos_new = self.centers[
                                  cluster_id
                              ].solution + epsilon * self.generator.normal(
                        0, 1, self.problem.n_dims
                    )
                else:
                    rand_idx = self.generator.integers(0, self.m_solution)
                    pos_new = self.pop_group[cluster_id][
                                  rand_idx
                              ].solution + self.generator.normal(0, 1, self.problem.n_dims)
            else:
                id1, id2 = self.generator.choice(
                    range(0, self.m_clusters), 2, replace=False
                )
                if self.generator.uniform() < self.p4:
                    pos_new = 0.5 * (
                            self.centers[id1].solution + self.centers[id2].solution
                    ) + epsilon * self.generator.normal(0, 1, self.problem.n_dims)
                else:
                    rand_id1 = self.generator.integers(0, self.m_solution)
                    rand_id2 = self.generator.integers(0, self.m_solution)
                    pos_new = 0.5 * (
                            self.pop_group[id1][rand_id1].solution
                            + self.pop_group[id2][rand_id2].solution
                    ) + epsilon * self.generator.normal(0, 1, self.problem.n_dims)
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_group[cluster_id][location_id] = agent
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                pop_group[cluster_id][location_id] = self._get_better_agent(
                    agent, self.pop_group[cluster_id][location_id], self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            for idx in range(0, self.m_clusters):
                pop_group[idx] = self._update_target_for_population(pop_group[idx])
                pop_group[idx] = self._greedy_selection_population(
                    self.pop_group[idx], pop_group[idx], self.problem.sense
                )
        # Needed to update the centers and population
        self.centers = self.find_cluster__(pop_group)
        self.objs = []
        for idx in range(0, self.m_clusters):
            self.objs += pop_group[idx]

#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 07:44, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class ImprovedBSO(AgentListOptimizer):
    """
    The improved version: Improved Brain Storm Optimization (IBSO)

    Notes:
        + Remove some probability parameters, and some unnecessary equations.
        + The Levy-flight technique is employed to enhance the algorithm's robustness and resilience in challenging environments.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + m_clusters (int): [3, 10], number of clusters (m in the paper)
        + p1 (float): 25% percent
        + p2 (float): 50% percent changed by its own (local search), 50% percent changed by outside (global search)
        + p3 (float): 75% percent develop the old idea, 25% invented new idea based on levy-flight
        + p4 (float): [0.4, 0.6], Need more weights on the centers instead of the random position

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import BSO    >>> import numpy as np
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
    >>> model = BSO.ImprovedBSO(epoch=1000, pop_size=50, m_clusters = 5, p1 = 0.25, p2 = 0.5, p3 = 0.75, p4 = 0.6)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] El-Abd, M. (2017). Global-best brain storm optimization algorithm. Swarm and evolutionary computation, 37, 27-44.
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        m_clusters: int = 5,
        p1: float = 0.25,
        p2: float = 0.5,
        p3: float = 0.75,
        p4: float = 0.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            m_clusters (int): number of clusters (m in the paper)
            p1 (float): 25% percent
            p2 (float): 50% percent changed by its own (local search), 50% percent changed by outside (global search)
            p3 (float): 75% percent develop the old idea, 25% invented new idea based on levy-flight
            p4 (float): Need more weights on the centers instead of the random position
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "m_clusters", "p1", "p2", "p3", "p4"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.m_clusters = cy.validator(int, m_clusters, [2, int(self.pop_size / 5)], "m_clusters")
        self.p1 = cy.validator(float, p1, (0, 1.0), "p1")
        self.p2 = cy.validator(float, p2, (0, 1.0), "p2")
        self.p3 = cy.validator(float, p3, (0, 1.0), "p3")
        self.p4 = cy.validator(float, p4, (0, 1.0), "p4")
        self.m_solution = int(self.pop_size / self.m_clusters)
        self.pop_group, self.centers = None, None

    def find_cluster__(self, pop_group):
        centers = []
        for idx in range(0, self.m_clusters):
            local_best = self.get_best_agent(pop_group[idx], self.problem.minmax)
            centers.append(local_best.copy())
        return centers

    cdef void initialization(self):
        AgentListOptimizer.initialization(self)
        self.pop_group = self.generate_group_population(
            self.objs, self.m_clusters, self.m_solution
        )
        self.centers = self.find_cluster__(self.pop_group)
        self.pop = self.mirror__()

    def evolve_agents(self, epoch):
        epsilon = 1.0 - 1.0 * epoch / self.epoch  # 1. Changed here, no need: k
        if self.generator.uniform() < self.p1:  # p_5a
            idx = self.generator.integers(0, self.m_clusters)
            self.centers[idx] = self.generate_agent()
        pop_group = self.pop_group
        for idx in range(0, self.pop_size):  # Generate new individuals
            cluster_id = int(idx / self.m_solution)
            location_id = int(idx % self.m_solution)

            if self.generator.uniform() < self.p2:  # p_6b
                if self.generator.uniform() < self.p3:
                    pos_new = self.centers[
                                  cluster_id
                              ].solution + epsilon * self.generator.normal(
                        0, 1, self.problem.n_dims
                    )
                else:  # 2. Using levy flight here
                    levy_step = self.get_levy_flight_step(
                        beta=1.0, multiplier=0.001, size=self.problem.n_dims, case=-1
                    )
                    pos_new = (
                            self.pop_group[cluster_id][location_id].solution + levy_step
                    )
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
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_group[cluster_id][location_id] = agent
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop_group[cluster_id][location_id] = self.get_better_agent(
                    agent, self.pop_group[cluster_id][location_id], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            for idx in range(0, self.m_clusters):
                pop_group[idx] = self.update_target_for_population(pop_group[idx])
                pop_group[idx] = self.greedy_selection_population(
                    self.pop_group[idx], pop_group[idx], self.problem.minmax
                )

        # Needed to update the centers and population
        self.centers = self.find_cluster__(pop_group)
        self.objs = []
        for idx in range(0, self.m_clusters):
            self.objs += pop_group[idx]

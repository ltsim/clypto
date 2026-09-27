cimport clypto.core as cy
#!/usr/bin/env python
# Created by "Thieu" at 07:44, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%



cdef class ImprovedBSO(cy.Optimizer):
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
    >>> from clypto.native.collection.human_based import BSO    >>> import numpy as np
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
    >>> model = BSO.ImprovedBSO(epoch=1000, pop_size=50, m_clusters = 5, p1 = 0.25, p2 = 0.5, p3 = 0.75, p4 = 0.6)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

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
            **kwargs: object
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
        super().__init__(parameters=["epoch", "pop_size", "m_clusters", "p1", "p2", "p3", "p4"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[10, 10000])
        self.m_clusters = cy.validator(int, m_clusters, [2, int(self.population.size() / 5)], "m_clusters")
        self.p1 = cy.validator(float, p1, (0, 1.0), "p1")
        self.p2 = cy.validator(float, p2, (0, 1.0), "p2")
        self.p3 = cy.validator(float, p3, (0, 1.0), "p3")
        self.p4 = cy.validator(float, p4, (0, 1.0), "p4")
        self.m_solution = int(self.population.size() / self.m_clusters)
        self.pop_group, self.centers = None, None

    def find_cluster__(self, pop_group):
        centers = []
        for idx in range(0, self.m_clusters):
            local_best = cy.duplicate_agent(cy.sort_agents(pop_group[idx], self.problem.sense)[0])
            centers.append(cy.duplicate_agent(local_best))
        return centers

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        self.pop_group = cy.split_groups(self.population, self.m_clusters, self.m_solution)
        self.centers = self.find_cluster__(self.pop_group)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        epsilon = 1.0 - 1.0 * epoch / self.epoch  # 1. Changed here, no need: k
        if self.generator.uniform() < self.p1:  # p_5a
            idx = self.generator.integers(0, self.m_clusters)
            self.centers[idx] = self.population.generate_agent()
        pop_group = self.pop_group
        for idx in range(0, pop_size):  # Generate new individuals
            cluster_id = int(idx / self.m_solution)
            location_id = int(idx % self.m_solution)

            if self.generator.uniform() < self.p2:  # p_6b
                if self.generator.uniform() < self.p3:
                    x = self.centers[
                                  cluster_id
                              ].solution + epsilon * self.generator.normal(
                        0, 1, self.problem.n_dims
                    )
                else:  # 2. Using levy flight here
                    levy_step = cy.levy_flight(self.generator, beta=1.0, multiplier=0.001, size=self.problem.n_dims, case=-1)
                    x = (
                            self.pop_group[cluster_id][location_id].solution + levy_step
                    )
            else:
                id1, id2 = self.generator.choice(
                    range(0, self.m_clusters), 2, replace=False
                )
                if self.generator.uniform() < self.p4:
                    x = 0.5 * (
                            self.centers[id1].solution + self.centers[id2].solution
                    ) + epsilon * self.generator.normal(0, 1, self.problem.n_dims)
                else:
                    rand_id1 = self.generator.integers(0, self.m_solution)
                    rand_id2 = self.generator.integers(0, self.m_solution)
                    x = 0.5 * (
                            self.pop_group[id1][rand_id1].solution
                            + self.pop_group[id2][rand_id2].solution
                    ) + epsilon * self.generator.normal(0, 1, self.problem.n_dims)
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop_group[cluster_id][location_id] = agent
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                pop_group[cluster_id][location_id] = cy.get_better_agent(agent, self.pop_group[cluster_id][location_id], self.problem.sense)
        if self.mode != "sequential":
            for idx in range(0, self.m_clusters):
                pop_group[idx] = self.population.evaluate(pop_group[idx], self.mode)
                pop_group[idx] = cy.greedy_agents(self.pop_group[idx], pop_group[idx], self.problem.sense)

        # Needed to update the centers and population
        self.centers = self.find_cluster__(pop_group)
        self.population = self.population.spawn([])
        for idx in range(0, self.m_clusters):
            self.population += pop_group[idx]

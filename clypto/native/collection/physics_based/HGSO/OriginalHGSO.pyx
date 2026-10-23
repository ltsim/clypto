#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalHGSO(cy.Optimizer):
    """
    The original version of: Henry Gas Solubility Optimization (HGSO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0167739X19306557

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_clusters (int): [2, 10], number of clusters, default = 2

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.physics_based import HGSO    >>> import numpy as np
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
    >>> model = HGSO.OriginalHGSO(epoch=1000, pop_size=50, n_clusters = 3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F.A., Houssein, E.H., Mabrouk, M.S., Al-Atabany, W. and Mirjalili, S., 2019. Henry gas solubility
    optimization: A novel physics-based algorithm. Future Generation Computer Systems, 101, pp.646-667.
    """

    cdef public int n_clusters

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            n_clusters: int = 2,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_clusters (int): number of clusters, default = 2
        """
        super().__init__(parameters=["epoch", "pop_size", "n_clusters"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[10, 10000])
        self.n_clusters = cy.validator(int, n_clusters, [2, int(self.population.size() / 5)], "n_clusters")
        self.n_elements = int(self.population.size() / self.n_clusters)
        self.T0 = 298.15
        self.K = 1.0
        self.beta = 1.0
        self.alpha = 1
        self.epsilon = 0.05
        self.l1 = 5e-2
        self.l2 = 100.0
        self.l3 = 1e-2

    def initialize_variables(self):
        self.H_j = self.l1 * self.generator.uniform()
        self.P_ij = self.l2 * self.generator.uniform()
        self.C_j = self.l3 * self.generator.uniform()
        self.pop_group, self.p_best = None, None

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        self.pop_group = cy.split_groups(self.population, self.n_clusters, self.n_elements)
        self.p_best = self.get_best_solution_in_team__(
            self.pop_group
        )  # multiple element

    def flatten_group__(self, group):
        pop = []
        for idx in range(0, self.n_clusters):
            pop += group[idx]
        return pop

    def get_best_solution_in_team__(self, group=None):
        list_best = []
        for idx in range(len(group)):
            best_agent = cy.duplicate_agent(cy.sort_agents(group[idx], self.problem.sense)[0])
            list_best.append(best_agent)
        return list_best

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Loop based on the number of cluster in swarm (number of gases type)
        for idx in range(self.n_clusters):
            ### Loop based on the number of individual in each gases type
            pop_new = []
            for jdx in range(self.n_elements):
                F = -1.0 if self.generator.uniform() < 0.5 else 1.0
                ##### Based on Eq. 8, 9, 10
                self.H_j = self.H_j * np.exp(
                    -self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0)
                )
                S_ij = self.K * self.H_j * self.P_ij
                gama = self.beta * np.exp(
                    -(
                            (self.p_best[idx].fitness + self.epsilon)
                            / (self.pop_group[idx][jdx].fitness + self.epsilon)
                    )
                )
                pos_new = (
                        self.pop_group[idx][jdx].solution
                        + F
                        * self.generator.uniform()
                        * gama
                        * (self.p_best[idx].solution - self.pop_group[idx][jdx].solution)
                        + F
                        * self.generator.uniform()
                        * self.alpha
                        * (S_ij * self.g_best.solution - self.pop_group[idx][jdx].solution)
                )
                pos_new = cy.correct_solution(self.problem, pos_new)
                agent = self.population.create_agent(pos_new)
                pop_new.append(agent)
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.pop_group[idx] = pop_new
        self.population = self.population.spawn(self.flatten_group__(self.pop_group))

        ## Update Henry's coefficient using Eq.8
        self.H_j = self.H_j * np.exp(
            -self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0)
        )
        ## Update the solubility of each gas using Eq.9
        S_ij = self.K * self.H_j * self.P_ij
        ## Rank and select the number of worst agents using Eq. 11
        N_w = int(pop_size * (self.generator.uniform(0, 0.1) + 0.1))
        ## Update the position of the worst agents using Eq. 12
        sorted_id_pos = np.argsort([x.fitness for x in self.population])

        pop_new = []
        pop_idx = []
        for item in range(N_w):
            id = sorted_id_pos[item]
            pos_new = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
            pos_new = cy.correct_solution(self.problem, pos_new)
            agent = self.population.create_agent(pos_new)
            pop_idx.append(id)
            pop_new.append(agent)
        pop_new = self.population.evaluate(pop_new, self.mode)
        for idx, id_selected in enumerate(pop_idx):
            self.population[id_selected] = cy.duplicate_agent(pop_new[idx])
        self.pop_group = cy.split_groups(self.population, self.n_clusters, self.n_elements)
        self.p_best = self.get_best_solution_in_team__(self.pop_group)

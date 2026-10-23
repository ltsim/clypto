#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DS_GWO(cy.Optimizer):
    """
    The original version of: Diversity enhanced Strategy based Grey Wolf Optimizer (DS-GWO)

    This implementation includes:
        1. Group-stage competition mechanism
        2. Exploration-exploitation balance mechanism

    Links:
        1. https://doi.org/10.1016/j.knosys.2022.109100

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.DS_GWO(epoch=1000, pop_size=50, explore_ratio=0.4, n_groups=5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Jiang, Jianhua, Ziying Zhao, Yutong Liu, Weihua Li, and Huan Wang. "DSGWO: An improved grey wolf optimizer with diversity enhanced strategy based on group-stage competition and balance mechanisms." Knowledge-Based Systems 250 (2022): 109100.
    """

    cdef public double explore_ratio
    cdef public int n_groups

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        explore_ratio: float = 0.4,
        n_groups: int = 5,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            explore_ratio (float): ratio to control exploration, default = 0.4
            n_groups (int): number of groups for group-stage competition, default = 5
        """
        super().__init__(parameters=["epoch", "pop_size", "explore_ratio", "n_groups"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[10, 10000])
        self.explore_ratio = cy.validator(float, explore_ratio, [0.0, 1.0], "explore_ratio")
        self.n_groups = cy.validator(int, n_groups, [5, 100], "n_groups")

    def initialize_variables(self):
        """
        Initialize any variables needed for the algorithm.
        """
        self.explore_epoch = int(self.epoch * self.explore_ratio)

    def before_main_loop(self):
        """
        Initialize variables before the main loop starts.
        """
        self.group_stage_competition()

    def get_coefficients(self, a: float) -> tuple:
        """
        Generate coefficients A and C for position update equations.

        Args:
            a (float): Coefficient that decreases over epochs

        Returns:
            tuple: Coefficients A, C
        """
        A = a * (2 * self.generator.random(self.problem.n_dims) - 1)
        C = 2 * self.generator.random(self.problem.n_dims)
        return A, C

    def group_stage_competition(self):
        """
        Group-stage competition mechanism:
        1. Divide population into 6 subgroups
        2. Select best wolf from each subgroup as delta candidates
        3. Set best overall as alpha
        4. Set delta candidate farthest from alpha as beta
        """
        pop_size = self.population.size()
        # Divide population into n_groups
        group_size = pop_size // self.n_groups
        self.delta_candidates = []

        for idx in range(self.n_groups):
            start_idx = idx * group_size
            if idx == self.n_groups - 1:  # Last group takes remaining wolves
                end_idx = pop_size
            else:
                end_idx = (idx + 1) * group_size

            # Get group members
            group_population = self.population[start_idx:end_idx]
            # Find best wolf in group
            group_sorted = cy.sort_agents(group_population, self.problem.sense)
            self.delta_candidates.append(cy.duplicate_agent(group_sorted[0]))

        # Set alpha wolf (best among all delta candidates)
        ranked = cy.sort_agents(self.delta_candidates, self.problem.sense)
        list_best = [cy.duplicate_agent(agent) for agent in ranked[:1]]
        self.alpha = cy.duplicate_agent(list_best[0])

        # Set beta wolf (delta candidate farthest from alpha)
        delta_pos = np.array([agent.solution for agent in self.delta_candidates])
        distances = np.linalg.norm(delta_pos - self.alpha.solution, axis=1)
        beta_idx = np.argmax(distances)
        self.beta = cy.duplicate_agent(self.delta_candidates[beta_idx])

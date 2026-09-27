cimport clypto.core as cy
#!/usr/bin/env python
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%



cdef class OriginalBeesA(cy.Optimizer):
    """
    The original version of: Bees Algorithm (BeesA)

    Links:
        1. https://www.sciencedirect.com/science/article/pii/B978008045157250081X
        2. https://www.tandfonline.com/doi/full/10.1080/23311916.2015.1091540

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + selected_site_ratio (float): default = 0.5
        + elite_site_ratio (float): default = 0.4
        + selected_site_bee_ratio (float): default = 0.1
        + elite_site_bee_ratio (float): default = 2.0
        + dance_radius (float): default = 0.1
        + dance_reduction (float): default = 0.99

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import BeesA    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "sense": "min",
    >>> }
    >>>
    >>> model = BeesA.OriginalBeesA(epoch=1000, pop_size=50, selected_site_ratio=0.5, elite_site_ratio=0.4,
    >>>         selected_site_bee_ratio=0.1, elite_site_bee_ratio=2.0, dance_radius=0.1, dance_reduction=0.99)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Pham, D.T., Ghanbarzadeh, A., Koç, E., Otri, S., Rahim, S. and Zaidi, M., 2006. The bees algorithm—a novel tool
    for complex optimisation problems. In Intelligent production machines and systems (pp. 454-459). Elsevier Science Ltd.
    """

    cdef public double dance_radius
    cdef public double dance_reduction
    cdef public double elite_site_bee_ratio
    cdef public double elite_site_ratio
    cdef public double selected_site_bee_ratio
    cdef public double selected_site_ratio

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            selected_site_ratio: float = 0.5,
            elite_site_ratio: float = 0.4,
            selected_site_bee_ratio: float = 0.1,
            elite_site_bee_ratio: float = 2.0,
            dance_radius: float = 0.1,
            dance_reduction: float = 0.99,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            selected_site_ratio (float):
            elite_site_ratio (float):
            selected_site_bee_ratio (float):
            elite_site_bee_ratio (float):
            dance_radius (float):
            dance_reduction (float):
        """
        super().__init__(parameters=[ "epoch", "pop_size", "selected_site_ratio", "elite_site_ratio", "selected_site_bee_ratio", "elite_site_bee_ratio", "dance_radius", "dance_reduction", ], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        # (Scout Bee Count or Population Size, Selected Sites Count)
        self.selected_site_ratio = cy.validator(float, selected_site_ratio, (0, 1.0), "selected_site_ratio")
        self.elite_site_ratio = cy.validator(float, elite_site_ratio, (0, 1.0), "elite_site_ratio")
        self.selected_site_bee_ratio = cy.validator(float, selected_site_bee_ratio, (0, 1.0), "selected_site_bee_ratio")
        self.elite_site_bee_ratio = cy.validator(float, elite_site_bee_ratio, (0, 3.0), "elite_site_bee_ratio")
        self.dance_radius = cy.validator(float, dance_radius, (0, 1.0), "dance_radius")
        self.dance_reduction = cy.validator(float, dance_reduction, (0, 1.0), "dance_reduction")
        # Initial Value of Dance Radius
        self.dyn_radius = self.dance_radius
        self.n_selected_bees = int(round(self.selected_site_ratio * self.population.size()))
        self.n_elite_bees = int(round(self.elite_site_ratio * self.n_selected_bees))
        self.n_selected_bees_local = int(
            round(self.selected_site_bee_ratio * self.population.size())
        )
        self.n_elite_bees_local = int(
            round(self.elite_site_bee_ratio * self.n_selected_bees_local)
        )

    def perform_dance__(self, position, r):
        jdx = self.generator.choice(range(0, self.problem.n_dims))
        position[jdx] = position[jdx] + r * self.generator.uniform(-1, 1)
        return self.population.correct_solution(position)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_new = self.population.copy()
        for idx in range(0, pop_size):
            # Elite Sites
            if idx < self.n_elite_bees:
                pop_child = []
                for j in range(0, self.n_elite_bees_local):
                    pos_new = self.perform_dance__(
                        self.population[idx].solution, self.dyn_radius
                    )
                    agent = self.population.create_agent(pos_new)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].evaluate(self.problem)
                pop_child = self.population.evaluate(pop_child, self.mode)
                local_best = cy.sort_agents(pop_child, self.problem.sense)[0].copy()
                if cy.is_better(local_best, self.population[idx], self.problem.sense):
                    pop_new[idx] = local_best
            elif self.n_elite_bees <= idx < self.n_selected_bees:
                # Selected Non-Elite Sites
                pop_child = []
                for j in range(0, self.n_selected_bees_local):
                    pos_new = self.perform_dance__(
                        self.population[idx].solution, self.dyn_radius
                    )
                    agent = self.population.create_agent(pos_new)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].evaluate(self.problem)
                pop_child = self.population.evaluate(pop_child, self.mode)
                local_best = cy.sort_agents(pop_child, self.problem.sense)[0].copy()
                if cy.is_better(local_best, self.population[idx], self.problem.sense):
                    pop_new[idx] = local_best
            else:
                # Non-Selected Sites
                pop_new[idx] = self.population.generate_agent()
        self.population = pop_new
        # Damp Dance Radius
        self.dyn_radius = self.dance_reduction * self.dance_radius

#!/usr/bin/env python
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class ProbBeesA(cy.Optimizer):
    """
    The original version of: Probabilistic Bees Algorithm (P-BeesA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + recruited_bee_ratio (float): percent of bees recruited, default = 0.1
        + dance_factor (tuple, list): (radius, reduction) - Bees Dance Radius, default=(0.1, 0.99)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import BeesA    >>> import numpy as np
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
    >>> model = BeesA.ProbBeesA(epoch=1000, pop_size=50, recruited_bee_ratio = 0.1, dance_radius = 0.1, dance_reduction = 0.99)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Pham, D.T. and Castellani, M., 2015. A comparative study of the Bees Algorithm as a tool for
    function optimisation. Cogent Engineering, 2(1), p.1091540.
    """

    cdef public double dance_radius
    cdef public double dance_reduction
    cdef public double recruited_bee_ratio

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            recruited_bee_ratio: float = 0.1,
            dance_radius: float = 0.1,
            dance_reduction: float = 0.99,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            recruited_bee_ratio (float): percent of bees recruited, default = 0.1
            dance_radius (float): Bees Dance Radius, default=0.1
            dance_reduction (float): Bees Dance Radius Reduction Rate, default=0.99
        """
        super().__init__(parameters=[ "epoch", "pop_size", "recruited_bee_ratio", "dance_radius", "dance_reduction", ], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.recruited_bee_ratio = cy.validator(float, recruited_bee_ratio, (0, 1.0), "recruited_bee_ratio")
        self.dance_radius = cy.validator(float, dance_radius, (0, 1.0), "dance_radius")
        self.dance_reduction = cy.validator(float, dance_reduction, (0, 1.0), "dance_reduction")
        # Initial Value of Dance Radius
        self.dyn_radius = self.dance_radius
        self.recruited_bee_count = int(round(self.recruited_bee_ratio * self.population.size()))

    def perform_dance__(self, position, r):
        jdx = self.generator.choice(list(range(0, self.problem.n_dims)))
        position[jdx] = position[jdx] + r * self.generator.uniform(-1, 1)
        return cy.correct_solution(self.problem, position)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Calculate Scores
        fit_list = np.array([agent.fitness for agent in self.population])
        fit_list = 1.0 / (fit_list + self.EPSILON)
        d_fit = fit_list / np.mean(fit_list)
        for idx, agent in enumerate(self.population.toarray()):
            # Determine Rejection Probability based on Score
            if d_fit[idx] < 0.9:
                reject_prob = 0.6
            elif 0.9 <= d_fit[idx] < 0.95:
                reject_prob = 0.2
            elif 0.95 <= d_fit[idx] < 1.15:
                reject_prob = 0.05
            else:
                reject_prob = 0
            # Check for Acceptance/Rejection
            if self.generator.random() >= reject_prob:  # Acceptance
                # Calculate New Bees Count
                bee_count = int(np.ceil(d_fit[idx] * self.recruited_bee_count))
                if bee_count < 2:
                    bee_count = 2
                if bee_count > pop_size:
                    bee_count = pop_size
                # Create New Bees(Solutions)
                pop_child = []
                for j in range(0, bee_count):
                    x = self.perform_dance__(
                        agent.solution, self.dyn_radius
                    )
                    child = self.population.create_agent(x)
                    pop_child.append(child)
                pop_child = self.population.evaluate(pop_child, self.mode)
                local_best = cy.duplicate_agent(cy.sort_agents(pop_child, self.problem.sense)[0])
                if cy.is_better(local_best, agent, self.problem.sense):
                    self.population[idx] = local_best
            else:
                self.population[idx] = self.population.generate_agent()
        # Damp Dance Radius
        self.dyn_radius = self.dance_reduction * self.dance_radius

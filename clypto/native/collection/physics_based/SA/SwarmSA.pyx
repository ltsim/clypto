#!/usr/bin/env python
# Created by "Thieu" at 22:08, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class SwarmSA(cy.Optimizer):
    """
    The swarm version of: Simulated Annealing (SwarmSA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + max_sub_iter (int): [5, 10, 15], Maximum Number of Sub-Iteration (within fixed temperature), default=5
        + t0 (int): Fixed parameter, Initial Temperature, default=1000
        + t1 (int): Fixed parameter, Final Temperature, default=1
        + move_count (int): [5, 20], Move Count per Individual Solution, default=5
        + mutation_rate (float): [0.01, 0.2], Mutation Rate, default=0.1
        + mutation_step_size (float): [0.05, 0.1, 0.15], Mutation Step Size, default=0.1
        + mutation_step_size_damp (float): [0.8, 0.99], Mutation Step Size Damp, default=0.99

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.physics_based import SA    >>> import numpy as np
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
    >>> model = SA.SwarmSA(epoch=1000, pop_size=50, max_sub_iter = 5, t0 = 1000, t1 = 1,
    >>>         move_count = 5, mutation_rate = 0.1, mutation_step_size = 0.1, mutation_step_size_damp = 0.99)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Van Laarhoven, P.J. and Aarts, E.H., 1987. Simulated annealing. In Simulated
    annealing: Theory and applications (pp. 7-15). Springer, Dordrecht.
    """

    cdef public int max_sub_iter
    cdef public int move_count
    cdef public double mutation_rate
    cdef public double mutation_step_size
    cdef public double mutation_step_size_damp
    cdef public int t0
    cdef public int t1

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            max_sub_iter: int = 5,
            t0: int = 1000,
            t1: int = 1,
            move_count: int = 5,
            mutation_rate: float = 0.1,
            mutation_step_size: float = 0.1,
            mutation_step_size_damp: float = 0.99,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            max_sub_iter (int): Maximum Number of Sub-Iteration (within fixed temperature), default=5
            t0 (int): Initial Temperature, default=1000
            t1 (int): Final Temperature, default=1
            move_count (int): Move Count per Individual Solution, default=5
            mutation_rate (float): Mutation Rate, default=0.1
            mutation_step_size (float): Mutation Step Size, default=0.1
            mutation_step_size_damp (float): Mutation Step Size Damp, default=0.99
        """
        super().__init__(parameters=[ "epoch", "pop_size", "max_sub_iter", "t0", "t1", "move_count", "mutation_rate", "mutation_step_size", "mutation_step_size_damp", ], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.max_sub_iter = cy.validator(int, max_sub_iter, [1, 100000], "max_sub_iter")
        self.t0 = cy.validator(int, t0, [500, 2000], "t0")
        self.t1 = cy.validator(int, t1, [1, 100], "t1")
        self.move_count = cy.validator(int, move_count, [2, int(self.population.size() / 2)], "move_count")
        self.mutation_rate = cy.validator(float, mutation_rate, (0, 1.0), "mutation_rate")
        self.mutation_step_size = cy.validator(float, mutation_step_size, (0, 1.0), "mutation_step_size")
        self.mutation_step_size_damp = cy.validator(float, mutation_step_size_damp, (0, 1.0), "mutation_step_size_damp")
        self.dyn_t, self.t_damp, self.dyn_sigma = None, None, None

    def mutate__(self, position, sigma):
        # Select Mutating Variables
        x = position + sigma * self.generator.uniform(
            self.problem.bounds.low, self.problem.bounds.up
        )
        x = np.where(
            self.generator.random(self.problem.n_dims) < self.mutation_rate,
            position,
            x,
        )
        if np.all(x == position):  # Select at least one variable to mutate
            x[self.generator.integers(0, self.problem.n_dims)] = (
                self.generator.uniform()
            )
        return cy.correct_solution(self.problem, x)

    def initialization(self):
        pop_size = self.population.size()
        # Initial Temperature
        self.dyn_t = self.t0  # Initial Temperature
        self.t_damp = (self.t1 / self.t0) ** (
                1.0 / self.epoch
        )  # Calculate Temperature Damp Rate
        self.dyn_sigma = self.mutation_step_size  # Initial Value of Step Size
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Sub-Iterations
        for g in range(0, self.max_sub_iter):
            # Create new population
            pop_new = []
            for idx in range(0, pop_size):
                for j in range(0, self.move_count):
                    # Perform Mutation (Move)
                    x = self.mutate__(self.population[idx].solution, self.dyn_sigma)
                    x = cy.correct_solution(self.problem, x)
                    agent = self.population.create_agent(x)
                    pop_new.append(agent)
            pop_new = self.population.evaluate(pop_new, self.mode)
            # Columnize and Sort Newly Created Population
            pop_new = cy.sort_agents(pop_new, self.problem.sense)[:pop_size]
            # Randomized Selection
            for idx, agent in enumerate(self.population.toarray()):
                # Check if new solution is better than current
                if cy.is_better(pop_new[idx], agent, self.problem.sense):
                    self.population[idx] = cy.duplicate_agent(pop_new[idx])
                else:
                    # Compute difference according to problem type
                    delta = np.abs(
                        pop_new[idx].fitness - self.population[idx].fitness
                    )
                    p = np.exp(-delta / self.dyn_t)  # Compute Acceptance Probability
                    if self.generator.uniform() <= p:  # Accept / Reject
                        self.population[idx] = cy.duplicate_agent(pop_new[idx])
        # Update Temperature
        self.dyn_t = self.t_damp * self.dyn_t
        self.dyn_sigma = self.mutation_step_size_damp * self.dyn_sigma

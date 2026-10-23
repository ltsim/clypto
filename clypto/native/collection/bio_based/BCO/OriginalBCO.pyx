#!/usr/bin/env python
# Created by "Thieu" at 12:24, 18/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalBCO(cy.Optimizer):
    """
    The original version of: Bacterial Colony Optimization (BCO)

    Links:
        1. https://ieeexplore.ieee.org/abstract/document/4475427

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_m (float): (0, 1) -> better [0.01, 0.2], Mutation probability
        + n_elites (int): (2, pop_size/2) -> better [2, 5], Number of elites will be keep for next generation

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.bio_based import BCO    >>> import numpy as np
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
    >>> model = BCO.OriginalBCO(epoch=1000, pop_size=50, p_m=0.01, n_elites=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Niu, B., & Wang, H. (2012). Bacterial colony optimization. Discrete dynamics in nature and society, 2012(1), 698057.
    """

    cdef public int c_max
    cdef public double c_min
    cdef public double energy_threshold
    cdef public int max_swim_steps
    cdef public double migration_prob
    cdef public int n_chemotaxis

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            c_min: float = 0.01,
            c_max: float = 0.2,
            n_chemotaxis: int = 1,
            max_swim_steps: int = 4,
            energy_threshold: float = 0.5,
            migration_prob: float = 0.1,
            **kwargs: object
    ) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 100
            c_min: Minimum chemotaxis step size
            c_max: Maximum chemotaxis step size
            n_chemotaxis: Nonlinear parameter for chemotaxis step
            max_swim_steps: Maximum swimming steps
            energy_threshold: Energy threshold for reproduction/elimination
            migration_prob: Migration probability
        """
        super().__init__(parameters=[ "epoch", "pop_size", "c_min", "c_max", "n_chemotaxis", "max_swim_steps", "energy_threshold", "migration_prob", ], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.c_min = cy.validator(float, c_min, (0.0, 1.0), "c_min")
        self.c_max = cy.validator(int, c_max, (c_min, 10.0), "c_max")
        self.n_chemotaxis = cy.validator(int, n_chemotaxis, [1, 5], "n_chemotaxis")
        self.max_swim_steps = cy.validator(int, max_swim_steps, (2, 10), "max_swim_steps")
        self.energy_threshold = cy.validator(float, energy_threshold, (0, 1.0), "energy_threshold")
        self.migration_prob = cy.validator(float, migration_prob, (0, 1.0), "migration_prob")

    def initialize_variables(self):
        pop_size = self.population.size()
        self.energy = self.generator.uniform(0, 1, pop_size)

    def initialization(self) -> None:
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        self.pop_local = self.population.copy()

    def get_energy(self, best, worst, fits):
        pop_size = self.population.size()
        if best.fitness == worst.fitness:
            norm_fit = np.zeros(pop_size)
        else:
            norm_fit = (fits - worst.fitness) / (
                    best.fitness - worst.fitness
            )
        # Energy inversely proportional to fitness (lower fitness = higher energy)
        return 1 - norm_fit

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch: The current iteration
        """
        pop_size = self.population.size()
        # Normalize fitness to [0, 1]
        ranked = self.population.sort()
        best = [cy.duplicate_agent(agent) for agent in ranked[:1]]
        worst = [cy.duplicate_agent(agent) for agent in ranked[::-1][:1]]
        fits = np.array([agent.fitness for agent in self.population])
        energy = self.get_energy(best[0], worst[0], fits)

        # Calculate adaptive chemotaxis step size
        step = (
                self.c_min
                + (self.c_max - self.c_min) * (1 - epoch / self.epoch) ** self.n_chemotaxis
        )
        pop = []
        ## Perform chemotaxis and communication
        for idx, agent in enumerate(self.population.toarray()):

            # Random factor for personal vs global best
            f_i = self.generator.random()
            personal_direction = self.pop_local[idx].solution - agent.solution
            global_direction = self.g_best.solution - agent.solution

            # Tumbling (with random turbulence)
            turbulent = self.generator.normal(0, 0.1, self.problem.n_dims)
            x = (
                    f_i * global_direction + (1 - f_i) * personal_direction + turbulent
            )
            for jdx in range(0, self.max_swim_steps):
                # Swimming (no turbulence)
                x = f_i * global_direction + (1 - f_i) * personal_direction
            x = agent.solution + step * x
            x = cy.correct_solution(self.problem, x)
            agent_new = self.population.create_agent(x)
            pop.append(agent_new)
            if self.mode == "sequential":
                agent_new.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent_new, sense=self.problem.sense)
        if self.mode != "sequential":
            pop = self.population.evaluate(pop, self.mode)
            self.population = self.population.greedy(pop)

        ## Perform interactive exchange between bacteria
        for idx, agent in enumerate(self.population.toarray()):
            exchange_type = self.generator.choice(["individual", "group"])
            if exchange_type == "individual":
                if self.generator.random() < 0.5:
                    # Dynamic neighbor oriented
                    if idx == 0:
                        neighbor = 1
                    elif idx == pop_size - 1:
                        neighbor = pop_size - 2
                    else:
                        neighbor = idx + 1 if self.generator.random() < 0.5 else idx - 1
                else:
                    # Random oriented
                    neighbor = self.generator.choice(
                        list(set(range(pop_size)) - {idx})
                    )

                # Exchange information if neighbor is better
                if cy.is_better(self.population[neighbor], agent, "min"):
                    self.population[idx] = self.population[neighbor]
            else:
                # Group exchange
                if cy.is_better(self.population[idx], self.g_best, "min"):
                    # Move towards global best
                    self.population[idx].solution += 0.1 * (
                            self.g_best.solution - self.population[idx].solution
                    )
        self.population = self.population.spawn(self.population.evaluate(self.population, self.mode))

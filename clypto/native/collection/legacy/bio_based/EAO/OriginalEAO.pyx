#!/usr/bin/env python
# Created by "Thieu" at 23:50, 28/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalEAO(cy.Optimizer):
    """
    The original version of: Enzyme Action Optimizer (EAO)

    Notes:
        + This algorithm used 3 fitness calculations for each update enzyme. Therefor, it is slower 3 times than other algorithms.

    Links:
        1. https://mathworks.com/matlabcentral/fileexchange/170296-enzyme-action-optimizer-a-novel-bio-inspired-optimization

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.bio_based import EAO    >>> import numpy as np
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
    >>> model = EAO.OriginalEAO(epoch=1000, pop_size=50, p_m=0.01, n_elites=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Rodan, A., Al-Tamimi, A. K., Al-Alnemer, L., Mirjalili, S., & Tiňo, P. (2025).
    Enzyme action optimizer: a novel bio-inspired optimization algorithm. The Journal of Supercomputing, 81(5), 686.
    """

    cdef public double ec

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, ec: float = 0.1, **kwargs: object
    ) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 100
            ec: Enzyme Concentration, default=0.1
        """
        super().__init__(parameters=["epoch", "pop_size", "ec"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.ec = cy.validator(float, ec, [0.0, 100], "ec")

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch: The current iteration
        """
        pop_size = self.population.size()
        # Adaptation Factor - tăng dần theo thời gian
        AF = np.sqrt(epoch / self.epoch)

        # Handle each enzyme
        for idx in range(pop_size):
            # 1. Update FirstSubstratePosition
            r1 = self.generator.random(size=self.problem.n_dims)
            pos1 = (self.g_best.solution - self.population[idx].solution) + r1 * np.sin(
                AF * self.population[idx].solution
            )
            pos1 = self.population.correct_solution(pos1)
            agent1 = self.population.generate_agent(pos1)

            # 2. Select 2 randoms
            j1, j2 = self.generator.choice(
                list(set(range(0, pop_size)) - {idx}), size=2, replace=False
            )

            ## Candidate A: vector-valued random factors
            scA1 = self.ec + (1 - self.ec) * self.generator.random(
                size=self.problem.n_dims
            )
            exA = AF * (
                    self.ec
                    + (1 - self.ec) * self.generator.random(size=self.problem.n_dims)
            )
            posA = (
                    self.population[idx].solution
                    + scA1 * (self.population[j1].solution - self.population[j2].solution)
                    + exA * (self.g_best.solution - self.population[idx].solution)
            )
            posA = self.population.correct_solution(posA)
            agentA = self.population.generate_agent(posA)

            ## Candidate B: scalar random factors
            scB1 = self.ec + (1 - self.ec) * self.generator.random()
            exB = AF * (self.ec + (1 - self.ec) * self.generator.random())
            posB = (
                    self.population[idx].solution
                    + scB1 * (self.population[j1].solution - self.population[j2].solution)
                    + exB * (self.g_best.solution - self.population[idx].solution)
            )
            posB = self.population.correct_solution(posB)
            agentB = self.population.generate_agent(posB)

            pop_new = [self.population[idx], agent1, agentA, agentB]
            self.population[idx] = cy.sort_agents(pop_new, self.problem.sense)[0].copy()

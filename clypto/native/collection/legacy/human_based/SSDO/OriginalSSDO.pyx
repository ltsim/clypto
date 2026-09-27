#!/usr/bin/env python
# Created by "Thieu" at 11:17, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class OriginalSSDOAgent(cy.Agent):
    cdef public object velocity
    cdef public object local_solution


cdef class OriginalSSDOPopulation(cy.Population):
    """Agents of :class:`OriginalSSDO`."""

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
        pos_local = solution.copy()
        return OriginalSSDOAgent(
            solution=solution, velocity=velocity, local_solution=pos_local
        )


cdef class OriginalSSDO(cy.Optimizer):
    """
    The original version of: Social Ski-Driver Optimization (SSDO)

    Links:
       1. https://doi.org/10.1007/s00521-019-04159-z
       2. https://www.mathworks.com/matlabcentral/fileexchange/71210-social-ski-driver-ssd-optimization-algorithm-2019

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import SSDO    >>> import numpy as np
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
    >>> model = SSDO.OriginalSSDO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Tharwat, A. and Gabel, T., 2020. Parameters optimization of support vector machines for imbalanced
    data using social ski driver algorithm. Neural Computing and Applications, 32(11), pp.6925-6938.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=OriginalSSDOPopulation)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0
        ## Calculate the mean of the best three solutions in each dimension. Eq 9
        ranked = self.population.sort()
        pop_best3 = [agent.copy() for agent in ranked[:3]]
        pos_mean = np.mean(np.array([agent.solution for agent in pop_best3]))
        pop_new = [agent.copy() for agent in self.population]
        # Updating velocity vectors
        r1 = self.generator.uniform()  # r1, r2 is a random number in [0,1]
        r2 = self.generator.uniform()
        for i in range(0, pop_size):
            if r2 <= 0.5:  ## Use Sine function to move
                vel_new = c * np.sin(r1) * (
                    self.population[i].local_solution - self.population[i].solution
                ) + (2 - c) * np.sin(r1) * (pos_mean - self.population[i].solution)
            else:  ## Use Cosine function to move
                vel_new = c * np.cos(r1) * (
                    self.population[i].local_solution - self.population[i].solution
                ) + (2 - c) * np.cos(r1) * (pos_mean - self.population[i].solution)
            pop_new[i].velocity = vel_new
        ## Reproduction
        for idx in range(0, pop_size):
            pos_new = (
                self.generator.normal(0, 1, self.problem.n_dims) * pop_new[idx].solution
                + self.generator.random() * pop_new[idx].velocity
            )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            agent.local_solution = self.population[idx].solution.copy()
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, pop_new[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

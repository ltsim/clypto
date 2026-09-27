#!/usr/bin/env python
# Created by "Thieu" at 17:19, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalSPBO(cy.Optimizer):
    """
    The original version of: Student Psychology Based Optimization (SPBO)

    Notes:
        1. This algorithm is a weak algorithm in solving several problems
        2. It also consumes too much time because of ndim * pop_size updating times.

    Links:
       1. https://www.sciencedirect.com/science/article/abs/pii/S0965997820301484
       2. https://www.mathworks.com/matlabcentral/fileexchange/80991-student-psycology-based-optimization-spbo-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import SPBO    >>> import numpy as np
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
    >>> model = SPBO.OriginalSPBO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Das, B., Mukherjee, V., & Das, D. (2020). Student psychology based optimization algorithm: A new population based
    optimization algorithm for solving optimization problems. Advances in Engineering software, 146, 102804.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        for jdx in range(0, self.problem.n_dims):
            fitness = self.population.fitness
            idx_best = int(np.argmin(fitness) if self.problem.sense == "min" else np.argmax(fitness))
            mid = self.generator.integers(1, pop_size - 1)
            x_mean = np.mean([agent.solution for agent in self.population], axis=0)
            pop_new = []
            for idx in range(0, pop_size):
                if idx == idx_best:
                    k = self.generator.choice([1, 2])
                    j = self.generator.choice(
                        list(set(range(0, pop_size)) - {idx})
                    )
                    new_pos = self.g_best.solution + (-1) ** k * self.generator.random(
                        self.problem.n_dims
                    ) * (self.g_best.solution - self.population[j].solution)
                elif idx < mid:
                    ## Good Student
                    if self.generator.random() > self.generator.random():
                        new_pos = self.g_best.solution + self.generator.random(
                            self.problem.n_dims
                        ) * (self.g_best.solution - self.population[idx].solution)
                    else:
                        new_pos = (
                                self.population[idx].solution
                                + self.generator.random(self.problem.n_dims)
                                * (self.g_best.solution - self.population[idx].solution)
                                + self.generator.random()
                                * (self.population[idx].solution - x_mean)
                        )
                else:
                    ## Average Student
                    if self.generator.random() > self.generator.random():
                        new_pos = self.population[idx].solution + self.generator.random(
                            self.problem.n_dims
                        ) * (x_mean - self.population[idx].solution)
                    else:
                        new_pos = self.problem.generate_solution()
                new_pos = self.population.correct_solution(new_pos)
                agent = self.population.create_agent(new_pos)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.population.evaluate(pop_new, self.mode)
                self.population = self.population.greedy(pop_new)

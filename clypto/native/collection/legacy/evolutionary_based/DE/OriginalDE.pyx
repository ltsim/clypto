#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class OriginalDE(cy.Optimizer):
    """
    The original version of: Differential Evolution (DE)

    Links:
        1. https://doi.org/10.1016/j.swevo.2018.10.006

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wf (float): [-1., 1.0], weighting factor, default = 0.1
        + cr (float): [0.5, 0.95], crossover rate, default = 0.9
        + strategy (int): [0, 5], there are lots of variant version of DE algorithm,
            + 0: DE/current-to-rand/1/bin
            + 1: DE/best/1/bin
            + 2: DE/best/2/bin
            + 3: DE/rand/2/bin
            + 4: DE/current-to-best/1/bin
            + 5: DE/current-to-rand/1/bin

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.evolutionary_based import DE    >>> import numpy as np
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
    >>> model = DE.OriginalDE(epoch=1000, pop_size=50, wf = 0.7, cr = 0.9, strategy = 0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Jambi, K.M., 2019. Novel mutation strategy for enhancing SHADE and
    LSHADE algorithms for global numerical optimization. Swarm and Evolutionary Computation, 50, p.100455.
    """

    cdef public double cr
    cdef public int strategy
    cdef public double wf

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        wf: float = 0.1,
        cr: float = 0.9,
        strategy: int = 0,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wf (float): weighting factor, default = 0.1
            cr (float): crossover rate, default = 0.9
            strategy (int): Different variants of DE, default = 0
        """
        super().__init__(parameters=["epoch", "pop_size", "wf", "cr", "strategy"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.wf = cy.validator(float, wf, (-3.0, 3.0), "wf")
        self.cr = cy.validator(float, cr, (0, 1.0), "cr")
        self.strategy = cy.validator(int, strategy, [0, 5], "strategy")

    def mutation__(self, current_pos, new_pos):
        condition = self.generator.random(self.problem.n_dims) < self.cr
        pos_new = np.where(condition, new_pos, current_pos)
        return self.population.correct_solution(pos_new)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop = []
        if self.strategy == 0:
            # Choose 3 random element and different to i
            for idx in range(0, pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx}), 3, replace=False
                )
                pos_new = self.population[idx_list[0]].solution + self.wf * (
                    self.population[idx_list[1]].solution - self.population[idx_list[2]].solution
                )
                pos_new = self.mutation__(self.population[idx].solution, pos_new)
                agent = self.population.create_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        elif self.strategy == 1:
            for idx in range(0, pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx}), 2, replace=False
                )
                pos_new = self.g_best.solution + self.wf * (
                    self.population[idx_list[0]].solution - self.population[idx_list[1]].solution
                )
                pos_new = self.mutation__(self.population[idx].solution, pos_new)
                agent = self.population.create_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        elif self.strategy == 2:
            for idx in range(0, pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx}), 4, replace=False
                )
                pos_new = (
                    self.g_best.solution
                    + self.wf
                    * (self.population[idx_list[0]].solution - self.population[idx_list[1]].solution)
                    + self.wf
                    * (self.population[idx_list[2]].solution - self.population[idx_list[3]].solution)
                )
                pos_new = self.mutation__(self.population[idx].solution, pos_new)
                agent = self.population.create_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        elif self.strategy == 3:
            for idx in range(0, pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx}), 5, replace=False
                )
                pos_new = (
                    self.population[idx_list[0]].solution
                    + self.wf
                    * (self.population[idx_list[1]].solution - self.population[idx_list[2]].solution)
                    + self.wf
                    * (self.population[idx_list[3]].solution - self.population[idx_list[4]].solution)
                )
                pos_new = self.mutation__(self.population[idx].solution, pos_new)
                agent = self.population.create_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        elif self.strategy == 4:
            for idx in range(0, pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx}), 2, replace=False
                )
                pos_new = (
                    self.population[idx].solution
                    + self.wf * (self.g_best.solution - self.population[idx].solution)
                    + self.wf
                    * (self.population[idx_list[0]].solution - self.population[idx_list[1]].solution)
                )
                pos_new = self.mutation__(self.population[idx].solution, pos_new)
                agent = self.population.create_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        else:
            for idx in range(0, pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx}), 3, replace=False
                )
                pos_new = (
                    self.population[idx].solution
                    + self.wf
                    * (self.population[idx_list[0]].solution - self.population[idx].solution)
                    + self.wf
                    * (self.population[idx_list[1]].solution - self.population[idx_list[2]].solution)
                )
                pos_new = self.mutation__(self.population[idx].solution, pos_new)
                agent = self.population.create_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.population.evaluate(pop, self.mode)
            self.population = self.population.greedy(pop)

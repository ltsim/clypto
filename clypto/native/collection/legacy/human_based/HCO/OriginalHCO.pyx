#!/usr/bin/env python
# Created by "Thieu" at 08:57, 12/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalHCO(cy.Optimizer):
    """
    The original version of: Human Conception Optimizer (HCO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/124200-human-conception-optimizer-hco
        2. https://www.nature.com/articles/s41598-022-25031-6

    Notes:
        1. This algorithm shares some similarities with the PSO algorithm (equations)
        2. The implementation of Matlab code is kinda different to the paper

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wfp (float): (0, 1.) - weight factor for probability of fitness selection, default=0.65
        + wfv (float): (0, 1.0) - weight factor for velocity update stage, default=0.05
        + c1 (float): (0., 3.0) - acceleration coefficient, same as PSO, default=1.4
        + c2 (float): (0., 3.0) - acceleration coefficient, same as PSO, default=1.4

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import HCO    >>> import numpy as np
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
    >>> model = HCO.OriginalHCO(epoch=1000, pop_size=50, wfp=0.65, wfv=0.05, c1=1.4, c2=1.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Acharya, D., & Das, D. K. (2022). A novel Human Conception Optimizer for solving optimization problems. Scientific Reports, 12(1), 21631.
    """

    cdef public double c1
    cdef public double c2
    cdef public double wfp
    cdef public double wfv

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            wfp: float = 0.65,
            wfv: float = 0.05,
            c1: float = 1.4,
            c2: float = 1.4,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wfp (float): weight factor for probability of fitness selection, default=0.65
            wfv (float): weight factor for velocity update stage, default=0.05
            c1 (float): acceleration coefficient, same as PSO, default=1.4
            c2 (float): acceleration coefficient, same as PSO, default=1.4
        """
        super().__init__(parameters=["epoch", "pop_size", "wfp", "wfv", "c1", "c2"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.wfp = cy.validator(float, wfp, [0, 1.0], "wfp")
        self.wfv = cy.validator(float, wfv, [0, 1.0], "wfv")
        self.c1 = cy.validator(float, c1, [0.0, 100.0], "c1")
        self.c2 = cy.validator(float, c2, [1.0, 100.0], "c2")

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        pop_op = []
        for idx in range(0, pop_size):
            pos_new = self.problem.bounds.up + self.problem.bounds.low - self.population[idx].solution
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_op.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_op = self.population.evaluate(pop_op, self.mode)
            self.population = self.population.greedy(pop_op)
        ranked = self.population.sort()
        (best,) = [agent.copy() for agent in ranked[:1]]
        (worst,) = [agent.copy() for agent in ranked[::-1][:1]]
        pfit = (
                       worst.fitness - best.fitness
               ) * self.wfp + best.fitness
        for idx in range(0, pop_size):
            if cy.better_fitness(pfit, self.population[idx].fitness, self.problem.sense):
                while True:
                    agent = self.population.generate_agent()
                    if cy.better_fitness(agent.fitness, pfit, self.problem.sense):
                        self.population[idx] = agent
                        break
        self.vec = self.generator.uniform(
            self.problem.bounds.low, self.problem.bounds.up, (pop_size, self.problem.n_dims)
        )
        self.pop_p = [agent.copy() for agent in self.population]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        lamda = self.generator.random()
        neu = 2
        fits = np.array([agent.fitness for agent in self.population])
        fit_mean = np.mean(fits)
        RR = (self.g_best.fitness - fits) ** 2
        rr = (fit_mean - fits) ** 2
        ll = RR - rr
        LL = self.g_best.fitness - fit_mean
        VV = lamda * (ll / (4 * neu * LL))
        pop_new = []
        for idx in range(0, pop_size):
            a1 = self.pop_p[idx].solution - self.population[idx].solution
            a2 = self.g_best.solution - self.population[idx].solution
            self.vec[idx] = (
                    self.wfv * (VV[idx] + self.vec[idx])
                    + self.c1 * a1 * np.sin(2 * np.pi * epoch / self.epoch)
                    + self.c2 * a2 * np.sin(2 * np.pi * epoch / self.epoch)
            )
            pos_new = self.population[idx].solution + self.vec[idx]
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].evaluate(self.problem)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)

        for idx in range(0, pop_size):
            if cy.is_better(pop_new[idx], self.population[idx], self.problem.sense):
                self.population[idx] = pop_new[idx].copy()
                if cy.is_better(pop_new[idx], self.pop_p[idx], self.problem.sense):
                    self.pop_p[idx] = pop_new[idx].copy()

#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevGCO(cy.Optimizer):
    """
    The developed version: Germinal Center Optimization (GCO)

    Notes:
        + The global best solution and 2 random solutions are used instead of randomizing 3 solutions

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + cr (float): [0.5, 0.95], crossover rate, default = 0.7 (Same as DE algorithm)
        + wf (float): [1.0, 2.0], weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.system_based import GCO    >>> import numpy as np
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
    >>> model = GCO.DevGCO(epoch=1000, pop_size=50, cr = 0.7, wf = 1.25)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            cr: float = 0.7,
            wf: float = 1.25,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            cr (float): crossover rate, default = 0.7 (Same as DE algorithm)
            wf (float): weighting factor (f in the paper), default = 1.25 (Same as DE algorithm)
        """
        super().__init__(parameters=["epoch", "pop_size", "cr", "wf"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.cr = cy.validator(float, cr, (0, 1.0), "cr")
        self.wf = cy.validator(float, wf, (0, 3.0), "wf")

    def initialize_variables(self):
        pop_size = self.population.size()
        self.dyn_list_cell_counter = np.ones(pop_size)  # CEll Counter
        self.dyn_list_life_signal = 70 * np.ones(
            pop_size
        )  # 70% to duplicate, and 30% to die  # LIfe-Signal

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Dark-zone process    (can be parallelization)
        pop_new = []
        for idx in range(0, pop_size):
            if self.generator.uniform(0, 100) < self.dyn_list_life_signal[idx]:
                self.dyn_list_cell_counter[idx] += 1
            else:
                self.dyn_list_cell_counter[idx] = 1
            # Mutate process
            r1, r2 = self.generator.choice(
                list(set(range(0, pop_size)) - {idx}), 2, replace=False
            )
            pos_new = self.g_best.solution + self.wf * (
                    self.population[r2].solution - self.population[r1].solution
            )
            condition = self.generator.random(self.problem.n_dims) < self.cr
            pos_new = np.where(condition, pos_new, self.population[idx].solution)
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].evaluate(self.problem)
        pop_new = self.population.evaluate(pop_new, self.mode)
        for idx in range(0, pop_size):
            if cy.is_better(pop_new[idx], self.population[idx], self.problem.sense):
                self.dyn_list_cell_counter[idx] += 10
                self.population[idx] = pop_new[idx].copy()
        ## Light-zone process   (no needs parallelization)
        for idx in range(0, pop_size):
            self.dyn_list_cell_counter[idx] = 10
            fit_list = np.array([agent.fitness for agent in self.population])
            fit_max = np.max(fit_list)
            fit_min = np.min(fit_list)
            self.dyn_list_cell_counter[idx] += (
                    10
                    * (self.population[idx].fitness - fit_max)
                    / (fit_min - fit_max + self.EPSILON)
            )

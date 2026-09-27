#!/usr/bin/env python
# Created by "Thieu" at 19:38, 10/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.stats import qmc
cimport clypto.core as cy



cdef class OriginalPSS(cy.Optimizer):
    """
    The original version of: Pareto-like Sequential Sampling (PSS)

    Links:
        1. https://doi.org/10.1007/s00500-021-05853-8
        2. https://github.com/eesd-epfl/pareto-optimizer

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + acceptance_rate (float): [0.7-0.96], the probability of accepting a solution in the normal range, default=0.9
        + sampling_method (str): 'LHS': Latin-Hypercube or 'MC': 'MonteCarlo', default="LHS"

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.math_based import PSS    >>> import numpy as np
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
    >>> model = PSS.OriginalPSS(epoch=1000, pop_size=50, acceptance_rate = 0.8, sampling_method = "LHS")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Shaqfa, M. and Beyer, K., 2021. Pareto-like sequential sampling heuristic for global optimisation. Soft Computing, 25(14), pp.9077-9096.
    """

    cdef public double acceptance_rate
    cdef public str sampling_method

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            acceptance_rate: float = 0.9,
            sampling_method: str = "LHS",
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            acceptance_rate (float): the probability of accepting a solution in the normal range, default = 0.9
            sampling_method (str): 'LHS': Latin-Hypercube or 'MC': 'MonteCarlo', default = "LHS"
        """
        super().__init__(parameters=["epoch", "pop_size", "acceptance_rate", "sampling_method"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.acceptance_rate = cy.validator(float, acceptance_rate, (0, 1.0), "acceptance_rate")
        self.sampling_method = cy.validator(str, sampling_method, ["MC", "LHS"], "sampling_method")

    def initialize_variables(self):
        self.step = 10e-10
        self.steps = np.ones(self.problem.n_dims) * self.step
        self.new_solution = True

    def create_population(self, pop_size=None):
        if self.sampling_method == "MC":
            pop = self.generator.random(self.population.size(), self.problem.n_dims)
        else:  # Default: "LHS"
            sampler = qmc.LatinHypercube(d=self.problem.n_dims)
            pop = sampler.random(n=pop_size)
        return pop

    def initialization(self):
        pop_size = self.population.size()
        lb_pop = np.repeat(np.reshape(self.problem.bounds.low, (1, -1)), pop_size, axis=0)
        ub_pop = np.repeat(np.reshape(self.problem.bounds.up, (1, -1)), pop_size, axis=0)
        steps_mat = np.repeat(np.reshape(self.steps, (1, -1)), pop_size, axis=0)
        random_pop = self.create_population(pop_size)
        pop = (
                np.round((lb_pop + random_pop * (ub_pop - lb_pop)) / steps_mat) * steps_mat
        )
        self.population = self.population.spawn([])
        for pos in pop:
            pos_new = cy.correct_solution(self.problem, pos)
            agent = self.population.generate_agent(pos_new)
            self.population.append(agent)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        pop_rand = self.create_population(pop_size)
        for idx, agent in enumerate(self.population.toarray()):
            x = agent.solution.copy()
            for k in range(self.problem.n_dims):
                # Update the ranges
                deviation = self.generator.uniform(
                    min(0, self.g_best.solution[k]), max(0, self.g_best.solution[k])
                )
                if self.new_solution:
                    # The deviation is positive dynamic real number
                    deviation = abs(
                        0.5
                        * (1.0 - self.acceptance_rate)
                        * (self.problem.bounds.up[k] - self.problem.bounds.low[k])
                    ) * (1 - (epoch / self.epoch))
                reduced_lb = self.g_best.solution[k] - deviation
                reduced_lb = np.amax([reduced_lb, self.problem.bounds.low[k]])
                reduced_ub = reduced_lb + deviation * 2.0
                reduced_ub = np.amin([reduced_ub, self.problem.bounds.up[k]])
                # Choose new solution
                if self.generator.random() <= self.acceptance_rate:
                    # choose a solution from the prominent domain
                    x[k] = reduced_lb + pop_rand[idx, k] * (
                            reduced_ub - reduced_lb
                    )
                else:
                    # choose a solution from the overall domain
                    x[k] = self.problem.bounds.low[k] + pop_rand[idx, k] * (
                            self.problem.bounds.up[k] - self.problem.bounds.low[k]
                    )
                # Round for the step size
                x = np.round(x / self.steps) * self.steps
            # Check the bound
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                n_population[-1].evaluate(self.problem)
        self.population = self.population.spawn(self.population.evaluate(n_population, self.mode))
        current_best = cy.duplicate_agent(cy.sort_agents(n_population, self.problem.sense)[0])
        if cy.is_better(current_best, self.g_best, self.problem.sense):
            self.new_solution = True
        else:
            self.new_solution = False

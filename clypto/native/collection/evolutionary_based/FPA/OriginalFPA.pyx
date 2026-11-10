#!/usr/bin/env python
# Created by "Thieu" at 19:34, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy


cdef object repair(object problem, object solution):
    """Out-of-bounds values come from a new random solution, then the problem's own correction."""
    condition = np.logical_and(problem.bounds.low <= solution, solution <= problem.bounds.up)
    return problem.correct_solution(np.where(condition, solution, problem.generate_solution()))


cdef class OriginalFPA(cy.Optimizer):
    """
    The original version of: Flower Pollination Algorithm (FPA)

    Links:
        1. https://doi.org/10.1007/978-3-642-32894-7_27

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_s (float): [0.5, 0.95], switch probability, default = 0.8
        + levy_multiplier: [0.0001, 1000], mutiplier factor of Levy-flight trajectory, depends on the problem

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.evolutionary_based import FPA    >>> import numpy as np
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
    >>> model = FPA.OriginalFPA(epoch=1000, pop_size=50, p_s = 0.8, levy_multiplier = 0.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2012, September. Flower pollination algorithm for global optimization. In International
    conference on unconventional computing and natural computation (pp. 240-249). Springer, Berlin, Heidelberg.
    """

    cdef public double levy_multiplier
    cdef public double p_s

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            p_s: float = 0.8,
            levy_multiplier: float = 0.1,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_s (float): switch probability, default = 0.8
            levy_multiplier (float): multiplier factor of Levy-flight trajectory, default = 0.2
        """
        super().__init__(parameters=["epoch", "pop_size", "p_s", "levy_multiplier"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000])
        self.p_s = cy.validator(float, p_s, (0, 1.0), "p_s")
        self.levy_multiplier = cy.validator(float, levy_multiplier, (-10000, 10000), "levy_multiplier")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop = []
        for idx, agent in enumerate(self.population.toarray()):
            if self.generator.uniform() < self.p_s:
                levy = cy.levy_flight(self.generator, beta=1.0, multiplier=self.levy_multiplier, size=self.problem.n_dims, case=-1)
                x = agent.solution + 1.0 / np.sqrt(epoch) * levy * (
                        agent.solution - self.g_best.solution
                )
            else:
                id1, id2 = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx}), 2, replace=False
                )
                x = agent.solution + self.generator.uniform() * (
                        self.population[id1].solution - self.population[id2].solution
                )
            x = repair(self.problem, x)
            child = self.population.create_agent(x)
            pop.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop = self.population.evaluate(pop, self.mode)
            self.population = self.population.greedy(pop)

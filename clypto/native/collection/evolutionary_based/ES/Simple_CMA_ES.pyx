#!/usr/bin/env python
# Created by "Thieu" at 18:14, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class Simple_CMA_ES(cy.Optimizer):
    """
    The simple version of: Covariance Matrix Adaptation Evolution Strategy (Simple-CMA-ES)

    Links:
        1. Inspired from this version: https://github.com/jenkspt/CMA-ES
        2. https://ieeexplore.ieee.org/abstract/document/6790628/

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.evolutionary_based import ES    >>> import numpy as np
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
    >>> model = ES.Simple_CMA_ES(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. Evolutionary computation, 9(2), 159-195.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    def before_main_loop(self):
        pop_size = self.population.size()
        self.mu = int(np.round(pop_size / 2))

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pos_list = np.array([agent.solution for agent in self.population]).T
        pop_sorted = self.population.sort()
        pos_topk = np.array([agent.solution for agent in pop_sorted[: self.mu]]).T
        # Covariance of top k but using mean of entire population
        centered = pos_list - pos_topk.mean(1, keepdims=True)
        C = (centered @ centered.T) / (self.mu - 1)
        # Eigenvalue decomposition
        w, E = np.linalg.eigh(C)
        if np.any(np.diag(w) < 0):
            w[w < 0] = 0
        # Generate new population
        # Sample from multivariate gaussian with mean of topk
        N = self.generator.normal(size=(self.problem.n_dims, pop_size))
        X = pos_topk.mean(1, keepdims=True) + (E @ np.diag(np.sqrt(w)) @ N)
        X = X.T
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx in range(0, pop_size):
            x = cy.correct_solution(self.problem, X[idx])
            agent = self.population.create_agent(x)
            n_population.append(agent)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)

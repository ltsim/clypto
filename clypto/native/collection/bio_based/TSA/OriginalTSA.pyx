#!/usr/bin/env python
# Created by "Thieu" at 17:23, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalTSA(cy.Optimizer):
    """
    The original version: Tunicate Swarm Algorithm (TSA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0952197620300385?via%3Dihub
        2. https://www.mathworks.com/matlabcentral/fileexchange/75182-tunicate-swarm-algorithm-tsa

    Notes:
        1. This algorithm has some limitations
        2. The paper has several wrong equations in algorithm
        3. The implementation in Matlab code has some difference to the paper
        4. This algorithm shares some similarities with the Barnacles Mating Optimizer (BMO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.bio_based import TSA    >>> import numpy as np
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
    >>> model = TSA.OriginalTSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Kaur, S., Awasthi, L. K., Sangal, A. L., & Dhiman, G. (2020). Tunicate Swarm Algorithm: A new bio-inspired
    based metaheuristic paradigm for global optimization. Engineering Applications of Artificial Intelligence, 90, 103541.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pmin, pmax = 1, 4
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            c3 = self.generator.random(self.problem.n_dims)
            c2 = self.generator.random(self.problem.n_dims)
            c1 = self.generator.random(self.problem.n_dims)
            M = np.fix(pmin + self.generator.random() * (pmax - pmin))
            A = (c2 + c3 - 2 * c1) / M
            t1 = self.g_best.solution + A * np.abs(
                self.g_best.solution - c2 * agent.solution
            )
            t2 = self.g_best.solution - A * np.abs(
                self.g_best.solution - c2 * agent.solution
            )
            x = np.where(c3 >= 0.5, t1, t2)
            if idx != 0:
                x = (x + self.population[idx - 1].solution) / 2
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                n_population[-1].evaluate(self.problem)
        self.population = self.population.spawn(self.population.evaluate(n_population, self.mode))

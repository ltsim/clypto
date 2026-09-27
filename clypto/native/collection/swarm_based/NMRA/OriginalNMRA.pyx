cimport clypto.core as cy
#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%



cdef class OriginalNMRA(cy.Optimizer):
    """
    The original version of: Naked Mole-Rat Algorithm (NMRA)

    Links:
        1. https://www.doi.org10.1007/s00521-019-04464-7

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.5, 0.95], probability of breeding, default = 0.75

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import NMRA    >>> import numpy as np
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
    >>> model = NMRA.OriginalNMRA(epoch=1000, pop_size=50, pb = 0.75)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Salgotra, R. and Singh, U., 2019. The naked mole-rat algorithm. Neural Computing and Applications, 31(12), pp.8837-8857.
    """

    cdef public double pb

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            pb: float = 0.75,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pb (float): probability of breeding, default = 0.75
        """
        super().__init__(parameters=["epoch", "pop_size", "pb"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.pb = cy.validator(float, pb, (0, 1.0), "pb")
        self.size_b = int(self.population.size() / 5)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            x = agent.solution.copy()
            if idx < self.size_b:  # breeding operators
                if self.generator.uniform() < self.pb:
                    alpha = self.generator.uniform()
                    x = (1 - alpha) * agent.solution + alpha * (
                            self.g_best.solution - agent.solution
                    )
            else:  # working operators
                t1, t2 = self.generator.choice(
                    range(self.size_b, pop_size), 2, replace=False
                )
                x = agent.solution + self.generator.uniform() * (
                        self.population[t1].solution - self.population[t2].solution
                )
            x = cy.correct_solution(self.problem, x)
            child = self.population.generate_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], child, self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)

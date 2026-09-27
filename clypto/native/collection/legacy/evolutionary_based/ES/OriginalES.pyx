#!/usr/bin/env python
# Created by "Thieu" at 18:14, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class OriginalESAgent(cy.Agent):
    cdef public object strategy


cdef class OriginalESPopulation(cy.Population):
    """Agents of :class:`OriginalES`."""
    cdef public object distance

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        strategy = self.generator.uniform(0, self.distance)
        return OriginalESAgent(solution=solution, strategy=strategy)


cdef class OriginalES(cy.Optimizer):
    """
    The original version of: Evolution Strategies (ES)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): [0.5, 1.0], Percentage of child agents evolving in the next generation

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.evolutionary_based import ES    >>> import numpy as np
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
    >>> model = ES.OriginalES(epoch=1000, pop_size=50, lamda = 0.75)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Beyer, H.G. and Schwefel, H.P., 2002. Evolution strategies–a comprehensive introduction. Natural computing, 1(1), pp.3-52.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        lamda: float = 0.75,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            lamda (float): Percentage of child agents evolving in the next generation, default=0.75
        """
        super().__init__(parameters=["epoch", "pop_size", "lamda"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=OriginalESPopulation)
        self.lamda = cy.validator(float, lamda, (0, 1.0), "lamda")
        self.n_child = int(self.lamda * self.population.size())

    def initialize_variables(self):
        self.distance = 0.05 * (self.problem.bounds.up - self.problem.bounds.low)
        self.population.distance = self.distance

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        child = []
        for idx in range(0, self.n_child):
            pos_new = self.population[idx].solution + self.population[
                idx
            ].strategy * self.generator.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.population.correct_solution(pos_new)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** (-1.0)
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** (-1.0)
            strategy = np.exp(
                tau_p * self.generator.normal(0, 1.0, self.problem.n_dims)
                + tau * self.generator.normal(0, 1.0, self.problem.n_dims)
            )
            agent = self.population.create_agent(pos_new)
            agent.update(solution=pos_new, strategy=strategy)
            child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                child[-1].evaluate(self.problem)
        child = self.population.evaluate(child, self.mode)
        self.population = cy.sort_agents(child + self.population, self.problem.sense)[:pop_size]
